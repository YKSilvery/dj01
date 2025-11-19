#include "aim_auto/serial_port.hpp"
#include "aim_auto/serial_protocol.hpp"

#include <aimbot_msgs/msg/detection_array.hpp>
#include <aimbot_msgs/msg/serial_feedback.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace
{
constexpr double kNanosecondsToMilliseconds = 1e-6;
constexpr double kNanosecondsToSeconds = 1e-9;
}

namespace aim_auto
{

struct TargetKinematics
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double yaw = 0.0;
  std::uint64_t timestamp = 0U;
  int class_id = -1;
};

class SerialBridgeNode : public rclcpp::Node
{
public:
  SerialBridgeNode()
  : Node("serial_bridge_node"), serial_port_()
  {
    const auto device = this->declare_parameter<std::string>("serial_device", "/dev/ttyUSB0");
    const int baud_rate = this->declare_parameter<int>("baud_rate", 115200);
    const int data_bits = this->declare_parameter<int>("data_bits", 8);
    const int stop_bits = this->declare_parameter<int>("stop_bits", 1);
    const std::string parity = this->declare_parameter<std::string>("parity", "n");
    tracker_topic_ = this->declare_parameter<std::string>("tracker_topic", "tracked_detections");
    gimbal_yaw_topic_ = this->declare_parameter<std::string>("gimbal_yaw_topic", "gimbal_yaw");
    gimbal_pitch_topic_ = this->declare_parameter<std::string>("gimbal_pitch_topic", "gimbal_pitch");
    serial_feedback_topic_ = this->declare_parameter<std::string>("serial_feedback_topic", "serial_feedback");
    status_code_ = static_cast<std::uint8_t>(this->declare_parameter<int>("status_code", 5));
    allround_mode_ = this->declare_parameter<bool>("allround_mode", false);
    message_hold_threshold_ = this->declare_parameter<int>("message_hold_threshold", 5);
    const double read_rate_hz = this->declare_parameter<double>("read_rate_hz", 200.0);

    if (!serial_port_.open(device, baud_rate, data_bits, stop_bits, parity.empty() ? 'n' : parity.front())) {
      RCLCPP_FATAL(get_logger(), "Failed to open serial device %s", device.c_str());
      throw std::runtime_error("serial open failed");
    }
    RCLCPP_INFO(get_logger(), "Serial device %s opened at %d baud", device.c_str(), baud_rate);

    detection_sub_ = create_subscription<aimbot_msgs::msg::DetectionArray>(
      tracker_topic_, rclcpp::SensorDataQoS(),
      std::bind(&SerialBridgeNode::detectionCallback, this, std::placeholders::_1));

    if (!gimbal_yaw_topic_.empty()) {
      gimbal_yaw_pub_ = create_publisher<std_msgs::msg::Float32>(gimbal_yaw_topic_, 10);
    }
    if (!gimbal_pitch_topic_.empty()) {
      gimbal_pitch_pub_ = create_publisher<std_msgs::msg::Float32>(gimbal_pitch_topic_, 10);
    }
    if (!serial_feedback_topic_.empty()) {
      serial_feedback_pub_ = create_publisher<aimbot_msgs::msg::SerialFeedback>(serial_feedback_topic_, 10);
    }

    if (read_rate_hz > 0.0) {
      const auto period = std::chrono::duration<double>(1.0 / read_rate_hz);
      read_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&SerialBridgeNode::pollSerial, this));
    }
  }

  ~SerialBridgeNode() override
  {
    serial_port_.close();
  }

private:
  void detectionCallback(const aimbot_msgs::msg::DetectionArray::SharedPtr msg)
  {
    if (msg == nullptr) {
      return;
    }

    RCLCPP_INFO(get_logger(), "Received DetectionArray with %zu detections, timestamp: %lu", msg->detections.size(), msg->timestamp);

    current_latency_ms_ = computeLatencyMs(msg->timestamp);
    const auto selected = selectDetection(*msg);

    MessData payload;
    payload.head = kVisionFrameHead;
    payload.tail = kVisionFrameTail;
    payload.status = status_code_;
    payload.allround = allround_mode_;
    payload.latency = static_cast<float>(current_latency_ms_);

    bool has_valid_detection = selected.has_value();
    if (has_valid_detection) {
      RCLCPP_INFO(get_logger(), "Selected detection: class_id=%d, x=%.3f, y=%.3f, z=%.3f", selected->class_id, selected->x, selected->y, selected->z);
      fillFromDetection(*selected, payload);
      updateCrc(payload);
      last_valid_message_ = payload;
      has_last_valid_message_ = true;
      loss_count_ = 0;
    } else {
      RCLCPP_INFO(get_logger(), "No valid detection selected");
      ++loss_count_;
      if (has_last_valid_message_ && loss_count_ <= message_hold_threshold_) {
        payload = last_valid_message_;
        payload.crc = 0;
      } else {
        payload = MessData{};
        payload.head = kVisionFrameHead;
        payload.tail = kVisionFrameTail;
        payload.status = status_code_;
        payload.allround = allround_mode_;
        payload.latency = static_cast<float>(current_latency_ms_);
      }
    }

    RCLCPP_INFO(get_logger(), "Sending MessData: yaw=%.3f, pitch=%.3f, latency=%.3f ms", payload.yaw, payload.pitch, payload.latency);
    RCLCPP_INFO(get_logger(), "Payload head: %02X, tail: %02X", payload.head, payload.tail);
    RCLCPP_INFO(get_logger(), "Payload crc: %04X", payload.crc);
    sendFrame(payload);
  }

  void pollSerial()
  {
    if (!serial_port_.isOpen()) {
      return;
    }

    std::array<std::uint8_t, kVisionFrameSize> buffer{};
    const std::size_t bytes = serial_port_.read(buffer.data(), buffer.size());
    if (bytes == 0) {
      return;
    }

    RCLCPP_INFO(get_logger(), "Read %zu bytes from serial", bytes);
    read_accumulator_.insert(read_accumulator_.end(), buffer.begin(), buffer.begin() + bytes);

    while (read_accumulator_.size() >= kVisionFrameSize) {
      if (read_accumulator_.front() != kVisionFrameHead) {
        read_accumulator_.erase(read_accumulator_.begin());
        continue;
      }
      if (read_accumulator_.size() < kVisionFrameSize) {
        break;
      }
      MessData inbound{};
      std::memcpy(&inbound, read_accumulator_.data(), kVisionFrameSize);
      read_accumulator_.erase(read_accumulator_.begin(), read_accumulator_.begin() + kVisionFrameSize);
      if (inbound.tail != kVisionFrameTail) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Invalid tail byte from serial");
        continue;
      }
      RCLCPP_INFO(get_logger(), "Received valid MessData frame from serial: yaw=%.3f, pitch=%.3f", inbound.yaw, inbound.pitch);
      handleFeedback(inbound);
    }
  }

  void handleFeedback(const MessData & message)
  {
    RCLCPP_INFO(get_logger(), "Handling feedback: publishing yaw=%.3f, pitch=%.3f", message.yaw, message.pitch);
    if (gimbal_yaw_pub_) {
      std_msgs::msg::Float32 yaw_msg;
      yaw_msg.data = message.yaw;
      gimbal_yaw_pub_->publish(yaw_msg);
    }
    if (gimbal_pitch_pub_) {
      std_msgs::msg::Float32 pitch_msg;
      pitch_msg.data = message.pitch;
      gimbal_pitch_pub_->publish(pitch_msg);
    }
    if (serial_feedback_pub_) {
      aimbot_msgs::msg::SerialFeedback feedback;
      feedback.header.stamp = this->get_clock()->now();
      feedback.yaw = message.yaw;
      feedback.pitch = message.pitch;
      feedback.status = message.status;
      feedback.armor_flag = message.armor_flag;
      feedback.allround = message.allround;
      feedback.latency = message.latency;
      feedback.x_c = message.x_c;
      feedback.v_x = message.v_x;
      feedback.y_c = message.y_c;
      feedback.v_y = message.v_y;
      feedback.z1 = message.z1;
      feedback.z2 = message.z2;
      feedback.r1 = message.r1;
      feedback.r2 = message.r2;
      feedback.yaw_a = message.yaw_a;
      feedback.vyaw = message.vyaw;
      serial_feedback_pub_->publish(feedback);
    }
  }

  void sendFrame(const MessData & message)
  {
    if (!serial_port_.isOpen()) {
      RCLCPP_WARN(get_logger(), "Serial port not open, cannot send frame");
      return;
    }
    std::array<std::uint8_t, 64> buffer{};
    std::memcpy(buffer.data(), &message, sizeof(MessData));
    std::memset(buffer.data() + sizeof(MessData), 0, 64 - sizeof(MessData));
    RCLCPP_INFO(get_logger(), "Buffer bytes 57-63: %02X %02X %02X %02X %02X %02X %02X", buffer[57], buffer[58], buffer[59], buffer[60], buffer[61], buffer[62], buffer[63]);
    const std::size_t written = serial_port_.write(buffer.data(), 64);
    RCLCPP_INFO(get_logger(), "Attempted to write 64 bytes, actually wrote %zu bytes", written);
    if (written != 64) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Serial write incomplete (%zu/64 bytes)", written);
    } else {
      RCLCPP_INFO(get_logger(), "Successfully wrote 64 bytes to serial");
    }
  }

  double computeLatencyMs(std::uint64_t current_timestamp)
  {
    double latency_ms = 0.0;
    if (last_frame_timestamp_ != 0U && current_timestamp > last_frame_timestamp_) {
      const std::uint64_t diff = current_timestamp - last_frame_timestamp_;
      latency_ms = static_cast<double>(diff) * kNanosecondsToMilliseconds;
    }
    last_frame_timestamp_ = current_timestamp;
    return latency_ms;
  }

  std::optional<TargetKinematics> selectDetection(const aimbot_msgs::msg::DetectionArray & array_msg)
  {
    if (array_msg.detections.empty()) {
      return std::nullopt;
    }

    const aimbot_msgs::msg::Detection * best = nullptr;
    for (const auto & detection : array_msg.detections) {
      if (best == nullptr || detection.confidence > best->confidence) {
        best = &detection;
      }
    }
    if (best == nullptr) {
      return std::nullopt;
    }

    TargetKinematics state;
    state.x = static_cast<double>(best->translation_vector[0]);
    state.y = static_cast<double>(best->translation_vector[1]);
    state.z = static_cast<double>(best->translation_vector[2]);
    state.yaw = static_cast<double>(best->rotation_vector[2]);
    state.class_id = best->class_id;
    state.timestamp = array_msg.timestamp;
    return state;
  }

  void fillFromDetection(const TargetKinematics & state, MessData & payload)
  {
    payload.armor_flag = static_cast<std::uint8_t>(std::max(state.class_id, 0));

    const double distance_xy = std::hypot(state.x, state.y);
    payload.yaw = static_cast<float>(std::atan2(state.y, state.x));
    payload.pitch = static_cast<float>(std::atan2(state.z, distance_xy));

    payload.x_c = static_cast<float>(state.x);
    payload.y_c = static_cast<float>(state.y);
    payload.z1 = static_cast<float>(state.z);
    payload.z2 = static_cast<float>(state.z);
    payload.r1 = static_cast<float>(distance_xy);
    payload.r2 = static_cast<float>(distance_xy);
    payload.yaw_a = static_cast<float>(state.yaw);

    if (last_target_state_.has_value()) {
      const auto & last = *last_target_state_;
      const double dt = static_cast<double>(state.timestamp - last.timestamp) * kNanosecondsToSeconds;
      if (dt > 1e-6) {
        payload.v_x = static_cast<float>((state.x - last.x) / dt);
        payload.v_y = static_cast<float>((state.y - last.y) / dt);
        payload.vyaw = static_cast<float>((state.yaw - last.yaw) / dt);
      }
    }

    last_target_state_ = state;
  }

  aim_auto::SerialPort serial_port_;
  std::string tracker_topic_;
  std::string gimbal_yaw_topic_;
  std::string gimbal_pitch_topic_;
  std::string serial_feedback_topic_;

  rclcpp::Subscription<aimbot_msgs::msg::DetectionArray>::SharedPtr detection_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gimbal_yaw_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gimbal_pitch_pub_;
  rclcpp::Publisher<aimbot_msgs::msg::SerialFeedback>::SharedPtr serial_feedback_pub_;
  rclcpp::TimerBase::SharedPtr read_timer_;

  std::vector<std::uint8_t> read_accumulator_;

  std::uint8_t status_code_ = 5;
  bool allround_mode_ = false;
  int message_hold_threshold_ = 5;
  int loss_count_ = 0;
  double current_latency_ms_ = 0.0;

  MessData last_valid_message_{};
  bool has_last_valid_message_ = false;
  std::uint64_t last_frame_timestamp_ = 0U;
  std::optional<TargetKinematics> last_target_state_;
};

}  // namespace aim_auto

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<aim_auto::SerialBridgeNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("serial_bridge_node"), "Unhandled exception: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
