#include <aimbot_msgs/msg/camera_image.hpp>
#include <aimbot_msgs/msg/detection.hpp>
#include <aimbot_msgs/msg/detection_array.hpp>
#include "aim_auto/armor_pnp_solver.hpp"
#include "aim_auto/ref_detection.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/polygon.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct YawReprojectionError
{
  YawReprojectionError(
    const cv::Mat & camera_matrix,
    const cv::Mat & dist_coeffs,
    const std::vector<cv::Point3d> & object_points,
    const std::array<cv::Point2d, 4> & image_points,
    const Eigen::Matrix3d & world_to_camera_rot,
    const Eigen::Vector3d & camera_translation_world,
    const Eigen::Vector3d & world_translation,
    double fixed_pitch_rad)
  : camera_matrix_(camera_matrix.clone()),
    dist_coeffs_(dist_coeffs.clone()),
    object_points_(object_points),
    image_points_(image_points),
    world_to_camera_rot_(world_to_camera_rot),
    camera_translation_world_(camera_translation_world),
    world_translation_(world_translation),
    fixed_pitch_rad_(fixed_pitch_rad),
    rotation_mount_(createRotationMount())
  {}

  bool operator()(const double * const yaw_world_ptr, double * residuals) const
  {
    const double yaw_world = *yaw_world_ptr;

    Eigen::Matrix3d mat_y;
    mat_y << std::cos(yaw_world), 0.0, std::sin(yaw_world),
      0.0, 1.0, 0.0,
      -std::sin(yaw_world), 0.0, std::cos(yaw_world);

    Eigen::Matrix3d mat_x;
    mat_x << 1.0, 0.0, 0.0,
      0.0, std::cos(fixed_pitch_rad_), -std::sin(fixed_pitch_rad_),
      0.0, std::sin(fixed_pitch_rad_), std::cos(fixed_pitch_rad_);

    const Eigen::Matrix3d rotation_total = world_to_camera_rot_ * rotation_mount_ * mat_y * mat_x;

    cv::Mat rotation_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        rotation_cv.at<double>(r, c) = rotation_total(r, c);
      }
    }

    cv::Mat rvec_cv;
    cv::Rodrigues(rotation_cv, rvec_cv);

    const Eigen::Vector3d camera_t = world_to_camera_rot_ * (world_translation_ - camera_translation_world_);
    cv::Vec3d tvec(camera_t.x(), camera_t.y(), camera_t.z());

    std::vector<cv::Point2d> projected_points;
    cv::projectPoints(object_points_, rvec_cv, tvec, camera_matrix_, dist_coeffs_, projected_points);
    if (projected_points.size() != image_points_.size()) {
      std::fill(residuals, residuals + 8, 0.0);
      return false;
    }

    for (std::size_t i = 0; i < image_points_.size(); ++i) {
      residuals[2 * i + 0] = projected_points[i].x - image_points_[i].x;
      residuals[2 * i + 1] = projected_points[i].y - image_points_[i].y;
    }
    return true;
  }

  double evaluate(double yaw_world) const
  {
    double residuals[8];
    operator()(&yaw_world, residuals);
    double error = 0.0;
    for (double value : residuals) {
      error += value * value;
    }
    return error;
  }

private:
  static Eigen::Matrix3d createRotationMount()
  {
    Eigen::Matrix3d rotation;
    rotation << 0.0, 0.0, 1.0,
      -1.0, 0.0, 0.0,
      0.0, -1.0, 0.0;
    return rotation;
  }

  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  std::vector<cv::Point3d> object_points_;
  std::array<cv::Point2d, 4> image_points_;
  Eigen::Matrix3d world_to_camera_rot_;
  Eigen::Vector3d camera_translation_world_;
  Eigen::Vector3d world_translation_;
  double fixed_pitch_rad_;
  Eigen::Matrix3d rotation_mount_;
};

struct ClassifiedLabel
{
  int relative_index;
  bool is_big;
  std::string display_name;
};

std::string normalizeLabel(const std::string & label)
{
  std::string normalized;
  normalized.reserve(label.size());
  for (char ch : label) {
    normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return normalized;
}

std::optional<ClassifiedLabel> mapLabel(const std::string & raw_label)
{
  const std::string label = normalizeLabel(raw_label);
  if (label.empty()) {
    return std::nullopt;
  }

  if (label == "1" || label == "hero") {
    return ClassifiedLabel{0, true, "1"};
  }
  if (label == "2") {
    return ClassifiedLabel{1, false, "2"};
  }
  if (label == "3") {
    return ClassifiedLabel{2, false, "3"};
  }
  if (label == "4") {
    return ClassifiedLabel{3, false, "4"};
  }
  if (label == "5") {
    return ClassifiedLabel{4, false, "5"};
  }
  if (label == "outpost" || label == "base") {
    return ClassifiedLabel{5, true, "outpost"};
  }
  if (label == "guard" || label == "sentry") {
    return ClassifiedLabel{6, true, "sentry"};
  }
  return std::nullopt;
}

}  // namespace

class RefDetectorNode : public rclcpp::Node
{
public:
  explicit RefDetectorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("detector_ref_node", options)
  {
    last_fps_time_ = std::chrono::steady_clock::now();
    frame_count_ = 0;
    fps_ = 0.0;

    this->declare_parameter<double>("confidence_threshold", 0.85);
    this->declare_parameter<std::string>("detection_topic", "detections");
    this->declare_parameter<std::string>("gimbal_yaw_topic", "");
    this->declare_parameter<std::string>("gimbal_pitch_topic", "");
    this->declare_parameter<double>("initial_gimbal_yaw_deg", 0.0);
    this->declare_parameter<double>("initial_gimbal_pitch_deg", 0.0);
    this->declare_parameter<double>("fixed_pitch_deg", 15.0);

    confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
    detection_topic_ = this->get_parameter("detection_topic").as_string();
    gimbal_yaw_topic_ = this->get_parameter("gimbal_yaw_topic").as_string();
    gimbal_pitch_topic_ = this->get_parameter("gimbal_pitch_topic").as_string();
    gimbal_yaw_rad_ = deg2rad(this->get_parameter("initial_gimbal_yaw_deg").as_double());
    gimbal_pitch_rad_ = deg2rad(this->get_parameter("initial_gimbal_pitch_deg").as_double());
    fixed_pitch_rad_ = deg2rad(this->get_parameter("fixed_pitch_deg").as_double());

    param_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&RefDetectorNode::parameter_callback, this, std::placeholders::_1));

    const auto source_dir = std::filesystem::path(__FILE__).parent_path();
    const auto package_root = source_dir.parent_path().parent_path();
    const auto default_config_dir = package_root / "config";
    const auto default_camera_config = (default_config_dir / "CameraConfig.yaml").string();
    const auto default_aimauto_config = (default_config_dir / "AimautoConfig.yaml").string();
    const auto default_detection_config = (default_config_dir / "DetectionConfig.yaml").string();
    const auto default_model_path = (package_root / "model" / "lenet.onnx").string();
    const auto default_label_path = (package_root / "model" / "label.txt").string();

    camera_config_path_ = this->declare_parameter<std::string>("camera_config_path", default_camera_config);
    aimauto_config_path_ = this->declare_parameter<std::string>("aimauto_config_path", default_aimauto_config);
    detection_config_path_ = this->declare_parameter<std::string>("detection_config_path", default_detection_config);
    number_model_path_ = this->declare_parameter<std::string>("number_model_path", default_model_path);
    number_label_path_ = this->declare_parameter<std::string>("number_label_path", default_label_path);

    const std::string target_color_param = this->declare_parameter<std::string>("target_color", "red");
    target_color_ = normalizeTargetColor(target_color_param);

    try {
      detector_params_ = aim_auto::loadDetectionParams(detection_config_path_);
      detector_ = std::make_unique<aim_auto::RefArmorDetector>(
        detector_params_, number_model_path_, number_label_path_);
      pnp_solver_ = std::make_unique<aim_auto::ArmorPnPSolver>(camera_config_path_, aimauto_config_path_);
      loadCameraExtrinsics();
      loadAimautoGeometry();
      {
        std::lock_guard<std::mutex> lock(transform_mutex_);
        updateTransformsLocked();
      }
      if (!gimbal_yaw_topic_.empty()) {
        gimbal_yaw_sub_ = create_subscription<std_msgs::msg::Float32>(
          gimbal_yaw_topic_, 10,
          std::bind(&RefDetectorNode::gimbalYawCallback, this, std::placeholders::_1));
      }
      if (!gimbal_pitch_topic_.empty()) {
        gimbal_pitch_sub_ = create_subscription<std_msgs::msg::Float32>(
          gimbal_pitch_topic_, 10,
          std::bind(&RefDetectorNode::gimbalPitchCallback, this, std::placeholders::_1));
      }
      RCLCPP_INFO(
        get_logger(),
        "Ref detector initialized (target_color=%s)",
        aim_auto::to_color_string(target_color_).c_str());
    } catch (const std::exception & e) {
      RCLCPP_FATAL(get_logger(), "Failed to initialize detector_ref_node: %s", e.what());
      throw;
    }

    image_sub_ = create_subscription<aimbot_msgs::msg::CameraImage>(
      "camera_image", rclcpp::SensorDataQoS().reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT),
      std::bind(&RefDetectorNode::image_callback, this, std::placeholders::_1));

    detection_pub_ = create_publisher<aimbot_msgs::msg::DetectionArray>(detection_topic_, 10);
  }

private:
  rcl_interfaces::msg::SetParametersResult parameter_callback(const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto & param : parameters) {
      if (param.get_name() == "confidence_threshold") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
          confidence_threshold_ = param.as_double();
          RCLCPP_INFO(get_logger(), "Updated confidence threshold to: %.3f", confidence_threshold_);
        } else {
          result.successful = false;
          result.reason = "confidence_threshold must be a double";
        }
      }
    }
    return result;
  }

  void image_callback(const aimbot_msgs::msg::CameraImage::SharedPtr msg)
  {
    try {
      const sensor_msgs::msg::Image & image_msg = msg->image;
      const uint64_t camera_timestamp = msg->timestamp;

      cv_bridge::CvImagePtr cv_ptr;
      if (image_msg.encoding == sensor_msgs::image_encodings::BGR8) {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
      } else if (image_msg.encoding == sensor_msgs::image_encodings::RGB8) {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8);
      } else {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Converted unsupported encoding %s to BGR8",
          image_msg.encoding.c_str());
      }

      cv::Mat frame = cv_ptr->image;
      if (frame.empty()) {
        RCLCPP_WARN(get_logger(), "Received empty image");
        return;
      }
      if (frame.channels() == 4) {
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
      } else if (frame.channels() == 1) {
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
      }

      auto armors = detector_->detect(frame, target_color_);

      aimbot_msgs::msg::DetectionArray detection_msg;
      detection_msg.header = image_msg.header;
      detection_msg.timestamp = camera_timestamp;
      detection_msg.confidence_threshold = confidence_threshold_;

#ifdef DEBUG_VISUALIZATION
      struct VisualInfo
      {
        cv::Rect box;
        std::string label;
        float confidence;
      };
      std::vector<VisualInfo> debug_visuals;
      debug_visuals.reserve(armors.size());
#endif

      detection_msg.detections.reserve(armors.size());
      for (const auto & armor : armors) {
        if (armor.confidence < confidence_threshold_) {
          continue;
        }
        const auto mapped_label = mapLabel(armor.number);
        if (!mapped_label) {
          continue;
        }

        // Debug: Print neural network detection results
        RCLCPP_INFO(get_logger(), "Neural network detected: number='%s', confidence=%.3f, mapped_label='%s' (relative_index=%d, is_big=%s)",
                    armor.number.c_str(), armor.confidence, mapped_label->display_name.c_str(),
                    mapped_label->relative_index, mapped_label->is_big ? "true" : "false");

        const std::array<cv::Point2f, 4> polygon{
          armor.left_light.bottom,
          armor.left_light.top,
          armor.right_light.top,
          armor.right_light.bottom};
        cv::Rect bbox = cv::boundingRect(std::vector<cv::Point2f>(polygon.begin(), polygon.end()));
        const cv::Rect image_bounds(0, 0, frame.cols, frame.rows);
        bbox = bbox & image_bounds;
        if (bbox.width <= 5 || bbox.height <= 5) {
          continue;
        }

        bool is_big = mapped_label->is_big || armor.type == aim_auto::ArmorType::LARGE;
        cv::Vec3d rvec(0.0, 0.0, 0.0);
        cv::Vec3d tvec(0.0, 0.0, 0.0);
        std::vector<cv::Point2f> image_points;
        bool pnp_success = false;
        if (pnp_solver_) {
          pnp_success = pnp_solver_->solve(frame, bbox, is_big ? 0 : 1, rvec, tvec, &image_points);
        }
        if (!pnp_success || image_points.size() != 4) {
          continue;
        }

        aim_auto::TargetColor color = target_color_;
        const int color_offset = color == aim_auto::TargetColor::BLUE ? 0 : 7;
        const int class_id = color_offset + mapped_label->relative_index;

        // Debug: Print final class_id calculation
        RCLCPP_INFO(get_logger(), "Final class_id calculation: color=%s (offset=%d) + relative_index=%d = class_id=%d",
                    aim_auto::to_color_string(color).c_str(), color_offset, mapped_label->relative_index, class_id);

        geometry_msgs::msg::Point32 p1, p2, p3, p4;
        p1.x = polygon[0].x;
        p1.y = polygon[0].y;
        p2.x = polygon[1].x;
        p2.y = polygon[1].y;
        p3.x = polygon[2].x;
        p3.y = polygon[2].y;
        p4.x = polygon[3].x;
        p4.y = polygon[3].y;

        aimbot_msgs::msg::Detection detection;
        detection.header = image_msg.header;
        detection.class_id = class_id;
        detection.class_name = aim_auto::to_color_string(color) + "_" + mapped_label->display_name;
        detection.confidence = armor.confidence;
        detection.bbox.points = {p1, p2, p3, p4};
        detection.center.x = static_cast<float>(bbox.x + bbox.width / 2.0);
        detection.center.y = static_cast<float>(bbox.y + bbox.height / 2.0);
        detection.center.z = 0.0f;

        Eigen::Matrix3d camera_to_world_rot;
        Eigen::Matrix3d world_to_camera_rot;
        Eigen::Vector3d camera_translation_world;
        double fixed_pitch_rad = fixed_pitch_rad_;
        {
          std::lock_guard<std::mutex> lock(transform_mutex_);
          if (transforms_dirty_) {
            updateTransformsLocked();
          }
          camera_to_world_rot = camera_to_world_rot_;
          world_to_camera_rot = world_to_camera_rot_;
          camera_translation_world = camera_translation_world_;
          fixed_pitch_rad = fixed_pitch_rad_;
        }

        Eigen::Vector3d camera_t(tvec[0], tvec[1], tvec[2]);
        Eigen::Vector3d world_translation = camera_to_world_rot * camera_t + camera_translation_world;

        std::array<cv::Point2d, 4> image_points_array{};
        for (std::size_t i = 0; i < image_points_array.size(); ++i) {
          image_points_array[i] = cv::Point2d(image_points[i]);
        }

        const auto & object_points = is_big ? big_armor_object_points_ : small_armor_object_points_;
        const double optimized_yaw = optimizeYaw(
          world_to_camera_rot,
          camera_translation_world,
          world_translation,
          object_points,
          image_points_array,
          fixed_pitch_rad);

        detection.rotation_vector[0] = 0.0f;
        detection.rotation_vector[1] = 0.0f;
        detection.rotation_vector[2] = static_cast<float>(optimized_yaw);
        detection.translation_vector[0] = static_cast<float>(world_translation.x());
        detection.translation_vector[1] = static_cast<float>(world_translation.y());
        detection.translation_vector[2] = static_cast<float>(world_translation.z());

        detection_msg.detections.push_back(detection);

#ifdef DEBUG_VISUALIZATION
        debug_visuals.push_back({bbox, detection.class_name, detection.confidence});
#endif
      }

      detection_pub_->publish(detection_msg);

      frame_count_++;
      const auto current_time = std::chrono::steady_clock::now();
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_time_).count();
      if (elapsed >= 1000) {
        fps_ = static_cast<double>(frame_count_) / (elapsed / 1000.0);
        frame_count_ = 0;
        last_fps_time_ = current_time;
      }

#ifdef DEBUG_VISUALIZATION
      cv::Mat debug_frame = frame.clone();
      const std::string fps_text = cv::format("FPS: %.1f", fps_);
      cv::putText(debug_frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
        cv::Scalar(0, 255, 0), 2);
      for (const auto & vis : debug_visuals) {
        cv::rectangle(debug_frame, vis.box, cv::Scalar(0, 255, 0), 1);
        std::string label = vis.label + cv::format(" %.2f", vis.confidence);
        cv::putText(debug_frame, label, cv::Point(vis.box.x, std::max(10, vis.box.y - 5)),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
      }
      cv::imshow("Ref Detection", debug_frame);
      cv::waitKey(1);
#endif

      RCLCPP_DEBUG_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Published %zu detections (FPS: %.1f, threshold: %.3f)",
        detection_msg.detections.size(), fps_, confidence_threshold_);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Ref detection failed: %s", e.what());
    }
  }

  static double deg2rad(double deg)
  {
    return deg * kPi / 180.0;
  }

  aim_auto::TargetColor normalizeTargetColor(const std::string & value)
  {
    const std::string normalized = normalizeLabel(value);
    return normalized == "blue" ? aim_auto::TargetColor::BLUE : aim_auto::TargetColor::RED;
  }

  void gimbalYawCallback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(transform_mutex_);
      gimbal_yaw_rad_ = msg->data;
      transforms_dirty_ = true;
    }
    RCLCPP_DEBUG(get_logger(), "Updated gimbal yaw to %.3f rad", gimbal_yaw_rad_);
  }

  void gimbalPitchCallback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(transform_mutex_);
      gimbal_pitch_rad_ = msg->data;
      transforms_dirty_ = true;
    }
    RCLCPP_DEBUG(get_logger(), "Updated gimbal pitch to %.3f rad", gimbal_pitch_rad_);
  }

  void updateTransformsLocked()
  {
    const double yaw = gimbal_yaw_rad_;
    const double pitch = gimbal_pitch_rad_;

    Eigen::Matrix3d m_yaw;
    m_yaw << std::cos(yaw), -std::sin(yaw), 0.0,
      std::sin(yaw), std::cos(yaw), 0.0,
      0.0, 0.0, 1.0;

    Eigen::Matrix3d m_pitch;
    m_pitch << std::cos(pitch), 0.0, -std::sin(pitch),
      0.0, 1.0, 0.0,
      std::sin(pitch), 0.0, std::cos(pitch);

    const Eigen::Matrix3d rotation_mount = []() {
      Eigen::Matrix3d rotation;
      rotation << 0.0, 0.0, 1.0,
        -1.0, 0.0, 0.0,
        0.0, -1.0, 0.0;
      return rotation;
    }();

    Eigen::Matrix3d r_mat = m_yaw * m_pitch;
    camera_to_world_rot_ = r_mat * rotation_mount;
    world_to_camera_rot_ = camera_to_world_rot_.transpose();

    Eigen::Vector3d camera_offset(vector_x_, vector_y_, vector_z_);
    camera_translation_world_ = r_mat * camera_offset;

    transforms_dirty_ = false;
  }

  void loadCameraExtrinsics()
  {
    cv::FileStorage fs(camera_config_path_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      throw std::runtime_error("Failed to open camera config: " + camera_config_path_);
    }

    fs["vector_x"] >> vector_x_;
    fs["vector_y"] >> vector_y_;
    fs["vector_z"] >> vector_z_;
  }

  void loadAimautoGeometry()
  {
    cv::FileStorage fs(aimauto_config_path_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      throw std::runtime_error("Failed to open aimauto config: " + aimauto_config_path_);
    }

    double small_a = 0.0;
    double small_b = 0.0;
    double big_a = 0.0;
    double big_b = 0.0;
    fs["small_armor_a"] >> small_a;
    fs["small_armor_b"] >> small_b;
    fs["big_armor_a"] >> big_a;
    fs["big_armor_b"] >> big_b;

    const double half_small_a = small_a * 0.5;
    const double half_small_b = small_b * 0.5;
    const double half_big_a = big_a * 0.5;
    const double half_big_b = big_b * 0.5;

    small_armor_object_points_.clear();
    small_armor_object_points_.reserve(4);
    small_armor_object_points_.emplace_back(-half_small_a, -half_small_b, 0.0);
    small_armor_object_points_.emplace_back(half_small_a, -half_small_b, 0.0);
    small_armor_object_points_.emplace_back(half_small_a, half_small_b, 0.0);
    small_armor_object_points_.emplace_back(-half_small_a, half_small_b, 0.0);

    big_armor_object_points_.clear();
    big_armor_object_points_.reserve(4);
    big_armor_object_points_.emplace_back(-half_big_a, -half_big_b, 0.0);
    big_armor_object_points_.emplace_back(half_big_a, -half_big_b, 0.0);
    big_armor_object_points_.emplace_back(half_big_a, half_big_b, 0.0);
    big_armor_object_points_.emplace_back(-half_big_a, half_big_b, 0.0);
  }

  double optimizeYaw(
    const Eigen::Matrix3d & world_to_camera_rot,
    const Eigen::Vector3d & camera_translation_world,
    const Eigen::Vector3d & world_translation,
    const std::vector<cv::Point3d> & object_points,
    const std::array<cv::Point2d, 4> & image_points,
    double fixed_pitch_rad) const
  {
    YawReprojectionError error(
      pnp_solver_->camera_matrix(),
      pnp_solver_->dist_coeffs(),
      object_points,
      image_points,
      world_to_camera_rot,
      camera_translation_world,
      world_translation,
      fixed_pitch_rad);

    double best_yaw = 0.0;
    double best_error = std::numeric_limits<double>::infinity();
    constexpr int kSamples = 72;
    for (int i = 0; i < kSamples; ++i) {
      const double yaw = -kPi + static_cast<double>(i) * 2.0 * kPi / static_cast<double>(kSamples);
      const double err = error.evaluate(yaw);
      if (err < best_error) {
        best_error = err;
        best_yaw = yaw;
      }
    }

    double yaw_param = best_yaw;
    ceres::Problem problem;
    auto * cost_function = new ceres::NumericDiffCostFunction<YawReprojectionError, ceres::CENTRAL, 8, 1>(
      new YawReprojectionError(error));
    problem.AddResidualBlock(cost_function, nullptr, &yaw_param);

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.termination_type != ceres::CONVERGENCE || !std::isfinite(yaw_param)) {
      return best_yaw;
    }
    return yaw_param;
  }

  rclcpp::Subscription<aimbot_msgs::msg::CameraImage>::SharedPtr image_sub_;
  rclcpp::Publisher<aimbot_msgs::msg::DetectionArray>::SharedPtr detection_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
  std::unique_ptr<aim_auto::ArmorPnPSolver> pnp_solver_;
  std::unique_ptr<aim_auto::RefArmorDetector> detector_;

  std::string camera_config_path_;
  std::string aimauto_config_path_;
  std::string detection_config_path_;
  std::string number_model_path_;
  std::string number_label_path_;

  double confidence_threshold_ = 0.85;
  std::string detection_topic_ = "detections";
  std::string gimbal_yaw_topic_;
  std::string gimbal_pitch_topic_;

  std::chrono::steady_clock::time_point last_fps_time_;
  int frame_count_ = 0;
  double fps_ = 0.0;

  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr gimbal_yaw_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr gimbal_pitch_sub_;

  double gimbal_yaw_rad_ = 0.0;
  double gimbal_pitch_rad_ = 0.0;
  double fixed_pitch_rad_ = 0.0;
  double vector_x_ = 0.0;
  double vector_y_ = 0.0;
  double vector_z_ = 0.0;

  aim_auto::TargetColor target_color_{aim_auto::TargetColor::RED};
  aim_auto::DetectionParams detector_params_{};

  mutable std::mutex transform_mutex_;
  Eigen::Matrix3d camera_to_world_rot_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d world_to_camera_rot_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d camera_translation_world_{0.0, 0.0, 0.0};
  bool transforms_dirty_ = false;

  std::vector<cv::Point3d> small_armor_object_points_;
  std::vector<cv::Point3d> big_armor_object_points_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RefDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
