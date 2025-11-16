#include <aimbot_msgs/msg/detection.hpp>
#include <aimbot_msgs/msg/detection_array.hpp>
#include <rclcpp/rclcpp.hpp>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "aim_auto/ekf_tracker.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

Eigen::Matrix3d createRotationMount()
{
  Eigen::Matrix3d rotation;
  rotation << 0.0, 0.0, -1.0,
    -1.0, 0.0, 0.0,
    0.0, 1.0, 0.0;
  return rotation;
}

}  // namespace

class TrackerNode : public rclcpp::Node
{
public:
  explicit TrackerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("tracker_node", options)
  {
    this->declare_parameter<std::string>("aimauto_config_path", "");
    this->declare_parameter<double>("fixed_pitch_deg", -15.0);

    aimauto_config_path_ = this->get_parameter("aimauto_config_path").as_string();
    fixed_pitch_rad_ = deg2rad(this->get_parameter("fixed_pitch_deg").as_double());

    if (aimauto_config_path_.empty()) {
      const auto source_dir = std::filesystem::path(__FILE__).parent_path();
      const auto default_config_dir = source_dir.parent_path().parent_path() / "config";
      aimauto_config_path_ = (default_config_dir / "AimautoConfig.yaml").string();
    }

    loadAimautoConfig();

    detection_sub_ = create_subscription<aimbot_msgs::msg::DetectionArray>(
      "detections", 10,
      std::bind(&TrackerNode::detectionCallback, this, std::placeholders::_1));

    tracked_pub_ = create_publisher<aimbot_msgs::msg::DetectionArray>("tracked_detections", 10);

#ifdef DEBUG_VISUALIZATION
  marker_array_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("armor_markers", 10);
#endif
  }

private:
  static double deg2rad(double deg)
  {
    return deg * kPi / 180.0;
  }

  void loadAimautoConfig()
  {
    cv::FileStorage fs(aimauto_config_path_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Failed to open aimauto config: %s", aimauto_config_path_.c_str());
      return;
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

  void detectionCallback(const aimbot_msgs::msg::DetectionArray::SharedPtr msg)
  {
    aimbot_msgs::msg::DetectionArray tracked_msg = *msg;

    std::vector<aim_auto::ArmorDetection> armor_detections;
    armor_detections.reserve(tracked_msg.detections.size());

#ifdef DEBUG_VISUALIZATION
    visualization_msgs::msg::MarkerArray marker_array;
#endif

    for (const auto & detection : tracked_msg.detections) {
      aim_auto::ArmorDetection armor_detection;
      armor_detection.class_id = detection.class_id;
      armor_detection.position = Eigen::Vector3d(
        static_cast<double>(detection.translation_vector[0]),
        static_cast<double>(detection.translation_vector[1]),
        static_cast<double>(detection.translation_vector[2]));
      armor_detection.yaw = static_cast<double>(detection.rotation_vector[2]);
      armor_detections.push_back(armor_detection);

#ifdef DEBUG_VISUALIZATION
      marker_array.markers.push_back(buildArmorMarker(detection, msg->header));
#endif
    }

    ekf_tracker_.updateDetections(armor_detections, msg->header.stamp);

#ifdef DEBUG_VISUALIZATION
    const auto predicted_armors = ekf_tracker_.predictedArmors();
    int armor_idx = 0;
    for (const auto & armor : predicted_armors) {
      marker_array.markers.push_back(buildPredictedArmorMarker(armor, msg->header, armor_idx++));
    }

    marker_array_pub_->publish(marker_array);
#endif

    tracked_pub_->publish(tracked_msg);
  }

#ifdef DEBUG_VISUALIZATION
  Eigen::Matrix3d buildArmorRotation(double yaw_world, double pitch_world) const
  {
    Eigen::Matrix3d mat_yaw;
    mat_yaw << std::cos(yaw_world), -std::sin(yaw_world), 0.0,
      std::sin(yaw_world), std::cos(yaw_world), 0.0,
      0.0, 0.0, 1.0;


     Eigen::Matrix3d mat_pitch;
     mat_pitch << 1.0, 0.0, 0.0,
       0.0, std::cos(pitch_world), -std::sin(pitch_world),
       0.0, std::sin(pitch_world), std::cos(pitch_world);

    return   mat_yaw * rotation_mount_ * mat_pitch;
  }

  visualization_msgs::msg::Marker buildArmorMarker(const aimbot_msgs::msg::Detection & detection, const std_msgs::msg::Header & header) const
  {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.header.frame_id = "map";
    marker.ns = "armor";
    marker.id = detection.class_id * 100 + 1;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.01;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    marker.lifetime = rclcpp::Duration::from_seconds(3.0);

    const auto & object_points = (detection.class_id == 0 || detection.class_id == 7) ?
      big_armor_object_points_ : small_armor_object_points_;

    const Eigen::Matrix3d rotation = buildArmorRotation(static_cast<double>(detection.rotation_vector[2]), static_cast<double>(detection.rotation_vector[1]));
    Eigen::Vector3d world_translation(
      static_cast<double>(detection.translation_vector[0]),
      static_cast<double>(detection.translation_vector[1]),
      static_cast<double>(detection.translation_vector[2]));
    world_translation /= 1000.0;  // convert mm to m

    marker.points.clear();
    for (const auto & pt : object_points) {
      Eigen::Vector3d local_pt(pt.x, pt.y, pt.z);
      Eigen::Vector3d world_pt = world_translation + rotation * (local_pt / 1000.0);

      geometry_msgs::msg::Point p;
      p.x = world_pt.x();
      p.y = world_pt.y();
      p.z = world_pt.z();
      marker.points.push_back(p);
    }
    if (!marker.points.empty()) {
      marker.points.push_back(marker.points.front());
    }

    return marker;
  }

  visualization_msgs::msg::Marker buildPredictedArmorMarker(const aim_auto::ArmorDetection & armor, const std_msgs::msg::Header & header, int index) const
  {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.header.frame_id = "map";
    marker.ns = "predicted_armor";
    marker.id = armor.class_id * 100 + 2 + index * 10;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.01;
    if (index % 2 == 0) {
      // Upper armor (z1)
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 1.0;  // Blue
    } else {
      // Lower armor (z2)
      marker.color.r = 1.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;  // Red
    }
    marker.color.a = 1.0;
    marker.lifetime = rclcpp::Duration::from_seconds(3.0);

    std::vector<cv::Point3d> object_points;
    if (armor.class_id == 5 || armor.class_id == 12) {
      const double r = 276.5;
      object_points.emplace_back(-r, 0.0, 0.0);
      object_points.emplace_back(r * 0.5, r * std::sqrt(3) * 0.5, 0.0);
      object_points.emplace_back(r * 0.5, -r * std::sqrt(3) * 0.5, 0.0);
    } else {
      object_points = (armor.class_id == 0 || armor.class_id == 7) ?
        big_armor_object_points_ : small_armor_object_points_;
    }

    const Eigen::Matrix3d rotation = buildArmorRotation(armor.yaw, fixed_pitch_rad_);
    Eigen::Vector3d world_translation = armor.position / 1000.0;

    marker.points.clear();
    for (const auto & pt : object_points) {
      Eigen::Vector3d local_pt(pt.x, pt.y, pt.z);
      Eigen::Vector3d world_pt = world_translation + rotation * (local_pt / 1000.0);

      geometry_msgs::msg::Point p;
      p.x = world_pt.x();
      p.y = world_pt.y();
      p.z = world_pt.z();
      marker.points.push_back(p);
    }
    if (!marker.points.empty()) {
      marker.points.push_back(marker.points.front());
    }

    return marker;
  }
#endif

  rclcpp::Subscription<aimbot_msgs::msg::DetectionArray>::SharedPtr detection_sub_;
  rclcpp::Publisher<aimbot_msgs::msg::DetectionArray>::SharedPtr tracked_pub_;
#ifdef DEBUG_VISUALIZATION
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_;
#endif

  std::string aimauto_config_path_;
  double fixed_pitch_rad_ = 0.0;
  const Eigen::Matrix3d rotation_mount_ = createRotationMount();

  std::vector<cv::Point3d> small_armor_object_points_;
  std::vector<cv::Point3d> big_armor_object_points_;

  aim_auto::EkfTracker ekf_tracker_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrackerNode>());
  rclcpp::shutdown();
  return 0;
}