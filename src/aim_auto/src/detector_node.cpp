#include <aimbot_msgs/msg/camera_image.hpp>
#include <aimbot_msgs/msg/detection.hpp>
#include <aimbot_msgs/msg/detection_array.hpp>
#include "aim_auto/armor_pnp_solver.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/polygon.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct LetterboxResult {
    cv::Mat image;
    float scale;
    int pad_x;
    int pad_y;
};

LetterboxResult letterbox(const cv::Mat &src, int target) {
    const int orig_w = src.cols;
    const int orig_h = src.rows;
    if (orig_w == 0 || orig_h == 0) {
        throw std::runtime_error("Empty input image");
    }

    const float scale = std::min(static_cast<float>(target) / static_cast<float>(orig_w),
                                                             static_cast<float>(target) / static_cast<float>(orig_h));
    const int new_w = static_cast<int>(std::round(orig_w * scale));
    const int new_h = static_cast<int>(std::round(orig_h * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    const int pad_w = target - new_w;
    const int pad_h = target - new_h;
    const int pad_left = pad_w / 2;
    const int pad_top = pad_h / 2;
    const int pad_right = pad_w - pad_left;
    const int pad_bottom = pad_h - pad_top;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
                                         cv::Scalar(114, 114, 114));

    LetterboxResult result;
    result.image = std::move(padded);
    result.scale = scale;
    result.pad_x = pad_left;
    result.pad_y = pad_top;
    return result;
}

}  // namespace

namespace {

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

}  // namespace

class DetectorNode : public rclcpp::Node {
public:
    explicit DetectorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
    : Node("detector_node", options)
    {
    // Initialize FPS calculation
    last_fps_time_ = std::chrono::steady_clock::now();
    frame_count_ = 0;
    fps_ = 0.0;

    // Declare parameters
    this->declare_parameter<float>("confidence_threshold", 0.85f);
    this->declare_parameter<std::string>("detection_topic", "detections");
    this->declare_parameter<std::string>("gimbal_yaw_topic", "");
    this->declare_parameter<std::string>("gimbal_pitch_topic", "");
    this->declare_parameter<double>("initial_gimbal_yaw_deg", 0.0);
    this->declare_parameter<double>("initial_gimbal_pitch_deg", 0.0);
    this->declare_parameter<double>("fixed_pitch_deg", 15.0);

    // Get parameters
    confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
    detection_topic_ = this->get_parameter("detection_topic").as_string();
    gimbal_yaw_topic_ = this->get_parameter("gimbal_yaw_topic").as_string();
    gimbal_pitch_topic_ = this->get_parameter("gimbal_pitch_topic").as_string();
    gimbal_yaw_rad_ = deg2rad(this->get_parameter("initial_gimbal_yaw_deg").as_double());
    gimbal_pitch_rad_ = deg2rad(this->get_parameter("initial_gimbal_pitch_deg").as_double());
    fixed_pitch_rad_ = deg2rad(this->get_parameter("fixed_pitch_deg").as_double());

    // Parameter callback for dynamic reconfiguration
    param_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&DetectorNode::parameter_callback, this, std::placeholders::_1));

        const auto source_dir = std::filesystem::path(__FILE__).parent_path();
        const auto default_config_dir = source_dir.parent_path().parent_path() / "config";
        const std::string default_camera_config = (default_config_dir / "CameraConfig.yaml").string();
        const std::string default_aimauto_config = (default_config_dir / "AimautoConfig.yaml").string();

        camera_config_path_ = this->declare_parameter<std::string>("camera_config_path", default_camera_config);
        aimauto_config_path_ = this->declare_parameter<std::string>("aimauto_config_path", default_aimauto_config);

                try {
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
                                std::bind(&DetectorNode::gimbalYawCallback, this, std::placeholders::_1));
                        }
                        if (!gimbal_pitch_topic_.empty()) {
                            gimbal_pitch_sub_ = create_subscription<std_msgs::msg::Float32>(
                                gimbal_pitch_topic_, 10,
                                std::bind(&DetectorNode::gimbalPitchCallback, this, std::placeholders::_1));
                        }
                        RCLCPP_INFO(get_logger(), "ArmorPnPSolver initialized with camera config '%s'", camera_config_path_.c_str());
                } catch (const std::exception & e) {
                        RCLCPP_ERROR(get_logger(), "Failed to initialize ArmorPnPSolver: %s", e.what());
                }

    const std::filesystem::path model_rel = source_dir.parent_path().parent_path() / "model" / "best.onnx";
    const std::string model_path = model_rel.string();
    try {
      model_ = core_.read_model(model_path);
      RCLCPP_INFO(get_logger(), "Loaded model: %s", model_path.c_str());

      input_port_ = model_->input();
      output_port_ = model_->output();

      const auto input_shape = input_port_.get_shape();
      const auto output_shape = output_port_.get_shape();
      if (input_shape.size() != 4 || output_shape.size() != 3) {
        throw std::runtime_error("Unexpected model input/output rank");
      }
      target_img_size_ = static_cast<int>(input_shape[2]);
      num_attrs_ = static_cast<int>(output_shape[1]);
      num_predictions_ = static_cast<int>(output_shape[2]);
            num_classes_ = num_attrs_ - 4;
      if (num_classes_ <= 0) {
        throw std::runtime_error("Model must have at least one class");
      }

      compiled_model_ = core_.compile_model(model_, "CPU");
      infer_request_ = compiled_model_.create_infer_request();
      RCLCPP_INFO(get_logger(), "OpenVINO model ready (predictions=%d, classes=%d)", num_predictions_, num_classes_);
    } catch (const std::exception &e) {
      RCLCPP_FATAL(get_logger(), "Failed to initialize model: %s", e.what());
      throw;
    }

    image_sub_ = create_subscription<aimbot_msgs::msg::CameraImage>(
        "camera_image", rclcpp::SensorDataQoS().reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT),
        std::bind(&DetectorNode::image_callback, this, std::placeholders::_1));

    detection_pub_ = create_publisher<aimbot_msgs::msg::DetectionArray>(detection_topic_, 10);
  }

private:
  rcl_interfaces::msg::SetParametersResult parameter_callback(const std::vector<rclcpp::Parameter> &parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto &param : parameters) {
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
        void image_callback(const aimbot_msgs::msg::CameraImage::SharedPtr msg) {
        try {
            // 获取图像和时间戳
            const sensor_msgs::msg::Image &image_msg = msg->image;
            const uint64_t camera_timestamp = msg->timestamp;

            cv_bridge::CvImagePtr cv_ptr;
            if (image_msg.encoding == sensor_msgs::image_encodings::BGR8) {
                cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
            } else if (image_msg.encoding == sensor_msgs::image_encodings::RGB8) {
                cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8);
            } else {
                cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Converted unsupported encoding %s to BGR8",
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

            const int orig_w = frame.cols;
            const int orig_h = frame.rows;

            const LetterboxResult preprocess = letterbox(frame, target_img_size_);

            cv::Mat blob = cv::dnn::blobFromImage(preprocess.image, 1.0 / 255.0,
                                                                                        cv::Size(target_img_size_, target_img_size_), cv::Scalar(), true, false);

            ov::Tensor input_tensor(input_port_.get_element_type(), input_port_.get_shape(), blob.ptr<float>());
            infer_request_.set_input_tensor(input_tensor);
            infer_request_.infer();

            const ov::Tensor output_tensor = infer_request_.get_output_tensor(output_port_.get_index());
            const auto &out_shape = output_tensor.get_shape();
            const float *output_data = output_tensor.data<float>();

            const int predictions = static_cast<int>(out_shape[2]);
            const int attrs = static_cast<int>(out_shape[1]);
            if (attrs <= 4) {
                throw std::runtime_error("Model output has insufficient attributes per prediction");
            }
            const int classes = attrs - 4;

            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;
            boxes.reserve(predictions);
            confidences.reserve(predictions);
            class_ids.reserve(predictions);

            const float conf_threshold = confidence_threshold_;
            const float iou_threshold = 0.45f;

            for (int i = 0; i < predictions; ++i) {
                const float cx = output_data[i + 0 * predictions];
                const float cy = output_data[i + 1 * predictions];
                const float w = output_data[i + 2 * predictions];
                const float h = output_data[i + 3 * predictions];
                float max_class_prob = 0.0f;
                int best_class = -1;
                for (int c = 0; c < classes; ++c) {
                    const float class_prob = output_data[i + (4 + c) * predictions];
                    if (class_prob > max_class_prob) {
                        max_class_prob = class_prob;
                        best_class = c;
                    }
                }

                if (best_class < 0 || max_class_prob < conf_threshold) {
                    continue;
                }

                const float x0 = (cx - w / 2.0f - static_cast<float>(preprocess.pad_x)) / preprocess.scale;
                const float y0 = (cy - h / 2.0f - static_cast<float>(preprocess.pad_y)) / preprocess.scale;
                const float x1 = (cx + w / 2.0f - static_cast<float>(preprocess.pad_x)) / preprocess.scale;
                const float y1 = (cy + h / 2.0f - static_cast<float>(preprocess.pad_y)) / preprocess.scale;

                int left = static_cast<int>(std::round(std::clamp(x0, 0.0f, static_cast<float>(orig_w - 1))));
                int top = static_cast<int>(std::round(std::clamp(y0, 0.0f, static_cast<float>(orig_h - 1))));
                int right = static_cast<int>(std::round(std::clamp(x1, 0.0f, static_cast<float>(orig_w - 1))));
                int bottom = static_cast<int>(std::round(std::clamp(y1, 0.0f, static_cast<float>(orig_h - 1))));

                const int box_w = std::max(0, right - left);
                const int box_h = std::max(0, bottom - top);
                if (box_w == 0 || box_h == 0) {
                    continue;
                }

                boxes.emplace_back(left, top, box_w, box_h);
                confidences.emplace_back(max_class_prob);
                class_ids.emplace_back(best_class);
            }

            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);

            // Publish detection results
            aimbot_msgs::msg::DetectionArray detection_msg;
            detection_msg.header = image_msg.header;
            detection_msg.timestamp = camera_timestamp;
            detection_msg.confidence_threshold = conf_threshold;
            detection_msg.detections.reserve(indices.size());

            for (int idx : indices) {
                const cv::Rect &box = boxes[idx];
                const float confidence = confidences[idx];
                const int class_id = class_ids[idx];

                aimbot_msgs::msg::Detection detection;
                detection.header = image_msg.header;
                detection.class_id = class_id;
                detection.class_name = "class_" + std::to_string(class_id);  // You can map to actual class names
                detection.confidence = confidence;

                // Create bbox polygon
                geometry_msgs::msg::Point32 p1, p2, p3, p4;
                p1.x = static_cast<float>(box.x);
                p1.y = static_cast<float>(box.y);
                p2.x = static_cast<float>(box.x + box.width);
                p2.y = static_cast<float>(box.y);
                p3.x = static_cast<float>(box.x + box.width);
                p3.y = static_cast<float>(box.y + box.height);
                p4.x = static_cast<float>(box.x);
                p4.y = static_cast<float>(box.y + box.height);
                detection.bbox.points = {p1, p2, p3, p4};

                // Center point
                detection.center.x = static_cast<float>(box.x + box.width / 2.0);
                detection.center.y = static_cast<float>(box.y + box.height / 2.0);
                detection.center.z = 0.0f;

                // PnP solving
                cv::Vec3d rvec(0.0, 0.0, 0.0);
                cv::Vec3d tvec(0.0, 0.0, 0.0);
                std::vector<cv::Point2f> image_points;
                bool pnp_success = false;
                if (pnp_solver_) {
                    if (pnp_solver_->solve(frame, box, class_id, rvec, tvec, &image_points)) {
                        RCLCPP_DEBUG(get_logger(), "PnP solved: class=%d, t=(%.2f, %.2f, %.2f)",
                                     class_id, tvec[0], tvec[1], tvec[2]);
                        pnp_success = true;
                    } else {
                        RCLCPP_DEBUG(get_logger(), "PnP solver failed for detection class=%d", class_id);
                    }
                }

                // Only publish if PnP succeeded
                                if (pnp_success && image_points.size() == 4) {
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

                                        const auto & object_points =
                                            (class_id == 0 || class_id == 7) ? big_armor_object_points_ : small_armor_object_points_;

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
                }
            }

            detection_pub_->publish(detection_msg);

            // Calculate and display FPS
            frame_count_++;
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_time_).count();
            
            if (elapsed >= 1000) {  // Update FPS every second
                fps_ = static_cast<double>(frame_count_) / (elapsed / 1000.0);
                frame_count_ = 0;
                last_fps_time_ = current_time;
            }

#ifdef DEBUG_VISUALIZATION
            // Debug visualization with FPS
            std::string fps_text = cv::format("FPS: %.1f", fps_);
            cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 255, 0), 2);

            for (int idx : indices) {
                const cv::Rect &box = boxes[idx];
                cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 1);
                const std::string label = "cls " + std::to_string(class_ids[idx]) +
                                                                    " (" + cv::format("%.2f", confidences[idx]) + ")";
                cv::putText(frame, label, cv::Point(box.x, std::max(10, box.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                        cv::Scalar(0, 255, 0), 1);

                // Draw coordinate axes for PnP result with optimized yaw
                if (pnp_solver_) {
                    cv::Vec3d rvec, tvec;
                    std::vector<cv::Point2f> image_points;
                    if (pnp_solver_->solve(frame, box, class_ids[idx], rvec, tvec, &image_points) && image_points.size() == 4) {
                        // Calculate optimized yaw
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

                        const auto & object_points =
                            (class_ids[idx] == 0 || class_ids[idx] == 7) ? big_armor_object_points_ : small_armor_object_points_;

                        const double optimized_yaw = optimizeYaw(
                            world_to_camera_rot,
                            camera_translation_world,
                            world_translation,
                            object_points,
                            image_points_array,
                            fixed_pitch_rad);

                        // Construct rvec with optimized yaw
                        Eigen::Matrix3d mat_y;
                        mat_y << std::cos(optimized_yaw), 0.0, std::sin(optimized_yaw),
                            0.0, 1.0, 0.0,
                            -std::sin(optimized_yaw), 0.0, std::cos(optimized_yaw);

                        Eigen::Matrix3d mat_x;
                        mat_x << 1.0, 0.0, 0.0,
                            0.0, std::cos(fixed_pitch_rad), -std::sin(fixed_pitch_rad),
                            0.0, std::sin(fixed_pitch_rad), std::cos(fixed_pitch_rad);

                        const Eigen::Matrix3d rotation_mount = []() {
                            Eigen::Matrix3d rotation;
                            rotation << 0.0, 0.0, 1.0,
                                -1.0, 0.0, 0.0,
                                0.0, -1.0, 0.0;
                            return rotation;
                        }();

                        const Eigen::Matrix3d rotation_total = world_to_camera_rot * rotation_mount * mat_y * mat_x;

                        cv::Mat rotation_cv(3, 3, CV_64F);
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                rotation_cv.at<double>(r, c) = rotation_total(r, c);
                            }
                        }

                        cv::Rodrigues(rotation_cv, rvec);

                        cv::drawFrameAxes(frame, pnp_solver_->camera_matrix(), pnp_solver_->dist_coeffs(), rvec, tvec, 100, 2);
                    }
                }
            }

            cv::imshow("Detection", frame);
            cv::waitKey(1);
#endif

            // Print detection info with FPS
            RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 1000, 
                "Published %zu detections (FPS: %.1f, threshold: %.3f)", 
                detection_msg.detections.size(), fps_, conf_threshold);
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
        } catch (const std::exception &e) {
            RCLCPP_ERROR(get_logger(), "Detection failed: %s", e.what());
        }
    }

        static double deg2rad(double deg)
        {
            return deg * kPi / 180.0;
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

    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    ov::Output<const ov::Node> input_port_;
    ov::Output<const ov::Node> output_port_;
    rclcpp::Subscription<aimbot_msgs::msg::CameraImage>::SharedPtr image_sub_;
    rclcpp::Publisher<aimbot_msgs::msg::DetectionArray>::SharedPtr detection_pub_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    std::unique_ptr<aim_auto::ArmorPnPSolver> pnp_solver_;
    std::string camera_config_path_;
    std::string aimauto_config_path_;

    double confidence_threshold_ = 0.50;
    std::string detection_topic_ = "detections";
    std::string gimbal_yaw_topic_;
    std::string gimbal_pitch_topic_;

    // FPS calculation
    std::chrono::steady_clock::time_point last_fps_time_;
    int frame_count_ = 0;
    double fps_ = 0.0;

    int target_img_size_ = 640;
    int num_attrs_ = 0;
    int num_predictions_ = 0;
    int num_classes_ = 0;

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr gimbal_yaw_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr gimbal_pitch_sub_;

    double gimbal_yaw_rad_ = 0.0;
    double gimbal_pitch_rad_ = 0.0;
    double fixed_pitch_rad_ = 0.0;
    double vector_x_ = 0.0;
    double vector_y_ = 0.0;
    double vector_z_ = 0.0;

    mutable std::mutex transform_mutex_;
    Eigen::Matrix3d camera_to_world_rot_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d world_to_camera_rot_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d camera_translation_world_{0.0, 0.0, 0.0};
    bool transforms_dirty_ = false;

    std::vector<cv::Point3d> small_armor_object_points_;
    std::vector<cv::Point3d> big_armor_object_points_;
};



int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DetectorNode>());
    rclcpp::shutdown();
    return 0;
}
