#include "aim_auto/ref_detection.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace aim_auto
{
namespace
{
constexpr double kColorDiffThreshold = 20.0;

int channelIndex(TargetColor color)
{
  return color == TargetColor::RED ? 2 : 0;
}

int toInt(TargetColor color)
{
  return color == TargetColor::BLUE ? 1 : 0;
}
}  // namespace

class NumberClassifier
{
public:
  NumberClassifier(
    const std::string & model_path,
    const std::string & label_path,
    double threshold,
    std::vector<std::string> ignore_classes = {"negative"})
  : ignore_classes_(std::move(ignore_classes)), threshold_(threshold)
  {
    net_ = cv::dnn::readNetFromONNX(model_path);

    std::ifstream label_file(label_path);
    if (!label_file.is_open()) {
      throw std::runtime_error("Failed to open label file: " + label_path);
    }
    std::string line;
    while (std::getline(label_file, line)) {
      if (!line.empty()) {
        class_names_.push_back(line);
      }
    }
    if (class_names_.empty()) {
      throw std::runtime_error("Label file is empty: " + label_path);
    }
  }

  void extractNumbers(const cv::Mat & src, std::vector<UnsolvedArmor> & armors, TargetColor color)
  {
    constexpr int light_length = 12;
    constexpr int warp_height = 28;
    constexpr int small_armor_width = 32;
    constexpr int large_armor_width = 54;
    const cv::Size roi_size(20, 28);
    const cv::Size input_size(28, 28);

    for (auto & armor : armors) {
      cv::Point2f lights_vertices[4] = {
        armor.left_light.bottom,
        armor.left_light.top,
        armor.right_light.top,
        armor.right_light.bottom};

      const int top_light_y = (warp_height - light_length) / 2 - 1;
      const int bottom_light_y = top_light_y + light_length;
      const int warp_width = armor.type == ArmorType::SMALL ? small_armor_width : large_armor_width;

      cv::Point2f target_vertices[4] = {
        cv::Point(0, bottom_light_y),
        cv::Point(0, top_light_y),
        cv::Point(warp_width - 1, top_light_y),
        cv::Point(warp_width - 1, bottom_light_y)};

      cv::Mat number_image;
      const cv::Mat rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
      cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

      number_image = number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

      cv::Mat channels[3];
      cv::split(number_image, channels);
      const int channel = channelIndex(color);
      cv::Mat & select = channels[channel];
      for (int row = 0; row < select.rows; ++row) {
        for (int col = 0; col < select.cols; ++col) {
          if (select.at<uchar>(row, col) > 100) {
            select.at<uchar>(row, col) = 0;
          }
        }
      }

      cv::threshold(select, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
      cv::resize(number_image, number_image, input_size);
      armor.number_img = number_image;
    }
  }

  void classify(std::vector<UnsolvedArmor> & armors)
  {
    for (auto & armor : armors) {
      cv::Mat input = armor.number_img / 255.0;
      cv::Mat blob;
      cv::dnn::blobFromImage(input, blob);
      net_.setInput(blob);
      cv::Mat outputs = net_.forward().clone();

      double confidence = 0.0;
      cv::Point class_id_point;
      cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
      const int label_id = class_id_point.x;

      if (label_id < 0 || label_id >= static_cast<int>(class_names_.size())) {
        continue;
      }

      armor.confidence = static_cast<float>(confidence);
      armor.number = class_names_[label_id];

      std::ostringstream result_ss;
      result_ss << armor.number << ": " << std::fixed << std::setprecision(1)
                << (armor.confidence * 100.0f) << "%";
      armor.classification_result = result_ss.str();
    }

    armors.erase(
      std::remove_if(
        armors.begin(),
        armors.end(),
        [this](const UnsolvedArmor & armor) {
          if (armor.is_apriltag) {
            return false;
          }
          if (armor.confidence < threshold_) {
            return true;
          }
          for (const auto & ignore_class : ignore_classes_) {
            if (armor.number == ignore_class) {
              return true;
            }
          }
          bool mismatch = false;
          if (armor.type == ArmorType::LARGE) {
            mismatch = armor.number == "outpost" || armor.number == "2" || armor.number == "guard";
          } else if (armor.type == ArmorType::SMALL) {
            mismatch = armor.number == "1" || armor.number == "base";
          }
          return mismatch;
        }),
      armors.end());
  }

private:
  cv::dnn::Net net_;
  std::vector<std::string> class_names_;
  std::vector<std::string> ignore_classes_;
  double threshold_ = 0.8;
};

Light::Light(const cv::RotatedRect & box)
: top(0, 0), bottom(0, 0), raw_box(box)
{
  cv::Point2f points[4];
  box.points(points);
  std::sort(points, points + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
  top = (points[0] + points[1]) * 0.5f;
  bottom = (points[2] + points[3]) * 0.5f;
  center = box.center;
  length = cv::norm(top - bottom);
  width = cv::norm(points[0] - points[1]);
  tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y)) / CV_PI * 180.0f;
}

cv::Rect Light::boundingRect() const
{
  return raw_box.boundingRect();
}

cv::Rect Light::boundingRect2f() const
{
  return raw_box.boundingRect();
}

UnsolvedArmor::UnsolvedArmor(const Light & l1, const Light & l2)
{
  if (l1.center.x < l2.center.x) {
    left_light = l1;
    right_light = l2;
  } else {
    left_light = l2;
    right_light = l1;
  }
  center = (left_light.center + right_light.center) * 0.5f;
}

DetectionParams loadDetectionParams(const std::string & path)
{
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Failed to open detection config: " + path);
  }
  DetectionParams params;
  fs["min_ratio"] >> params.light.min_ratio;
  fs["max_ratio"] >> params.light.max_ratio;
  fs["max_angle_l"] >> params.light.max_angle;
  fs["min_light_ratio"] >> params.armor.min_light_ratio;
  fs["min_small_center_distance"] >> params.armor.min_small_center_distance;
  fs["max_small_center_distance"] >> params.armor.max_small_center_distance;
  fs["min_large_center_distance"] >> params.armor.min_large_center_distance;
  fs["max_large_center_distance"] >> params.armor.max_large_center_distance;
  fs["max_angle_a"] >> params.armor.max_angle;
  fs["num_threshold"] >> params.num_threshold;
  fs["blue_threshold"] >> params.blue_threshold;
  fs["red_threshold"] >> params.red_threshold;
  return params;
}

RefArmorDetector::RefArmorDetector(
  const DetectionParams & params,
  const std::string & model_path,
  const std::string & label_path)
: params_(params)
{
  classifier_ = std::make_unique<NumberClassifier>(model_path, label_path, params.num_threshold);
}

RefArmorDetector::~RefArmorDetector() = default;

std::vector<UnsolvedArmor> RefArmorDetector::detect(const cv::Mat & frame, TargetColor color)
{
  current_color_ = color;
  binary_threshold_ = color == TargetColor::RED ? params_.red_threshold : params_.blue_threshold;

  cv::Mat binary = preprocessImage(frame);
  auto lights = findLights(frame, binary);
  auto armors = matchLights(lights);
  if (!armors.empty()) {
    classifier_->extractNumbers(frame, armors, color);
    classifier_->classify(armors);
  }
  return armors;
}

cv::Mat RefArmorDetector::preprocessImage(const cv::Mat & rgb_img) const
{
  cv::Mat gray_img;
  cv::cvtColor(rgb_img, gray_img, cv::COLOR_BGR2GRAY);
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, binary_threshold_, 255, cv::THRESH_BINARY);
  return binary_img;
}

std::vector<Light> RefArmorDetector::findLights(const cv::Mat & rgb_img, const cv::Mat & binary_img) const
{
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<Light> lights;
  lights.reserve(contours.size());

  for (const auto & contour : contours) {
    if (contour.size() < 6) {
      continue;
    }

    const auto r_rect = cv::minAreaRect(contour);
    Light light(r_rect);

    const float ratio = light.width / (light.length + 1e-5f);
    const bool ratio_ok = params_.light.min_ratio < ratio && ratio < params_.light.max_ratio;
    const bool angle_ok = light.tilt_angle < params_.light.max_angle;
    const bool size_ok = light.length * light.width < 12800.0 && light.length > 10.0;
    if (!(ratio_ok && angle_ok && size_ok)) {
      continue;
    }

    const cv::Rect rect = light.boundingRect();
    if (rect.x < 0 || rect.y < 0 || rect.x + rect.width > rgb_img.cols || rect.y + rect.height > rgb_img.rows) {
      continue;
    }

    int sum_r = 0;
    int sum_b = 0;
    const auto roi = rgb_img(rect);
    for (int row = 0; row < roi.rows; ++row) {
      for (int col = 0; col < roi.cols; ++col) {
        const cv::Vec3b pixel = roi.at<cv::Vec3b>(row, col);
        sum_b += pixel[0];
        sum_r += pixel[2];
      }
    }

    if (std::abs(sum_r - sum_b) / static_cast<double>(contour.size()) > kColorDiffThreshold) {
      light.color = sum_r > sum_b ? 0 : 1;
    }
    light.center = r_rect.center;
    lights.emplace_back(light);
  }

  std::sort(lights.begin(), lights.end(), [](const Light & lhs, const Light & rhs) {
    return lhs.center.x < rhs.center.x;
  });

  return lights;
}

std::vector<UnsolvedArmor> RefArmorDetector::matchLights(const std::vector<Light> & lights) const
{
  std::vector<UnsolvedArmor> armors;

  for (auto light_1 = lights.begin(); light_1 != lights.end(); ++light_1) {
    if (light_1->color != toInt(current_color_)) {
      continue;
    }
    for (auto light_2 = light_1 + 1; light_2 != lights.end(); ++light_2) {
      if (light_2->color != toInt(current_color_)) {
        continue;
      }
      if (containLight(static_cast<int>(light_1 - lights.begin()), static_cast<int>(light_2 - lights.begin()), lights)) {
        continue;
      }
      auto type = isArmor(*light_1, *light_2);
      if (type != ArmorType::INVALID) {
        UnsolvedArmor armor(*light_1, *light_2);
        armor.type = type;
        armors.emplace_back(armor);
      }
    }
  }
  return armors;
}

bool RefArmorDetector::containLight(int i, int j, const std::vector<Light> & lights) const noexcept
{
  const Light & light_1 = lights.at(i);
  const Light & light_2 = lights.at(j);
  std::vector<cv::Point2f> points{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
  const auto bounding_rect = cv::boundingRect(points);
  const double avg_length = (light_1.length + light_2.length) / 2.0;
  const double avg_width = (light_1.width + light_2.width) / 2.0;

  for (int k = i + 1; k < j; ++k) {
    const Light & test_light = lights.at(k);
    if (test_light.width > 2.0 * avg_width) {
      continue;
    }
    if (test_light.length < 0.5 * avg_length) {
      continue;
    }
    if (bounding_rect.contains(test_light.top) ||
        bounding_rect.contains(test_light.bottom) ||
        bounding_rect.contains(test_light.center)) {
      return true;
    }
  }
  return false;
}

ArmorType RefArmorDetector::isArmor(const Light & light_1, const Light & light_2) const
{
  const float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                                   : light_2.length / light_1.length;
  const bool light_ratio_ok = light_length_ratio > params_.armor.min_light_ratio;

  const float avg_light_length = static_cast<float>((light_1.length + light_2.length) / 2.0);
  const float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
  const bool center_distance_ok =
    (params_.armor.min_small_center_distance <= center_distance &&
     center_distance < params_.armor.max_small_center_distance) ||
    (params_.armor.min_large_center_distance <= center_distance &&
     center_distance < params_.armor.max_large_center_distance);

  const cv::Point2f diff = light_1.center - light_2.center;
  const float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180.0f;
  const bool angle_ok = angle < params_.armor.max_angle;

  if (!(light_ratio_ok && center_distance_ok && angle_ok)) {
    return ArmorType::INVALID;
  }
  return center_distance > params_.armor.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
}

std::string to_color_string(TargetColor color)
{
  return color == TargetColor::RED ? "red" : "blue";
}

}  // namespace aim_auto
