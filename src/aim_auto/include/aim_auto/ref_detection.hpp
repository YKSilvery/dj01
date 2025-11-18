#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace aim_auto
{

enum class ArmorType { SMALL, LARGE, INVALID };

enum class TargetColor { RED = 0, BLUE = 1 };

struct Light
{
  Light() = default;
  explicit Light(const cv::RotatedRect & box);

  int color = 0;
  cv::Point2f top;
  cv::Point2f bottom;
  cv::Point2f center;
  double length = 0.0;
  double width = 0.0;
  float tilt_angle = 0.0f;
  cv::RotatedRect raw_box;

  cv::Rect boundingRect() const;
  cv::Rect boundingRect2f() const;
};

struct UnsolvedArmor
{
  UnsolvedArmor() = default;
  UnsolvedArmor(const Light & l1, const Light & l2);

  Light left_light;
  Light right_light;
  cv::Point2f center;
  ArmorType type = ArmorType::INVALID;

  cv::Mat number_img;
  std::string number;
  float confidence = 0.0f;
  std::string classification_result;
  bool is_apriltag = false;
};

struct LightParams
{
  double min_ratio = 0.0;
  double max_ratio = 0.0;
  double max_angle = 0.0;
};

struct ArmorParams
{
  double min_light_ratio = 0.0;
  double min_small_center_distance = 0.0;
  double max_small_center_distance = 0.0;
  double min_large_center_distance = 0.0;
  double max_large_center_distance = 0.0;
  double max_angle = 0.0;
};

struct DetectionParams
{
  LightParams light;
  ArmorParams armor;
  double num_threshold = 0.8;
  int blue_threshold = 65;
  int red_threshold = 70;
};

DetectionParams loadDetectionParams(const std::string & path);

class NumberClassifier;

class RefArmorDetector
{
public:
  RefArmorDetector(const DetectionParams & params, const std::string & model_path, const std::string & label_path);
  ~RefArmorDetector();

  std::vector<UnsolvedArmor> detect(const cv::Mat & frame, TargetColor color);

  const DetectionParams & params() const { return params_; }

private:
  cv::Mat preprocessImage(const cv::Mat & rgb_img) const;
  std::vector<Light> findLights(const cv::Mat & rgb_img, const cv::Mat & binary_img) const;
  std::vector<UnsolvedArmor> matchLights(const std::vector<Light> & lights) const;
  bool containLight(int i, int j, const std::vector<Light> & lights) const noexcept;
  ArmorType isArmor(const Light & light_1, const Light & light_2) const;

  DetectionParams params_{};
  int binary_threshold_ = 0;
  TargetColor current_color_ = TargetColor::RED;
  std::unique_ptr<NumberClassifier> classifier_;
};

std::string to_color_string(TargetColor color);

}  // namespace aim_auto
