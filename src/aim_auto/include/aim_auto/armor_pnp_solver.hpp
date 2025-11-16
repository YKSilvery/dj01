#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace aim_auto
{

struct PnPResult
{
  cv::Vec3d rvec;
  cv::Vec3d tvec;
};

class ArmorPnPSolver
{
public:
  ArmorPnPSolver(const std::string & camera_config_path, const std::string & aimauto_config_path);

  bool solve(
    const cv::Mat & frame,
    const cv::Rect & detection_box,
    int class_id,
    cv::Vec3d & rvec,
    cv::Vec3d & tvec,
    std::vector<cv::Point2f> * image_points = nullptr) const;

  const cv::Mat & camera_matrix() const { return camera_matrix_; }
  const cv::Mat & dist_coeffs() const { return dist_coeffs_; }

private:
  struct RectCandidate
  {
    cv::Point2f center;
    cv::Point2f top;
    cv::Point2f bottom;
  };

  void loadCameraConfig(const std::string & camera_config_path);
  void loadAimautoConfig(const std::string & aimauto_config_path);

  std::vector<cv::Point3f> small_armor_object_points_;
  std::vector<cv::Point3f> big_armor_object_points_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
};

}  // namespace aim_auto
