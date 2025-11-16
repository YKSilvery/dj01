#include "aim_auto/armor_pnp_solver.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace aim_auto
{

ArmorPnPSolver::ArmorPnPSolver(const std::string & camera_config_path, const std::string & aimauto_config_path)
{
  loadCameraConfig(camera_config_path);
  loadAimautoConfig(aimauto_config_path);
}

bool ArmorPnPSolver::solve(
  const cv::Mat & frame,
  const cv::Rect & detection_box,
  int class_id,
  cv::Vec3d & rvec,
  cv::Vec3d & tvec,
  std::vector<cv::Point2f> * image_points) const
{
  //std::cout << "Debug: Starting PnP solve for box: (" << detection_box.x << "," << detection_box.y << "," << detection_box.width << "," << detection_box.height << ")" << std::endl;

  if (frame.empty()) {
    //std::cout << "Debug: Frame is empty" << std::endl;
    return false;
  }

  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else if (frame.channels() == 4) {
    cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
  } else {
    gray = frame.clone();
  }

  const cv::Rect image_bounds(0, 0, frame.cols, frame.rows);
  cv::Rect roi = detection_box & image_bounds;
  if (roi.width <= 5 || roi.height <= 5) {
    //std::cout << "Debug: ROI too small: " << roi.width << "x" << roi.height << std::endl;
    return false;
  }

  //std::cout << "Debug: ROI size: " << roi.width << "x" << roi.height << std::endl;

  cv::Mat roi_gray = gray(roi).clone();
  cv::Mat binary;
  cv::threshold(roi_gray, binary, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);

  // Apply morphological operations to suppress noise
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

  // Clear border pixels to avoid boundary connections
  binary.row(0).setTo(0);
  binary.row(binary.rows - 1).setTo(0);
  binary.col(0).setTo(0);
  binary.col(binary.cols - 1).setTo(0);

  // Debug: show binary image
//   cv::imshow("Binary ROI", binary);
//   cv::waitKey(1);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  //std::cout << "Debug: Found " << contours.size() << " contours" << std::endl;

  // Debug: show binary image with contours
  cv::Mat debug_image;
  cv::cvtColor(binary, debug_image, cv::COLOR_GRAY2BGR);
  // cv::drawContours(debug_image, contours, -1, cv::Scalar(0, 0, 255), 1);  // Removed contour drawing

  if (contours.size() < 2) {
    //std::cout << "Debug: Less than 2 contours, returning false" << std::endl;
    return false;
  }

  std::vector<RectCandidate> candidates;
  candidates.reserve(contours.size());

  for (const auto & contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < 10.0) {
      continue;
    }

    cv::RotatedRect rect = cv::minAreaRect(contour);
    if (rect.size.width < 2.0 || rect.size.height < 2.0) {
      continue;
    }

    cv::Vec4f line_params;
    cv::fitLine(contour, line_params, cv::DIST_L2, 0, 0.01, 0.01);

    cv::Point2f direction(line_params[0], line_params[1]);
    const float norm = std::sqrt(direction.x * direction.x + direction.y * direction.y);
    if (norm < 1e-4f) {
      continue;
    }
    direction /= norm;

    const cv::Point2f line_point(line_params[2], line_params[3]);
    float min_proj = std::numeric_limits<float>::max();
    float max_proj = std::numeric_limits<float>::lowest();

    for (const auto & contour_point : contour) {
      const cv::Point2f pt(static_cast<float>(contour_point.x), static_cast<float>(contour_point.y));
      const float proj = (pt - line_point).dot(direction);
      min_proj = std::min(min_proj, proj);
      max_proj = std::max(max_proj, proj);
    }

    if (!std::isfinite(min_proj) || !std::isfinite(max_proj) || (max_proj - min_proj) < 3.0f) {
      continue;
    }

    const cv::Point2f roi_offset(static_cast<float>(roi.x), static_cast<float>(roi.y));
    const cv::Point2f endpoint_min = line_point + direction * min_proj + roi_offset;
    const cv::Point2f endpoint_max = line_point + direction * max_proj + roi_offset;

    RectCandidate candidate;
    if (endpoint_min.y <= endpoint_max.y) {
      candidate.top = endpoint_min;
      candidate.bottom = endpoint_max;
    } else {
      candidate.top = endpoint_max;
      candidate.bottom = endpoint_min;
    }
    candidate.center = 0.5f * (candidate.top + candidate.bottom);

    candidates.push_back(candidate);
  }

  //std::cout << "Debug: Valid candidates: " << candidates.size() << std::endl;

  if (candidates.size() < 2) {
    //std::cout << "Debug: Less than 2 candidates, returning false" << std::endl;
    return false;
  }

  std::sort(candidates.begin(), candidates.end(), [](const RectCandidate & lhs, const RectCandidate & rhs) {
    return lhs.center.x < rhs.center.x;
  });

  const RectCandidate & left = candidates.front();
  const RectCandidate & right = candidates.back();

  std::vector<cv::Point2f> img_points{
    left.top,
    right.top,
    right.bottom,
    left.bottom};

  //std::cout << "Debug: Image points: ";
  //for (const auto & pt : img_points) {
  //  std::cout << "(" << pt.x << "," << pt.y << ") ";
  //}
  //std::cout << std::endl;

  if (image_points) {
    *image_points = img_points;
  }

  // Draw PnP points on debug_image (convert to ROI coordinates)
  for (const auto & pt : img_points) {
    cv::Point roi_pt(static_cast<int>(pt.x - roi.x), static_cast<int>(pt.y - roi.y));
    if (roi_pt.x >= 0 && roi_pt.x < debug_image.cols && roi_pt.y >= 0 && roi_pt.y < debug_image.rows) {
      cv::circle(debug_image, roi_pt, 3, cv::Scalar(0, 255, 0), -1);  // Green circles for points
    }
  }

  cv::imshow("Binary ROI with Contours", debug_image);
  cv::waitKey(1);

  // Select object points based on class_id

  const auto & object_points = (class_id == 0 || class_id == 7) ? big_armor_object_points_ : small_armor_object_points_;

  if (object_points.size() != img_points.size()) {
    //std::cout << "Debug: Object points size mismatch: " << object_points.size() << " vs " << img_points.size() << std::endl;
    return false;
  }

  const bool success = cv::solvePnP(
    object_points,
    img_points,
    camera_matrix_,
    dist_coeffs_,
    rvec,
    tvec,
    false,
    cv::SOLVEPNP_ITERATIVE);

  //std::cout << "Debug: solvePnP success: " << (success ? "true" : "false") << std::endl;
  if (success) {
    //std::cout << "Debug: rvec: (" << rvec[0] << "," << rvec[1] << "," << rvec[2] << "), tvec: (" << tvec[0] << "," << tvec[1] << "," << tvec[2] << ")" << std::endl;

    // Ensure armor plate faces the camera (Z-axis points towards camera)
    if (tvec[2] < 0) {
      rvec = -rvec;
      tvec = -tvec;
      //std::cout << "Debug: Flipped pose to face camera" << std::endl;
    }
  }

  return success;
}

void ArmorPnPSolver::loadCameraConfig(const std::string & camera_config_path)
{
  cv::FileStorage fs(camera_config_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Failed to open camera config: " + camera_config_path);
  }

  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
  double k1 = 0.0, k2 = 0.0, k3 = 0.0, p1 = 0.0, p2 = 0.0;
  fs["fx"] >> fx;
  fs["fy"] >> fy;
  fs["cx"] >> cx;
  fs["cy"] >> cy;
  fs["k1"] >> k1;
  fs["k2"] >> k2;
  fs["k3"] >> k3;
  fs["p1"] >> p1;
  fs["p2"] >> p2;

  camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
  dist_coeffs_ = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
}

void ArmorPnPSolver::loadAimautoConfig(const std::string & aimauto_config_path)
{
  cv::FileStorage fs(aimauto_config_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Failed to open aimauto config: " + aimauto_config_path);
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
  small_armor_object_points_.emplace_back(static_cast<float>(-half_small_a), static_cast<float>(-half_small_b), 0.0f);
  small_armor_object_points_.emplace_back(static_cast<float>(half_small_a), static_cast<float>(-half_small_b), 0.0f);
  small_armor_object_points_.emplace_back(static_cast<float>(half_small_a), static_cast<float>(half_small_b), 0.0f);
  small_armor_object_points_.emplace_back(static_cast<float>(-half_small_a), static_cast<float>(half_small_b), 0.0f);

  big_armor_object_points_.clear();
  big_armor_object_points_.reserve(4);
  big_armor_object_points_.emplace_back(static_cast<float>(-half_big_a), static_cast<float>(-half_big_b), 0.0f);
  big_armor_object_points_.emplace_back(static_cast<float>(half_big_a), static_cast<float>(-half_big_b), 0.0f);
  big_armor_object_points_.emplace_back(static_cast<float>(half_big_a), static_cast<float>(half_big_b), 0.0f);
  big_armor_object_points_.emplace_back(static_cast<float>(-half_big_a), static_cast<float>(half_big_b), 0.0f);
}

}  // namespace aim_auto
