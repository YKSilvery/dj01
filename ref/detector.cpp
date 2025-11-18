// Copyright (c) 2022 ChenJun
// Licensed under the MIT License.

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <filesystem>

#include "detector.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/highgui.hpp"

#ifdef APRILTAG
void Detector::find_apriltag(cv::Mat &src, std::vector<UnsolvedArmor> &armors){
    //=====================AprilTag识别======================//
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
     cv::convertScaleAbs(gray, gray, 1, 200);
    // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    apriltagDetector -> detect(gray, tags, ids);
    // std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    // std::cout << "AprilTag time: " << time_span.count() << " seconds." << std::endl;
    // cv::Vec3d rvec, tvec;
    // apriltagDetector -> draw(src, tags, ids);
    // std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    // std::chrono::duration<double> time_span2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
    // std::cout << "AprilTag draw time: " << time_span2.count() << " seconds." << std::endl;
    cv::Mat rVec, tVec;
    tag_list.clear();
    for (int i = 0; i < tags.size(); i++)
    {
        if (ids[i] == 0){
            apriltagDetector -> solvePnP(tags[i], rVec, tVec);
            cv::Rodrigues(rVec, rVec);
            Armor tar;
            tar.center = cv::Point3f(tVec.at<double>(0), tVec.at<double>(1), tVec.at<double>(2));
            tar.angle = cv::Point3f(rVec.at<double>(0), rVec.at<double>(1), rVec.at<double>(2));
            tag_list.emplace_back(tar);
        }
    }
    for (auto &armor : armors){
        for (int i = 0; i < tags.size(); i++){
            if (ids[i] != 0) continue;
            std::vector<cv::Point2f> quad = {armor.left_light.bottom, armor.left_light.top, armor.right_light.top, armor.right_light.bottom};
            cv::Point2d point = (tags[i][0] + tags[i][1] + tags[i][2] + tags[i][3]) / 4.0;
            if(cv::pointPolygonTest(quad, point, false) > 0){
                armor.isApriltag = true;
                break;
            }
        }
    }
}
#endif

Detector::Detector(GlobalParam &gp)
{
    // int binary_thres = binary_threshold;
    // int detect_color = color;
    this->gp = &gp;
    int color = gp.color;
    this->detect_color = color;
    double min_ratio,
        max_ratio,
        max_angle_l,
        min_light_ratio,
        min_small_center_distance,
        max_small_center_distance,
        min_large_center_distance,
        max_large_center_distance,
        max_angle_a,
        num_threshold;
    min_ratio = gp.min_ratio;
    max_ratio = gp.max_ratio;
    max_angle_l = gp.max_angle_l;
    min_light_ratio = gp.min_light_ratio;
    min_small_center_distance = gp.min_small_center_distance;
    max_small_center_distance = gp.max_small_center_distance;
    min_large_center_distance = gp.min_large_center_distance;
    max_large_center_distance = gp.max_large_center_distance;
    max_angle_a = gp.max_angle_a;
    num_threshold = gp.num_threshold;
    this->red_threshold = gp.red_threshold;
    this->blue_threshold = gp.blue_threshold;
    binary_thres = color == RED ? this->red_threshold : this->blue_threshold;
    this->l = {
        .min_ratio = min_ratio,
        .max_ratio = max_ratio,
        .max_angle = max_angle_l};

    this->a = {
        .min_light_ratio = 0.7,
        .min_small_center_distance = min_small_center_distance,
        .max_small_center_distance = max_small_center_distance,
        .min_large_center_distance = min_large_center_distance,
        .max_large_center_distance = max_large_center_distance,
        .max_angle = max_angle_a};

    // Init classifier
    auto model_path = "../model/lenet.onnx";
    auto label_path = "../model/label.txt";
    std::vector<std::string> ignore_classes =
        std::vector<std::string>{"negative"};
    this->classifier =
        std::make_unique<NumberClassifier>(model_path, label_path, num_threshold, ignore_classes);
#ifdef APRILTAG
    apriltagDetector = new ApriltagDetector(75, gp.fx, gp.fy, gp.cx, gp.cy, gp.k1, gp.k2, gp.k3, gp.p1, gp.p2); // 初始化apriltag检测器
#endif
}

std::vector<UnsolvedArmor> Detector::detect(cv::Mat &input, const int color,bool &have_armor)
{
    this->binary_thres = color == RED ? this->red_threshold : this->blue_threshold;
    this->detect_color = color;
    binary_img = preprocessImage(input);
    using namespace cv;
#ifdef DEBUGCOLOR
    // cv::imshow("gray_img", gray_img);
    cv::imshow("binary", binary_img);
    cv::Mat tmp;
    cv::resize(binary_img, tmp, cv::Size(binary_img.size[1] * gp->resize, binary_img.size[0] * gp->resize));
    cv::imshow("binary", tmp);
#endif
#ifdef DILATED
    cv::Mat tmp1 = binary_img.clone();
    if (this->detect_color == 1)
    {
        auto kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        cv::dilate(tmp1, tmp1, kernel, cv::Point(-1, -1), 2);
    }
    else
    {
        auto kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        cv::dilate(tmp1, tmp1, kernel, cv::Point(-1, -1), 1);
    }
#ifdef DEBUGCOLOR
        cv::resize(tmp1, tmp1, cv::Size(tmp1.size[1] * gp->resize, tmp1.size[0] * gp->resize));
        cv::imshow("binary__", tmp1);
#endif
#endif
    lights_ = findLights(input, binary_img);
#ifdef DEBUGMODE
    // for (auto light : lights_)
    // {
    //     cv::rectangle(input, light.boundingRect2f(), cv::Scalar(255, 255, 255), 1);
    // }
#endif
    armors_ = matchLights(lights_);
    if (!armors_.empty())
    {
        classifier->extractNumbers(input, armors_, this->detect_color);
#ifdef APRILTAG
        find_apriltag(input, armors_);
#endif
        classifier->classify(armors_);
    }
    // for (auto &armor : armors_){
    //     refine_corner(armor.left_light, input);
    //     refine_corner(armor.right_light, input);
    // }
    if(armors_.size() > 0){
        have_armor = true;
    }
    else {
        have_armor = false;
    }
    return armors_;
}

cv::Mat Detector::preprocessImage(const cv::Mat &rgb_img)
{
    cv::Mat gray_img;
    cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);
    return binary_img;
}

std::vector<Light> Detector::findLights(const cv::Mat &rbg_img, const cv::Mat &binary_img)
{
    using std::vector;
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
#ifndef FYT
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
#else
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
#endif
    vector<Light> lights;
    for (const auto &contour : contours)
    {
        if (contour.size() < 6)
            continue;

        auto r_rect = cv::minAreaRect(contour);
        auto light = Light(r_rect);

        if (isLight(light))
        {
            auto rect = light.boundingRect();
            if ( // Avoid assertion failed
                0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols && 0 <= rect.y &&
                0 <= rect.height && rect.y + rect.height <= rbg_img.rows)
            {
                int sum_r = 0, sum_b = 0;
                auto roi = rbg_img(rect);
                auto roi_binary = binary_img(rect);
                std::vector<cv::Point2f> binary_points;
                // Iterate through the ROI
                for (int i = 0; i < roi.rows; i++)
                {
                    for (int j = 0; j < roi.cols; j++)
                    {
                        if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0)
                        {
                            sum_r += roi.at<cv::Vec3b>(i, j)[0];
                            sum_b += roi.at<cv::Vec3b>(i, j)[2];
                        }
                    }
                }
                // Sum of red pixels > sum of blue pixels ?
                if (std::abs(sum_r - sum_b) / static_cast<double>(contour.size()) > color_diff_threshold)
                {
                    light.color = sum_r > sum_b ? RED : BLUE;
                }
                lights.emplace_back(light);
            }
        }
    }
    std::sort(lights.begin(), lights.end(), 
    [](const Light &l1, const Light &l2) {
        return l1.center.x < l2.center.x;
    });

    return lights;
}

bool Detector::isLight(const Light &light)
{
    // The ratio of light (short side / long side)
    float ratio = light.width / light.length;
    bool ratio_ok = l.min_ratio < ratio && ratio < l.max_ratio;

    bool angle_ok = light.tilt_angle < l.max_angle;
    bool size_ok = light.length * light.width < 12800 and light.length > 10;
    bool is_light = ratio_ok && angle_ok && size_ok;

    return is_light;
}

std::vector<UnsolvedArmor> Detector::matchLights(const std::vector<Light> &lights)
{
    std::vector<UnsolvedArmor> armors;

    // Loop all the pairing of lights
    for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++)
    {   
        if( light_1->color != detect_color)
            continue;
        for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++)
        {
            if (light_2->color != detect_color)
                continue;
            if (containLight(light_1 - lights.begin(), light_2 - lights.begin(), lights))
            {   
                continue;
            }
            auto type = isArmor(*light_1, *light_2);
            if (type != ArmorType::INVALID)
            {
                auto armor = UnsolvedArmor(*light_1, *light_2);
                armor.type = type;
                armors.emplace_back(armor);
            }
        }
    }

    return armors;
}

// Check if there is another light in the boundingRect formed by the 2 lights
bool Detector::containLight(const int i, const int j, const std::vector<Light> &lights) noexcept {
  const Light &light_1 = lights.at(i), light_2 = lights.at(j);
  auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
  auto bounding_rect = cv::boundingRect(points);
  double avg_length = (light_1.length + light_2.length) / 2.0;
  double avg_width = (light_1.width + light_2.width) / 2.0;
  // Only check lights in between
  for (int k = i + 1; k < j; k++) {
    const Light &test_light = lights.at(k);

    // 防止数字干扰
    if (test_light.width > 2 * avg_width) {
      continue;
    }
    // 防止红点准星或弹丸干扰
    if (test_light.length < 0.5 * avg_length) {
      continue;
    }

    if (bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
        bounding_rect.contains(test_light.center)) {
      return true;
    }
  }
  return false;
}

ArmorType Detector::isArmor(const Light &light_1, const Light &light_2)
{
    // Ratio of the length of 2 lights (short side / long side)
    float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                               : light_2.length / light_1.length;
    bool light_ratio_ok = light_length_ratio > a.min_light_ratio;
    // //std::cout<<"light_length_ratio: "<<light_length_ratio<<std::endl;
    // Distance between the center of 2 lights (unit : light length)
    float avg_light_length = (light_1.length + light_2.length) / 2;
    float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
    bool center_distance_ok = (a.min_small_center_distance <= center_distance &&
                               center_distance < a.max_small_center_distance) ||
                              (a.min_large_center_distance <= center_distance &&
                               center_distance < a.max_large_center_distance);

    // Angle of light center connection
    // //std::cout<<"center_distance: "<<center_distance<<std::endl;
    cv::Point2f diff = light_1.center - light_2.center;
    float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
    bool angle_ok = angle < a.max_angle;

    // //std::cout<<"angle: "<<angle<<std::endl;
    bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

    // Judge armor type
    ArmorType type;
    if (is_armor)
    {
        type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
    }
    else
    {
        type = ArmorType::INVALID;
    }

    return type;
}

cv::Mat Detector::getAllNumbersImage()
{
    if (armors_.empty())
    {
        return cv::Mat(cv::Size(20, 28), CV_8UC1);
    }
    else
    {
        std::vector<cv::Mat> number_imgs;
        number_imgs.reserve(armors_.size());
        for (auto &armor : armors_)
        {
            number_imgs.emplace_back(armor.number_img);
        }
        cv::Mat all_num_img;
        cv::vconcat(number_imgs, all_num_img);
        return all_num_img;
    }
}

void Detector::drawResults(cv::Mat &img)
{
    // Draw Lights
    for (const auto &light : lights_)
    {
        //cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
        //cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
        auto line_color = light.color == RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
        cv::line(img, light.top, light.bottom, line_color, 1);
    }

    // Draw armors
    for (const auto &armor : armors_)
    {
        cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
        cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
    }

    // Show numbers and confidence
    for (const auto &armor : armors_)
    {
        cv::putText(
            img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2);
    }
}
// cv::Point2d Detector::find_symmetry_axis(cv::Mat &src, cv::Mat &mask){
//     cv::Mat roi = cv::Mat::zeros(src.size(), src.type());
//     src.copyTo(roi, mask);
//     roi.convertTo(roi, CV_32F);
//     cv::normalize(roi, roi, 0, 50, cv::NORM_MINMAX, -1, mask);
//     std::vector<cv::Point2d> points;
//     for (int i = 0; i < roi.rows; i++) {                
//         for (int j = 0; j < roi.cols; j++) {
//             for (int k = 0; k < std::round(roi.at<float>(i, j)); k++) {
//                 points.emplace_back(cv::Point2d(j, i));
//             }
//         }
//     }
//     double n = double(points.size());                   
//     double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;    // 最小二乘法拟合
    
//     for (int i = 0; i < n; ++i) {
//         sum_x += points[i].x;
//         sum_y += points[i].y;
//         sum_xx += points[i].x * points[i].x;
//         sum_xy += points[i].x * points[i].y;
//     }

//     double a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
//     double b = (sum_y - a * sum_x) / n;
//     std::cout << "a: " << a << " b: " << b << std::endl;
//     return cv::Point2d(a, b);
// }

cv::Point2d Detector::find_symmetry_axis(cv::Mat &src, cv::Mat &mask){
    cv::Mat roi = cv::Mat::zeros(src.size(), src.type());
    src.copyTo(roi, mask);
    cv::imshow("22", roi);
    roi.convertTo(roi, CV_32F);
    cv::normalize(roi, roi, 0, 25, cv::NORM_MINMAX, -1, mask);
    std::vector<cv::Point2f> points;
    for (int i = 0; i < roi.rows; i++) {
        for (int j = 0; j < roi.cols; j++) {
            for (int k = 0; k < std::round(roi.at<float>(i, j)); k++) {
                points.emplace_back(cv::Point2f(j, i));
            }
        }
    }
    cv::Mat points_mat = cv::Mat(points).reshape(1);
    // PCA (Principal Component Analysis)
    auto pca = cv::PCA(points_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);
    // Get the symmetry axis
    cv::Point2f axis =
        cv::Point2f(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
    // Normalize the axis
    axis = axis / cv::norm(axis);
    if (axis.y < 0) {
        axis = -axis;
    }
    return axis;
}

bool Detector::refine_corner(Light &tar, cv::Mat &src){
    const double scale = 10;
    if (tar.width < 3) return false;
    cv::Rect box = tar.boundingRect();    // 获得灯条目标区域
    box = cv::Rect(box.x - scale, box.y - scale, box.width + 2 * scale, box.height + 2 * scale);
    box.x = std::max(0, box.x);
    box.y = std::max(0, box.y);
    box.width = std::min(src.cols - box.x, box.width);
    box.height = std::min(src.rows - box.y, box.height);
    cv::Mat roi = src(box).clone();
    // cv::Mat roi = cv::Mat::zeros(raw.size(), raw.type());
#ifdef DEBUGREFINE
    cv::Mat roi_show = roi.clone();
    cv::resize(roi_show, roi_show, cv::Size(roi.cols * 2, roi.rows * 2));
    cv::imshow("raw", roi_show);
#endif
    
    cv::Mat mask;                         // 获得灯条目标区域的mask
    cv::cvtColor(roi, mask, cv::COLOR_BGR2GRAY);
    int threshold = (detect_color == RED ? gp->red_threshold : gp->blue_threshold);
    cv::threshold(mask, mask, threshold - 10, 255, cv::THRESH_BINARY);

    cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);      // 转换到亮度图
    cv::Mat channel[3];
    cv::split(roi, channel);
    roi = channel[2];
    cv::GaussianBlur(roi, roi, cv::Size(3,3), 1, 1);
    cv::Mat raw = roi.clone();

    cv::Mat mask_roi = cv::Mat::zeros(roi.size(), roi.type());
    roi.copyTo(mask_roi, mask);
    cv::Moments moments = cv::moments(mask_roi, false);    // 求质心
    cv::Point2d centroid = cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);

    cv::Point2d axis = find_symmetry_axis(roi, mask); 

    cv::Canny(roi, roi, gp->grad_min, gp->grad_max);         // 边缘检测
#ifdef DEBUGREFINE
    cv::imshow("canny", roi);
#endif
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));  
    cv::dilate(roi, roi, kernel);             // 膨胀
    cv::erode(roi, roi, kernel);              // 腐蚀
#ifdef DEBUGREFINE
    cv::imshow("dilated", roi);
#endif
    std::vector<std::vector<cv::Point>> contours;           // 找到灯条所属轮廓
    cv::findContours(roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.size() == 0) return false;
    int index = 0;
    for (int i = 0; i < contours.size(); i++){
        auto &contour = contours[i];
        if (contour.size() > contours[index].size()) index = i;
    }
    std::vector<cv::Point> &light = contours[index]; 

   // 计算对称轴

    double k = axis.y / axis.x, b = centroid.y - k * centroid.x;

    cv::line(roi, cv::Point(0, b), cv::Point(roi.cols, roi.cols * k + b), cv::Scalar(255, 0, 255), 2);
    cv::imshow("11", roi);
    double min_top_dis = 1e9, min_bottom_dis = 1e9;
    
    // for (const cv::Point& point : light) {      // 对称轴与轮廓的交点作为角点
    //     // 直线方程 y = a * x + b
    //     double y_line = k * point.x + b;
    //     double distance = std::abs(point.y - y_line);
    //     // 判断该点是否在直线上（设置一个阈值）
    //     if (point.y < centroid.y && distance < min_top_dis) {
    //         tar.top = cv::Point2f(point) + cv::Point2f(box.x, box.y);
    //         min_top_dis = distance;
    //     }
    //     if (point.y > centroid.y && distance < min_bottom_dis) {
    //         tar.bottom = cv::Point2f(point) + cv::Point2f(box.x, box.y);
    //         min_bottom_dis = distance;
    //     }
    // }
    constexpr float START = 0.45;
    constexpr float END = 0.55;

    auto inImage = [&src](const cv::Point &point) -> bool {
        return point.x >= 0 && point.x < src.cols && point.y >= 0 && point.y < src.rows;
    };
    auto distance = [](float x0, float y0, float x1, float y1) -> float {
        return std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
    };

    float L = tar.length;
    // Select multiple corner candidates and take the average as the final corner
    int n = std::max(tar.width * 0.8 - 2, 0.0);
    int half_n = std::round(n / 2.0);
    // int half_n = 0;

    for (int k = 1; k >= -1; k-=2){    
         std::vector<cv::Point2d> candidates;
        for (int i = -half_n; i <= half_n; i++) {
            float dx = axis.x * k;
            float dy = axis.y * k;
            float x0 = centroid.x + k * L * START * axis.x + i;
            float y0 = centroid.y + k * L * START * axis.y;

            cv::Point2f prev = cv::Point2f(x0, y0);
            cv::Point2f corner = cv::Point2f(x0, y0);
            float max_brightness_diff = 0;
            bool has_corner = false;
            // Search along the symmetry axis to find the corner that has the maximum brightness difference
            for (float x = x0 + dx, y = y0 + dy; distance(x, y, x0, y0) < L * (END - START); x += dx, y += dy) {
                cv::Point2f cur = cv::Point2f(x, y);
                if (!inImage(cv::Point(cur))) break;
                uchar brightness = roi.at<uchar>(cur);
                // float brightness_diff = roi.at<uchar>(prev) - roi.at<uchar>(cur);
                // if (brightness_diff > max_brightness_diff && roi.at<uchar>(prev) > mean_val) {
                //     max_brightness_diff = brightness_diff;
                //     corner = prev;
                //     has_corner = true;
                // }
                if (brightness) candidates.push_back(cur);
                // prev = cur;
            }
        }
        if (!candidates.empty()) {
            cv::Point2d sum = cv::Point2d(0, 0);
            for (const cv::Point2d& point : candidates) {
                sum += point;
            }
            if(k==1)tar.bottom = sum / double(candidates.size()) + cv::Point2d(box.x, box.y);
            else tar.top = sum / double(candidates.size()) + cv::Point2d(box.x, box.y);
        }
    }
    

    std::vector<cv::Point2d> corners = {tar.top, tar.bottom};
    // cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 50, 0.01);
    // // 使用 cornerSubPix 进行亚像素精度化
    // cv::cornerSubPix(raw, corners, cv::Size(5, 5), cv::Size(-1, -1), criteria);

    tar.top = corners[0];
    tar.bottom = corners[1];
        
    // double length = cv::norm(corners[0] - corners[1]);
    // cv::Point2d axis = (1, line.x);
    // axis = axis / cv::norm(axis);
    // tar.top = centroid - axis * length / 2 + cv::Point2d(box.x, box.y);
    // tar.bottom = centroid + axis * length / 2 + cv::Point2d(box.x, box.y);
    
#ifdef DEBUGMODE
    cv::circle(src, tar.top, 1, cv::Scalar(255,255,255),-1);
    cv::circle(src, tar.bottom, 1, cv::Scalar(255,255,255),-1);
    cv::line(src, tar.top, tar.bottom, cv::Scalar(255,255,255), 1);
#endif
#ifdef DEBUGREFINE
    // const double s = 0.8;
    // cv::resize(src, roi, cv::Size(src.cols * s, src.rows * s));
    // cv::imshow("find light", src);
#endif
    return true;
}
// cv::Point2f Detector::find_symmetry_axis(cv::Mat &src){
//     cv::Mat roi = src.clone();
//     roi.convertTo(roi, CV_32F);
//     cv::normalize(roi, roi, 0, 30, cv::NORM_MINMAX);
//     std::vector<cv::Point2f> points;
//     for (int i = 0; i < roi.rows; i++) {
//         for (int j = 0; j < roi.cols; j++) {
//             for (int k = 0; k < std::round(roi.at<float>(i, j)); k++) {
//                 points.emplace_back(cv::Point2f(j, i));
//             }
//         }
//     }
//     cv::Mat points_mat = cv::Mat(points).reshape(1);
//     // PCA (Principal Component Analysis)
//     auto pca = cv::PCA(points_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);
//     // Get the symmetry axis
//     cv::Point2f axis =
//         cv::Point2f(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
//     // Normalize the axis
//     axis = axis / cv::norm(axis);
//     if (axis.y < 0) {
//         axis = -axis;
//     }
//     return axis;
// }

// bool Detector::refine_corner(Light &tar, cv::Mat &src){
//     const float scale = 0.3;
//     // if (tar.width < 5) return false;
//     cv::Rect box = tar.boundingRect();    // 获得灯条目标区域
//     // box = cv::Rect(box.x - box.width * scale, box.y - box.height * scale, box.width * (1 + 2 * scale), box.height * (1 + 2 * scale));
//     box = cv::Rect(box.x - 10, box.y - 10, box.width + 20, box.height + 20);
//     box.x = std::max(0, box.x);
//     box.y = std::max(0, box.y);
//     box.width = std::min(src.cols - box.x, box.width);
//     box.height = std::min(src.rows - box.y, box.height);
//     cv::Mat roi = src(box).clone();
// #ifdef DEBUGREFINE
//     cv::imshow("raw", roi);
// #endif
    
//     cv::Mat gray;
//     cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
//     cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);      // 转换到亮度图
//     cv::Mat channel[3];
//     cv::split(roi, channel);
//     roi = channel[2];
//     cv::GaussianBlur(roi, roi, cv::Size(3,3), 1, 1);
//     // ========================================================================
//     int threshold = (detect_color == RED ? gp->red_threshold : gp->blue_threshold) + 10;  // 求均值与质心
//     cv::threshold(gray, gray, threshold, 255, cv::THRESH_BINARY);
//     roi.copyTo(roi, gray);
//     cv::imshow("11",roi);
    
//     double mean_val = cv::mean(roi, gray)[0];
//     cv::Moments moments = cv::moments(roi, false);
//     cv::Point2f centroid = cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00);

//     cv::circle(src, centroid + cv::Point2f(box.x,box.y), 3, cv::Scalar(255, 255, 255), -1);
//     cv::Point2f axis = find_symmetry_axis(roi);    // 计算对称轴

//     constexpr float START = 0.45;
//     constexpr float END = 0.55;

//     auto inImage = [&src](const cv::Point &point) -> bool {
//         return point.x >= 0 && point.x < src.cols && point.y >= 0 && point.y < src.rows;
//     };
//     auto distance = [](float x0, float y0, float x1, float y1) -> float {
//         return std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
//     };

//     float L = tar.length;
//     // Select multiple corner candidates and take the average as the final corner
//     int n = tar.width * 0.75 - 2;
//     int half_n = std::round(n / 2);
//     for (int k = 1; k >= -1; k-=2){    
//         std::vector<cv::Point2f> candidates;
//         float dx = axis.x * k;
//         float dy = axis.y * k;
//         float x0 = centroid.x + k * L * START * axis.x;
//         float y0 = centroid.y + k * L * START * axis.y;

//         cv::Point2f prev = cv::Point2f(x0, y0);
//         cv::Point2f corner = cv::Point2f(x0, y0);
//         float max_brightness_diff = 0;
//         bool has_corner = false;
//         // Search along the symmetry axis to find the corner that has the maximum brightness difference
//         for (float x = x0 + dx, y = y0 + dy; distance(x, y, x0, y0) < L * (END - START); x += dx, y += dy) {
//             cv::Point2f cur = cv::Point2f(x, y);
//             if (!inImage(cv::Point(cur))) break;
//             float brightness_diff = roi.at<uchar>(prev) - roi.at<uchar>(cur);
//             // if (brightness_diff > max_brightness_diff && roi.at<uchar>(prev) > mean_val) {
//             //     max_brightness_diff = brightness_diff;
//             //     corner = prev;
//             //     has_corner = true;
//             // }
//             if (abs(roi.at<uchar>(cur) - mean_val) < 1e-2) corner = cur;
//             prev = cur;
//         }
//         if (has_corner) {
//             if(k==1) tar.bottom = corner + cv::Point2f(box.x, box.y);
//             else tar.top = corner + cv::Point2f(box.x, box.y);
//         }
//     }
//     tar.length = distance(tar.top.x, tar.top.y, tar.bottom.x, tar.bottom.y);
    // tar.top = centroid - axis * tar.length * 0.5 + cv::Point2f(box.x, box.y);
    // tar.bottom = centroid + axis * tar.length * 0.5 + cv::Point2f(box.x, box.y);

    // return true;
// ================================================================================
//     cv::Canny(roi, roi, gp->grad_min, gp->grad_max);
// #ifdef DEBUGREFINE
//     cv::imshow("canny", roi);
// #endif
//     cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
//     cv::dilate(roi, roi, kernel);
//     cv::erode(roi, roi, kernel);
// #ifdef DEBUGREFINE
//     cv::imshow("dilated", roi);
// #endif
    
//     std::vector<std::vector<cv::Point>> contours;
//     cv::findContours(roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(box.x, box.y));
//     if (contours.size() == 0) return false;

//     //============================strategy1================================
//     int index = 0;
//     for (int i = 0; i < contours.size(); i++){
//         auto &contour = contours[i];
//         if (contour.size() > contours[index].size()) index = i;
//     }
//     std::vector<cv::Point> &light = contours[index]; 

//     //============================strategy2================================
//     // int index1 = 0, index2 = 0;
//     // for (int i = 0; i < contours.size(); i++){
//     //     auto &contour = contours[i];
//     //     if (contour.size() < 25) continue;
//     //     if (contour.size() > contours[index1].size()) index2 = index1, index1 = i;
//     //     else if (contour.size() > contours[index2].size()) index2 = i;
//     // }
//     // std::vector<cv::Point> light;
//     // for(auto point : contours[index1]) light.push_back(point);
//     // for(auto point : contours[index2]) light.push_back(point);
    
//     //=====================================================================
// #ifdef DEBUGMODE
//     cv::drawContours(src, std::vector<std::vector<cv::Point>>{light}, 0, cv::Scalar(0,255,255));
// #endif
//     auto bbox = cv::minAreaRect(light);
//     cv::Point2f p[4];
//     bbox.points(p);
//     cv::drawContours(src, std::vector<std::vector<cv::Point>>{std::vector<cv::Point>{p[0], p[1], p[2], p[3]}}, 0, cv::Scalar(0,255,0));
//     std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
//     if (light.size() == 0) return false;

//     tar.top = (p[0] + p[1]) / 2;
//     tar.bottom = (p[2] + p[3]) / 2;   

//     std::vector<cv::Point2f> points = {tar.top, tar.bottom};
//     cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 100, 0.01);
//     printf("before: (%f , %f)  |  (%f , %f)\n", tar.top.x, tar.top.y, tar.bottom.x, tar.bottom.y);
//     cv::cornerSubPix(roi, points, cv::Size(5,5), cv::Size(-1, -1), criteria);
//     tar.top = points[0],  tar.bottom = points[1];
//     printf("after : (%f , %f)  |  (%f , %f)\n", tar.top.x, tar.top.y, tar.bottom.x, tar.bottom.y);
//     // tar.top = tar.bottom = cv::Point2f(light[0].x, light[0].y);
//     // for (auto Point : light){
//     //     cv::Point2f point(Point.x, Point.y);
//     //     if (pointToLineDistance(p[2] + cv::Point2f(0,15), p[3] + cv::Point2f(0,15), point) < pointToLineDistance(p[2] + cv::Point2f(0,15), p[3] + cv::Point2f(0,15), tar.bottom)) tar.bottom = point;
//     //     if (pointToLineDistance(p[0] - cv::Point2f(0,15), p[1] - cv::Point2f(0,15), point) < pointToLineDistance(p[0] - cv::Point2f(0,15), p[1] - cv::Point2f(0,15), tar.top)) tar.top = point;
//     // }

//     // std::vector<cv::Point2f> corners{top, bottom};
//     // cv::Size winSize = cv::Size(3, 3); // 搜索窗口大小
//     // cv::Size zeroZone = cv::Size(0, 0); // 中心 1x1 区域不进行计算
//     // cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 0.001);
//     // cornerSubPix(channel[2], corners, winSize, zeroZone, criteria);
//     // tar.top = corners[0];
//     // tar.bottom = corners[1];

// #ifdef DEBUGMODE
//     cv::circle(src, tar.top, 1, cv::Scalar(255,255,255),-1);
//     cv::circle(src, tar.bottom, 1, cv::Scalar(255,255,255),-1);
//     cv::line(src, tar.top, tar.bottom, cv::Scalar(255,255,255));
// #endif
//     // cv::resize( OutputArray dst, Size dsize)
//     cv::imshow("find light", src);
//     return true;
// }