// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

// STD
#include <cmath>
#include <string>
#include <vector>

#include "number_classifier.hpp"
#include "globalParam.hpp"

#ifdef APRILTAG
#include "ApriltagDetector.hpp"
#endif
class Detector{    
    public:
        struct LightParams
        {
            // width / height
            double min_ratio;
            double max_ratio;
            // vertical angle
            double max_angle;
        };

        struct ArmorParams
        {
            double min_light_ratio;
            // light pairs distance
            double min_small_center_distance;
            double max_small_center_distance;
            double min_large_center_distance;
            double max_large_center_distance;
            // horizontal angle
            double max_angle;
        };

        Detector(GlobalParam &gp);

        std::vector<UnsolvedArmor> detect(cv::Mat &input, const int color,bool &have_armor);

        cv::Mat preprocessImage(const cv::Mat &input);
        std::vector<Light> findLights(const cv::Mat &rbg_img, const cv::Mat &binary_img);
        std::vector<UnsolvedArmor> matchLights(const std::vector<Light> &lights);
        bool refine_corner(Light &tar, cv::Mat &src);
        // cv::Point2f find_symmetry_axis(cv::Mat &src);
        cv::Point2d find_symmetry_axis(cv::Mat &src, cv::Mat &mask);
        // For debug usage
        cv::Mat getAllNumbersImage();
        void drawResults(cv::Mat &img);

        // parameters
        int blue_threshold, red_threshold;
        int binary_thres;
        int detect_color;
        double color_diff_threshold = 20;
        LightParams l;
        ArmorParams a;
        std::unique_ptr<NumberClassifier> classifier;
#ifdef APRILTAG
        std::vector<Armor> tag_list;
#endif

        // Debug msgs
        cv::Mat binary_img;

    private:
        GlobalParam *gp;
        bool isLight(const Light &possible_light);
        bool containLight(const int i, const int j, const std::vector<Light> &lights) noexcept;
        ArmorType isArmor(const Light &light_1, const Light &light_2);

        std::vector<Light> lights_;
        std::vector<UnsolvedArmor> armors_;
};

#endif // ARMOR_DETECTOR__DETECTOR_HPP_
