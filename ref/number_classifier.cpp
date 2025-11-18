// Copyright 2022 Chen Jun
// Licensed under the MIT License.

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// STL
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <map>
#include <string.h>
#include <string>
#include <vector>

#include "globalParam.hpp"
#include "number_classifier.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/matx.hpp"

NumberClassifier::NumberClassifier(
    const std::string &model_path, const std::string &label_path, const double thre,
    const std::vector<std::string> &ignore_classes)
    : threshold(thre), ignore_classes_(ignore_classes)
{
    // auto my_model_path = model_path;
    this->net_ = cv::dnn::readNetFromONNX(model_path);
    std::ifstream label_file(label_path);
    std::string line;
    while (std::getline(label_file, line))
    {
        class_names_.push_back(line);
    }
}

void NumberClassifier::extractNumbers(const cv::Mat &src, std::vector<UnsolvedArmor> &armors, int detect_color)
{
    // Light length in image
    const int light_length = 12;  // 修改为与标准版本一致
    // Image size after warp
    const int warp_height = 28;
    const int small_armor_width = 32;
    const int large_armor_width = 54;
    // Number ROI size
    const cv::Size roi_size(20, 28);
    const cv::Size input_size(28, 28);

    for (auto &armor : armors)
    {
        // Warp perspective transform
        cv::Point2f lights_vertices[4] = {
            armor.left_light.bottom, armor.left_light.top, armor.right_light.top,
            armor.right_light.bottom};

        const int top_light_y = (warp_height - light_length) / 2 - 1;
        const int bottom_light_y = top_light_y + light_length;
        const int warp_width = armor.type == ArmorType::SMALL ? small_armor_width : large_armor_width;
        cv::Point2f target_vertices[4] = {
            cv::Point(0, bottom_light_y),
            cv::Point(0, top_light_y),
            cv::Point(warp_width - 1, top_light_y),
            cv::Point(warp_width - 1, bottom_light_y),
        };
        cv::Mat number_image;
        auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
        cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

        // Get ROI
        number_image = number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

#ifdef DEBUGNUM
        cv::Mat debug_image = number_image.clone();
        cv::resize(debug_image, debug_image, cv::Size(400, 400));
        cv::imshow("Original ROI", debug_image);
#endif
        // Split channels and process
        cv::Mat images[3];
        cv::split(number_image, images);
#ifdef DEBUGNUM
        int d_max = 0;
#endif
        for (int i = 0; i < images[2 - 2 * detect_color].cols; i++)
        {
            for (int j = 0; j < images[2 - 2 * detect_color].rows; j++)
            {
#ifdef DEBUGNUM
                if (images[2 - 2 * detect_color].at<uchar>(j, i) > d_max)
                    d_max = images[2 - 2 * detect_color].at<uchar>(j, i);
#endif
                if (images[2 - 2 * detect_color].at<uchar>(j, i) > 100)
                    images[2 - 2 * detect_color].at<uchar>(j, i) = 0;
            }
        }
#ifdef DEBUGNUM
        std::cout << "number max pixel: " << d_max << std::endl;
#endif
        cv::threshold(images[2 - 2 * detect_color], number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::resize(number_image, number_image, input_size);

#ifdef DEBUGNUM
        cv::Mat debug_binary = number_image.clone();
        cv::resize(debug_binary, debug_binary, cv::Size(400, 400));
        cv::imshow("Binary", debug_binary);
#endif

        armor.number_img = number_image;
    }
}

void NumberClassifier::classify(std::vector<UnsolvedArmor> &armors)
{
#ifdef DEBUGNUM
    int i = 0;
#endif

    for (auto &armor : armors)
    {
        // Normalize
        cv::Mat input = armor.number_img / 255.0;

        // Create blob from image
        cv::Mat blob;
        cv::dnn::blobFromImage(input, blob);

        // Set the input blob for the neural network
        net_.setInput(blob);

        // Forward pass the image blob through the model
        cv::Mat outputs = net_.forward().clone();

        // Decode the output
        double confidence;
        cv::Point class_id_point;
        minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
        int label_id = class_id_point.x;

        armor.confidence = confidence;
        armor.number = class_names_[label_id];

#ifdef DEBUGNUM
        cv::Mat debug_num = armor.number_img.clone();
        cv::resize(debug_num, debug_num, cv::Size(400, 400));
        cv::putText(debug_num, 
                    std::to_string(label_id) + ":" + std::to_string(confidence), 
                    cv::Point(50, 50), 
                    1, 2, cv::Scalar(255, 255, 255), 2);
        cv::imshow("num" + std::to_string(i++), debug_num);
#endif

        std::stringstream result_ss;
        result_ss << armor.number << ": " << std::fixed << std::setprecision(1)
                 << armor.confidence * 100.0 << "%";
        armor.classfication_result = result_ss.str();
    }

    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(),
            [this](const UnsolvedArmor &armor)
            {
                if (armor.isApriltag) return false;
                if (armor.confidence < threshold)
                {
                    return true;
                }

                for (const auto &ignore_class : ignore_classes_)
                {
                    if (armor.number == ignore_class)
                    {
                        return true;
                    }
                }

                bool mismatch_armor_type = false;
                if (armor.type == ArmorType::LARGE) {
                    mismatch_armor_type = armor.number == "outpost" || armor.number == "2" ||
                                        armor.number == "guard";
                } else if (armor.type == ArmorType::SMALL) {
                    mismatch_armor_type = armor.number == "1" || armor.number == "base";
                }
                return mismatch_armor_type;
            }),
        armors.end());

}