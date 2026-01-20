/**
 * @file position_estimator.hpp
 * @author Mateusz Wójcik (mateuszwojcikv@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2026-01-20
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include <vector>
#include <map>
#include <string>
#include <optional>

#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d.hpp>

#include "yolo_ros/detector.hpp"

namespace yolo_ros {

    /**
     * @brief Represents a 3D detection with associated 2D result.
     * 
     */
    struct Detection3D {
        Result2D result2d;
        geometry_msgs::msg::Point position;
        std_msgs::msg::Header header;
    };

    /**
     * @brief Position Estimator for computing 3D positions from 2D detections and depth data.
     * 
     */
    class PositionEstimator {
    public:
        /**
         * @brief Construct a new Position Estimator object
         * 
         * @param depth_samples Number of depth samples to take around detection center.
         */
        PositionEstimator(int depth_samples = 100);

        /**
         * @brief Set the map defining real-world radii for each class ID.
         * Used for fallback depth estimation when depth data is missing.
         * 
         * @param radii Map where Key = Class ID, Value = Radius in meters.
         */
        void set_class_radii(const std::map<int, float>& radii);


        /**
         * @brief Compute the 3D position of a detection.
         * 
         * @param detection The 2D detection result.
         * @param depth_image The depth image corresponding to the detection.
         * @param camera_info Camera intrinsic parameters.
         * @param header Message header containing timestamp and frame information.
         * @return Detection3D The computed 3D detection.
         */
        Detection3D compute_3d(
            const Result2D& detection,
            const cv::Mat& depth_image,
            const sensor_msgs::msg::CameraInfo& camera_info,
            const std_msgs::msg::Header& header
        );

    private:
        std::map<int, float> _class_radii; // Class ID -> typical radius in meters
        const int _depth_samples;

        /**
         * @brief Get the depth from region object
         * 
         * @param bbox Bounding box of the detected object.
         * @param depth Depth image.
         * @return float Estimated depth in meters.
         */
        float get_depth_from_region(const cv::Rect& bbox, const cv::Mat& depth);

        /**
         * @brief Estimate depth based on object size in the image and known real-world size.
         * 
         * @param bbox Bounding box of the detected object.
         * @param info Camera intrinsic parameters.
         * @param class_id Class ID of the detected object.
         * @return float Estimated depth in meters.
         */
        float estimate_depth_from_size(const cv::Rect& bbox, const sensor_msgs::msg::CameraInfo& info, int class_id);
    };

} // namespace yolo_ros
