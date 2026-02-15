/**
 * @file position_estimator.hpp
 * @author Mateusz Wójcik (mateuszwojcikv@gmail.com)
 * @brief Position Estimator for computing 3D positions from 2D detections and depth data.
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

#include "yolo_ros/interfaces.hpp"

namespace yolo_ros {

    /**
     * @brief Position Estimator implementation
     * 
     */
    class PositionEstimator : public IPositionEstimator {
    public:
        /**
         * @brief Construct a new Position Estimator object
         * 
         * @param depth_samples Number of depth samples to take around detection center.
         */
        PositionEstimator(int depth_samples = 100);

        void set_class_radii(const std::map<int, float>& radii) override;
        void set_camera_parameters(float height_m, float pitch_deg) override;

        Detection3D compute_3d(
            const Result2D& detection,
            const cv::Mat& depth_image,
            const sensor_msgs::msg::CameraInfo& camera_info,
            const std_msgs::msg::Header& header
        ) override;

    private:
        std::map<int, float> _class_radii; // Class ID -> typical radius in meters
        const int _depth_samples;
        float _camera_height = 0.0f; // meters
        float _camera_pitch = 0.0f;  // degrees

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
