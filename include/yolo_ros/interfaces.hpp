#pragma once

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <tf2_ros/buffer.h>

#include "yolo_ros/detector.hpp" // For Result2D

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
     * @brief Interface for Position Estimator components
     * 
     */
    class IPositionEstimator {
    public:
        virtual ~IPositionEstimator() = default;

        /**
         * @brief Set the map defining real-world radii for each class ID.
         */
        virtual void set_class_radii(const std::map<int, float>& radii) = 0;
        
        /**
         * @brief Set camera mounting parameters for ground plane projection (monocular depth estimation).
         */
        virtual void set_camera_parameters(float height_m, float pitch_deg) = 0;

        /**
         * @brief Compute the 3D position of a detection.
         */
        virtual Detection3D compute_3d(
            const Result2D& detection,
            const cv::Mat& depth_image,
            const sensor_msgs::msg::CameraInfo& camera_info,
            const std_msgs::msg::Header& header
        ) = 0;
    };

    /**
     * @brief Interface for Tracker components
     * 
     */
    class ITracker {
    public:
        using Ptr = std::unique_ptr<ITracker>;
        virtual ~ITracker() = default;

        /**
         * @brief Process new detections: transform, merge, and track.
         */
        virtual std::vector<Detection3D> process(
            const std::vector<Detection3D>& new_detections, 
            const std::shared_ptr<tf2_ros::Buffer>& tf_buffer,
            const std::string& target_frame,
            const rclcpp::Time& current_time
        ) = 0;
    };

} // namespace yolo_ros
