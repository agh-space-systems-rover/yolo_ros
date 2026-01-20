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

    struct Detection3D {
        Result2D result2d;
        geometry_msgs::msg::Point position;
        std_msgs::msg::Header header;
        // The frame_id in header is usually the camera frame
    };

    class PositionEstimator {
    public:
        PositionEstimator();

        void set_class_radii(const std::map<int, float>& radii);

        Detection3D compute_3d(
            const Result2D& detection,
            const cv::Mat& depth_image,
            const sensor_msgs::msg::CameraInfo& camera_info,
            const std_msgs::msg::Header& header
        );

    private:
        std::map<int, float> class_radii_; // Class ID -> typical radius in meters
        const int POSITION_ESTIMATION_DEPTH_SAMPLES = 100;

        float get_depth_from_region(const cv::Rect& bbox, const cv::Mat& depth);
        float estimate_depth_from_size(const cv::Rect& bbox, const sensor_msgs::msg::CameraInfo& info, int class_id);
    };

} // namespace yolo_ros
