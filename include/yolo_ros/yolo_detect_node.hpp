/**
 * @file yolo_detect_node.hpp
 * @author Mateusz Wójcik (mateuszwojcikv@gmail.com)
 * @brief Header for YOLO Detect Node for Kalman ROS2 Package
 * @version 0.1.0
 * @date 2026-01-20
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "yolo_ros/detector.hpp"
#include "yolo_ros/interfaces.hpp"
#include "yolo_ros/camera_manager.hpp"
#include "yolo_ros/visualizer.hpp"

namespace yolo_ros {

    /**
     * @brief YOLO Detect Node for object detection, 3D position estimation, and tracking.
     * 
     */
    class YoloDetectNode : public rclcpp_lifecycle::LifecycleNode {
    public:
        explicit YoloDetectNode(const rclcpp::NodeOptions & options);

        using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

        CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
        CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
        CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
        CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override;
        CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) override;

    private:
        // Parameters
        int num_cameras_;
        bool subscribe_depth_;
        std::string color_transport_;
        std::string depth_transport_;
        std::string model_path_;
        std::string class_names_path_; 
        std::vector<std::string> class_names_;
        std::vector<double> class_radii_param_;
        std::string world_frame_;
        double merge_radius_;
        int temporal_window_;
        int temporal_threshold_;
        float confidence_threshold_;
        bool publish_tf_;
        bool publish_annotated_;
        std::string annotated_transport_;
        bool debug_mode_;
        std::vector<std::string> rgbd_ids_;

        // Components
        std::unique_ptr<IDetector> detector_;
        std::unique_ptr<IPositionEstimator> estimator_;
        std::unique_ptr<ITracker> tracker_;
        std::unique_ptr<CameraManager> camera_man_;
        std::unique_ptr<Visualizer> visualizer_;

        // ROS Interfaces
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

        rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;
        
        rclcpp::TimerBase::SharedPtr timer_;

        /**
         * @brief A callback for timer events to process and publish detections
         * 
         */
        void timer_callback();

        /**
         * @brief Process detections from the detector and estimate their 3D positions
         * 
         * @param camera_data Vector of camera data
         * @param results_batch Vector of detection results for each image
         * @return std::vector<Detection3D> 
         */
        std::vector<Detection3D> process_detections(const std::vector<CameraData>& camera_data, const std::vector<std::vector<Result2D>>& results_batch);
        
        /**
         * @brief Publish detections to ROS topic
         * 
         * @param detections Vector of 3D detections to publish
         */
        void publish_detections(const std::vector<Detection3D>& detections);
    };
}
