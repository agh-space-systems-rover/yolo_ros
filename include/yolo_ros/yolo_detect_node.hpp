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
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/simple_filter.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <cv_bridge/cv_bridge.h>

#include "yolo_ros/detector.hpp"
#include "yolo_ros/position_estimator.hpp"
#include "yolo_ros/detection_tracker.hpp"

namespace yolo_ros {

    /**
     * @brief Helper class to wrap CompressedImage subscription into SimpleFilter interface
     * 
     */
    class CompressedSubscriberWrapper : public message_filters::SimpleFilter<sensor_msgs::msg::Image>
    {
    public:
        /**
         * @brief Construct a new Compressed Subscriber Wrapper object
         * 
         * @param node Pointer to the lifecycle node
         * @param topic Topic to subscribe to
         * @param qos Quality of Service profile
         */
        CompressedSubscriberWrapper(rclcpp_lifecycle::LifecycleNode* node, const std::string& topic, const rmw_qos_profile_t& qos)
        {
            sub_ = node->create_subscription<sensor_msgs::msg::CompressedImage>(
                topic, 
                rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos), qos),
                [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                    // Decode compressed image to raw cv::Mat
                    cv_bridge::CvImagePtr cv_ptr;
                    try {
                        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
                    } catch (cv_bridge::Exception& e) {
                        return;
                    }
                    // Signal the filter chain with the raw image message
                    this->signalMessage(cv_ptr->toImageMsg());
                }
            );
        }
    private:
        rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
    };

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
        std::string class_names_path_; // Path to file with class names if needed, or param
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

        // Objects
        std::unique_ptr<IDetector> detector_;
        std::unique_ptr<PositionEstimator> estimator_;
        std::unique_ptr<DetectionTracker> tracker_;

        // ROS Interfaces
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

        rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;
        
        // --- Synchronization for 1 camera ---
        using SyncPolicy1 = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo
        >;
        using Sync1 = message_filters::Synchronizer<SyncPolicy1>;
        

        struct CameraContext {
            int index;
            
            // We use SimpleFilter pointers to handle either Raw (Subscriber) or Compressed (Wrapper)
            std::shared_ptr<message_filters::SimpleFilter<sensor_msgs::msg::Image>> color_sub;
            std::shared_ptr<message_filters::SimpleFilter<sensor_msgs::msg::Image>> depth_sub;
            
            // Info is always raw
            std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo, rclcpp_lifecycle::LifecycleNode>> info_sub;
            
            std::shared_ptr<Sync1> sync;
            
            rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_pub; 

            // Buffers for timer-based loop
            sensor_msgs::msg::Image::ConstSharedPtr last_color;
            sensor_msgs::msg::Image::ConstSharedPtr last_depth;
            sensor_msgs::msg::CameraInfo::ConstSharedPtr last_info;
            bool has_new_data = false;
            
            // Keep ownership of the actual subscribers if they are not the filter interface
            std::shared_ptr<void> color_sub_handle;
            std::shared_ptr<void> depth_sub_handle;
        };
        
        std::vector<std::shared_ptr<CameraContext>> cameras_;
        rclcpp::TimerBase::SharedPtr timer_;

        /**
         * @brief A callback for when camera data is received
         * 
         * @param color Color image from the camera
         * @param depth Depth image from the camera
         * @param info Camera info message
         * @param camera_index Index of the camera
         */
        void on_camera_data(
            const sensor_msgs::msg::Image::ConstSharedPtr& color,
            const sensor_msgs::msg::Image::ConstSharedPtr& depth,
            const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info,
            int camera_index
        );

        /**
         * @brief A callback for timer events to process and publish detections
         * 
         */
        void timer_callback();

        /**
         * @brief Gather images from all cameras that have new data
         * 
         * @param images Vector to fill with gathered images
         * @param indices Vector to fill with corresponding camera indices
         * @return true if at least one image was gathered
         * @return false otherwise
         */
        bool gather_images(std::vector<cv::Mat>& images, std::vector<int>& indices);
        
        /**
         * @brief Process detections from the detector and estimate their 3D positions
         * 
         * @param images Vector of input images
         * @param indices Vector of corresponding camera indices
         * @param results_batch Vector of detection results for each image
         * @return std::vector<Detection3D> 
         */
        std::vector<Detection3D> process_detections(const std::vector<cv::Mat>& images, const std::vector<int>& indices, const std::vector<std::vector<Result2D>>& results_batch);
        
        /**
         * @brief Publish detections to ROS topic
         * 
         * @param detections Vector of 3D detections to publish
         */
        void publish_detections(const std::vector<Detection3D>& detections);
        
        /**
         * @brief Publish annotated images with detection results
         * 
         * @param images Vector of input images
         * @param indices Vector of corresponding camera indices
         * @param results_batch Vector of detection results for each image
         */
        void publish_annotated_images(const std::vector<cv::Mat>& images, const std::vector<int>& indices, const std::vector<std::vector<Result2D>>& results_batch);
    };

} // namespace yolo_ros
