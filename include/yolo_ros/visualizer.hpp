#pragma once

#include <vector>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "yolo_ros/detector.hpp"

namespace yolo_ros {

    /**
     * @brief Handles visualization of detections (annotated images)
     * 
     */
    class Visualizer {
    public:
        Visualizer(rclcpp_lifecycle::LifecycleNode* node);
        ~Visualizer();

        /**
         * @brief Setup publishers for annotated images
         * 
         * @param rgbd_ids Camera IDs
         */
        void setup_publishers(const std::vector<std::string>& rgbd_ids);

        /**
         * @brief Set the class names for labeling.
         * 
         * @param class_names List of class names
         */
        void set_class_names(const std::vector<std::string>& class_names);

        /**
         * @brief Toggle publication state
         */
        void set_publish_annotated(bool publish) { publish_annotated_ = publish; }

        /**
         * @brief Publish annotated images with bounding boxes
         * 
         * @param images Raw images
         * @param indices Corresponding camera indices
         * @param results_batch Detection results for each image
         */
        void publish_annotated_images(
            const std::vector<cv::Mat>& images, 
            const std::vector<int>& indices, 
            const std::vector<std::vector<Result2D>>& results_batch
        );

    private:
        rclcpp_lifecycle::LifecycleNode* node_;
        bool publish_annotated_ = true;
        std::vector<std::string> class_names_;
        std::vector<rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr> annotated_pubs_;
        
        cv::Scalar get_color(int id);
    };

} // namespace yolo_ros
