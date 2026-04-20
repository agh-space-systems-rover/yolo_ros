#pragma once

#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/simple_filter.h>
#include <cv_bridge/cv_bridge.hpp>

namespace yolo_ros {

    /**
     * @brief Helper class for compressed image subscription
     */
    class CompressedSubscriberWrapper : public message_filters::SimpleFilter<sensor_msgs::msg::Image>
    {
    public:
        /**
         * @brief Construct a new Compressed Subscriber Wrapper object
         * 
         * @param node Lifecycle node to create subscription on
         * @param topic Topic to subscribe to
         * @param qos QoS profile
         * @param target_encoding Optional target encoding for decoding (e.g. "bgr8", "mono16", or "" for auto)
         */
        CompressedSubscriberWrapper(rclcpp_lifecycle::LifecycleNode* node, const std::string& topic, 
                                   const rmw_qos_profile_t& qos, const std::string& target_encoding = "");
    private:
        rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
    };

    /**
     * @brief Structure to hold data gathered from a single camera sync event
     */
    struct CameraData {
        cv::Mat color;
        cv::Mat depth; 
        sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
        std_msgs::msg::Header header;
        int camera_index;
    };

    /**
     * @brief Manages subscriptions and synchronization for multiple RGBD cameras.
     */
    class CameraManager {
    public:
        CameraManager(rclcpp_lifecycle::LifecycleNode* node);
        ~CameraManager();

        void configure(
            int num_cameras,
            const std::vector<std::string>& rgbd_ids,
            const std::string& color_transport,
            const std::string& depth_transport,
            bool subscribe_depth
        );

        void activate();
        void deactivate();

        /**
         * @brief Gather images from all cameras that have new synchronized data.
         * 
         * @return std::vector<CameraData> List of camera data ready for processing
         */
        std::vector<CameraData> gather_images();

    private:
        rclcpp_lifecycle::LifecycleNode* node_;
        
        using SyncPolicy1 = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo
        >;
        using Sync1 = message_filters::Synchronizer<SyncPolicy1>;
        
        struct CameraContext {
            int index;
            std::shared_ptr<message_filters::SimpleFilter<sensor_msgs::msg::Image>> color_sub;
            std::shared_ptr<message_filters::SimpleFilter<sensor_msgs::msg::Image>> depth_sub;
            std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo, rclcpp_lifecycle::LifecycleNode>> info_sub;
            std::shared_ptr<Sync1> sync;
            
            // Buffers
            sensor_msgs::msg::Image::ConstSharedPtr last_color;
            sensor_msgs::msg::Image::ConstSharedPtr last_depth;
            sensor_msgs::msg::CameraInfo::ConstSharedPtr last_info;
            bool has_new_data = false;
            
            // Ownership handles for subscribers created inside
            std::shared_ptr<void> color_sub_handle;
            std::shared_ptr<void> depth_sub_handle;
        };

        std::vector<std::shared_ptr<CameraContext>> cameras_;
        
        // Params needed for re-activation
        int num_cameras_;
        std::vector<std::string> rgbd_ids_;
        std::string color_transport_;
        std::string depth_transport_;
        bool subscribe_depth_;

        void on_camera_data(
            const sensor_msgs::msg::Image::ConstSharedPtr& color,
            const sensor_msgs::msg::Image::ConstSharedPtr& depth,
            const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info,
            int camera_index
        );
    };

} // namespace yolo_ros
