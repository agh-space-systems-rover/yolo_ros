/**
 * @file detection_tracker.hpp
 * @author Mateusz Wójcik (mateuszwojcikv@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2026-02-01
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include <vector>
#include <deque>
#include <map>
#include <string>

#include <tf2_ros/buffer.h>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "yolo_ros/position_estimator.hpp"

namespace yolo_ros {

    class DetectionGroup {
    public:
        DetectionGroup() = default;
        DetectionGroup(const std::string& id, const Detection3D& initial_detection, int max_history_param);
        void add_measurement(const Detection3D& det);
        void pop_oldest(); // new method to match python logic
        bool is_empty() const { return measurements_.empty(); }
        bool is_confirmed(int temporal_threshold) const;
        bool is_stale(const rclcpp::Time& current_time, double max_age_seconds) const;
        Detection3D get_average_detection() const;
        const std::string& get_id() const { return id_; }

    private:
        std::string id_;
        std::deque<Detection3D> measurements_;
        int max_history_;
        rclcpp::Time last_update_;
    };

    class DetectionTracker {
    public:
        DetectionTracker(float merge_radius, int temporal_window, int temporal_threshold);

        std::vector<Detection3D> process(
            const std::vector<Detection3D>& new_detections, 
            const std::shared_ptr<tf2_ros::Buffer>& tf_buffer,
            const std::string& target_frame,
            const rclcpp::Time& current_time
        );

    private:
        float merge_radius_;
        int temporal_window_;
        int temporal_threshold_;
        int next_id_ = 1;

        std::map<std::string, DetectionGroup> history_; // ID -> Group

        // Core logic
        std::vector<Detection3D> transform_to_world(
             const std::vector<Detection3D>& dets,
             const std::shared_ptr<tf2_ros::Buffer>& tf_buffer, 
             const std::string& target_frame
        );
        std::vector<Detection3D> spatial_merge(const std::vector<Detection3D>& dets);
        std::vector<Detection3D> temporal_filter(const std::vector<Detection3D>& dets, const rclcpp::Time& current_time);
        
        static float dist3d(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2);
    };

} // namespace yolo_ros
