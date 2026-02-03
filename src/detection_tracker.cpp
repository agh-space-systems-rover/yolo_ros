#include "yolo_ros/detection_tracker.hpp"
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/exceptions.h>
#include <algorithm>
#include <random>

namespace yolo_ros {

/**
 * @brief Construct a new Detection Group:: Detection Group object
 * 
 * @param initial_detection 
 * @param max_history_param 
 */
DetectionGroup::DetectionGroup(const std::string& id, const Detection3D& initial_detection, int max_history_param) 
    : id_(id), max_history_(max_history_param)
{
    add_measurement(initial_detection);
}

/**
 * @brief Convert a new detection measurement into the group.
 * 
 * @param det 
 */
void DetectionGroup::add_measurement(const Detection3D& det) {
    measurements_.push_back(det);
    if (measurements_.size() > (size_t)max_history_) {
        measurements_.pop_front();
    }
    last_update_ = det.header.stamp;
}

/**
 * @brief Check if the detection group is confirmed based on the number of measurements.
 * 
 * @param temporal_threshold 
 * @return true 
 * @return false 
 */
bool DetectionGroup::is_confirmed(int temporal_threshold) const {
    return (int)measurements_.size() >= temporal_threshold;
}

/**
 * @brief Check if the detection group is stale based on the last update time.
 * 
 * @param current_time Current time for comparison.
 * @param max_age_seconds Maximum allowed age in seconds before considered stale.
 * @return true If the detection group is stale.
 * @return false If the detection group is not stale.
 */
bool DetectionGroup::is_stale(const rclcpp::Time& current_time, double max_age_seconds) const {
    double seconds = (current_time - last_update_).seconds();
    return seconds > max_age_seconds;
}

/**
 * @brief Calculate the average detection from the history of measurements.
 * 
 * @return Detection3D 
 */
Detection3D DetectionGroup::get_average_detection() const {
    if (measurements_.empty()) return Detection3D();

    Detection3D avg = measurements_.back(); // Start with latest metadata
    
    double x = 0, y = 0, z = 0;
    for (const auto& m : measurements_) {
        x += m.position.x;
        y += m.position.y;
        z += m.position.z;
    }
    size_t n = measurements_.size();
    avg.position.x = x / n;
    avg.position.y = y / n;
    avg.position.z = z / n;
    
    // ID assignment
    avg.result2d.id = std::stoi(id_); 

    return avg;
}

// -----------------------------------------------------------------------------

/**
 * @brief Construct a new Detection Tracker:: Detection Tracker object
 * 
 * @param merge_radius radius for spatial merging
 * @param temporal_window 
 * @param temporal_threshold 
 */
DetectionTracker::DetectionTracker(float merge_radius, int temporal_window, int temporal_threshold)
    : merge_radius_(merge_radius), temporal_window_(temporal_window), temporal_threshold_(temporal_threshold)
{}


/**
 * @brief Calculate the Euclidean distance between two 3D points.
 * 
 * @param p1 First 3D point.
 * @param p2 Second 3D point.
 * @return float Euclidean distance between p1 and p2.
 */
float DetectionTracker::dist3d(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

/**
 * @brief Process new detections: transform, merge, and track.
 * 
 * @param new_detections vector of new 3D detections
 * @param tf_buffer TF2 buffer for transformations
 * @param target_frame Target frame to transform detections into
 * @return std::vector<Detection3D> 
 */
std::vector<Detection3D> DetectionTracker::process(
    const std::vector<Detection3D>& new_detections, 
    const std::shared_ptr<tf2_ros::Buffer>& tf_buffer,
    const std::string& target_frame,
    const rclcpp::Time& current_time
) {
    // 1. Transform all incoming to World Frame
    auto world_detections = transform_to_world(new_detections, tf_buffer, target_frame);
    
    // 2. Spatial Merge (consolidate duplicates from overlapping cameras)
    auto merged = spatial_merge(world_detections);
    
    // 3. Temporal Filter (Tracker Logic)
    return temporal_filter(merged, current_time);
}

/**
 * @brief Transform detections to the target world frame using TF2.
 * 
 * @param dets Vector of 3D detections.
 * @param tf_buffer TF2 buffer for transformations.
 * @param target_frame Target frame to transform detections into.
 * @return std::vector<Detection3D> 
 */
std::vector<Detection3D> DetectionTracker::transform_to_world(
    const std::vector<Detection3D>& dets,
    const std::shared_ptr<tf2_ros::Buffer>& tf_buffer, 
    const std::string& target_frame
) {
    std::vector<Detection3D> output;
    output.reserve(dets.size());
    
    for (const auto& det : dets) {
        Detection3D det_world = det;
        try {
             // Look up transform
             geometry_msgs::msg::PoseStamped pose_in, pose_out;
             pose_in.header = det.header;
             pose_in.pose.position = det.position;
             pose_in.pose.orientation.w = 1.0;
             
             // Timeout 0.0 because strictly we should have the TF by now or we use latest
             tf_buffer->transform(pose_in, pose_out, target_frame, tf2::durationFromSec(0.05));
             
             det_world.position = pose_out.pose.position;
             det_world.header = pose_out.header;
             output.push_back(det_world);
        } catch (const tf2::TransformException& ex) {
            continue;
        }
    }
    return output;
}


/**
 * @brief Spatially merge detections that are close together and of the same class.
 * 
 * @param dets Vector of 3D detections.
 * @return std::vector<Detection3D> 
 */
std::vector<Detection3D> DetectionTracker::spatial_merge(const std::vector<Detection3D>& dets) {
    // Simple greedy clustering
    // If Det A and Det B are close (< radius) and same class -> merge
    // "Merge" means avg position
    
    if (dets.empty()) return {};
    
    std::vector<Detection3D> merged;
    std::vector<bool> used(dets.size(), false);
    
    for (size_t i = 0; i < dets.size(); ++i) {
        if (used[i]) continue;
        
        std::vector<Detection3D> cluster;
        cluster.push_back(dets[i]);
        used[i] = true;
        
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (used[j]) continue;
            
            // Check class
            if (dets[i].result2d.class_id != dets[j].result2d.class_id) continue;
            
            // Check dist
            if (dist3d(dets[i].position, dets[j].position) < merge_radius_) {
                cluster.push_back(dets[j]);
                used[j] = true;
            }
        }
        
        // Merge cluster
        Detection3D m = cluster[0];
        if (cluster.size() > 1) {
            double x=0, y=0, z=0;
            float max_score = 0;
            for (const auto& c : cluster) {
                x += c.position.x;
                y += c.position.y;
                z += c.position.z;
                if (c.result2d.score > max_score) max_score = c.result2d.score;
            }
            m.position.x = x / cluster.size();
            m.position.y = y / cluster.size();
            m.position.z = z / cluster.size();
            m.result2d.score = max_score;
        }
        merged.push_back(m);
    }
    return merged;
}

/**
 * @brief Apply temporal filtering to a set of 3D detections to track objects over time.
 * 
 * @param dets Vector of current 3D detections.
 * @return std::vector<Detection3D> Filtered and tracked 3D detections.
 */
std::vector<Detection3D> DetectionTracker::temporal_filter(const std::vector<Detection3D>& dets, const rclcpp::Time& current_time) {
    // Data Assocation: Nearest Neighbor
    // Update existing groups
    // Create new groups
    // Delete stale groups
    
    // This is a simplified tracker.
    
    std::vector<bool> matched(dets.size(), false);
    
    // 1. Update existing tracks
    for (auto& [id, group] : history_) {
        double best_dist = merge_radius_;
        int best_idx = -1;
        
        Detection3D center = group.get_average_detection();
        
        for (size_t i = 0; i < dets.size(); ++i) {
            if (matched[i]) continue;
             // Check class
            if (dets[i].result2d.class_id != center.result2d.class_id) continue;
            
            float d = dist3d(dets[i].position, center.position);
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
            }
        }
        
        if (best_idx != -1) {
            group.add_measurement(dets[best_idx]);
            matched[best_idx] = true;
        }
    }
    
    // 2. Create new tracks
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!matched[i]) {
            std::string new_id = std::to_string(next_id_++);
            DetectionGroup new_group(new_id, dets[i], temporal_window_);
            history_.insert({new_group.get_id(), new_group});
        }
    }
    
    // 3. Collect Output & Cleanup Stale
    std::vector<Detection3D> output;
    
    auto it = history_.begin();
    while (it != history_.end()) {
        // Remove if stale (e.g. 1 second no update)
        if (it->second.is_stale(current_time, 1.0)) {
            it = history_.erase(it);
        } else {
            if (it->second.is_confirmed(temporal_threshold_)) {
                auto out = it->second.get_average_detection();
                output.push_back(out);
            }
            ++it;
        }
    }
    
    return output;
}

} // namespace yolo_ros
