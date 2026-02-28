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
// Helper to remove oldest history if not updated
void DetectionGroup::pop_oldest() {
    if (!measurements_.empty()) {
        measurements_.pop_front();
    }
}

/**
 * @brief Construct a new Detection Tracker:: Detection Tracker object
 * 
 * @param merge_radius radius for spatial merging
 * @param temporal_window 
 * @param temporal_threshold 
 */
DetectionTracker::DetectionTracker(float merge_radius, int temporal_window, int temporal_threshold)
    : merge_radius_(merge_radius), temporal_window_(temporal_window), temporal_threshold_(temporal_threshold)
{
    // Initialize random seed if needed, though std::random_device is typical
}


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
    
    // 2. Temporal Filter (Tracker Logic updates tracks and outputs current stable set)
    // In Python code: temporal_filter does both spatial association to existing groups and creation of new ones.
    // Our C++ structure separates spatial_merge (within current frame) and temporal_filter (across time).
    // The Python `temporal_filter` function actually takes all incoming detections and updates history.
    // It does not explicitly merge duplicates within the same frame before updating history, 
    // BUT it checks `if np.linalg.norm(...) < group_radius` against existing groups.
    // If we have 2 overlapping detections in the SAME frame, the Python code might add both to the same group 
    // or different groups depending on implementation details (it pushes to first matching group).
    
    // To match Python logic closer:
    return temporal_filter(world_detections, current_time);
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
            if (det.header.frame_id.empty()) {
                 continue;
             }

             // Look up transform
             geometry_msgs::msg::PointStamped point_in, point_out;
             point_in.header.frame_id = det.header.frame_id;
             point_in.header.stamp = builtin_interfaces::msg::Time();
             point_in.point = det.position;
             
             // Use 0.0 timeout to just get the latest available transform or fail immediately if not available
             // (The python script used 'Time()' which implies latest available)
             tf_buffer->transform(point_in, point_out, target_frame, tf2::durationFromSec(0.05));
             
             det_world.position = point_out.point;
             det_world.header.frame_id = target_frame;
             det_world.header.stamp = det.header.stamp;
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
    // Implement logic similar to Python's temporal_filter
    
    // 1. Update existing groups with new detections
    std::vector<bool> incoming_matched(dets.size(), false);
    
    // We iterate over a copy or modify carefully - map is fine.
    for (auto& [id, group] : history_) {
        bool group_was_updated = false;
        
        // Python code: iterates all incoming, checks if close to group center
        // Adds ALL matching detections to the group.
        Detection3D center = group.get_average_detection();
        
        for (size_t i = 0; i < dets.size(); ++i) {
            if (incoming_matched[i]) continue; // Already assigned? Python script removes them.
            
            if (dets[i].result2d.class_id != center.result2d.class_id) continue;
            
            float d = dist3d(dets[i].position, center.position);
            // Uses group radius (merge_radius_)
            if (d < merge_radius_) {
                group.add_measurement(dets[i]);
                incoming_matched[i] = true;
                group_was_updated = true;
            }
        }
        
        if (!group_was_updated) {
             // Python logic: if no match, pop oldest to "move the window"
             // This effectively decays the track if it's not being updated.
             group.pop_oldest();
             if (group.is_empty()) { // Need is_empty on group
                  // We can't erase from map while iterating easily with this loop structure.
                  // But usually "stale" check in step 3 handles removal.
                  // However, python script removes it immediately from history list if empty.
                  // Here we can mark it for deletion?
                  // Let's rely on step 3 cleaning up empty groups or stale ones.
             }
        }
    }
    
    // 2. Create new groups for unmatched detections
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!incoming_matched[i]) {
            std::string new_id = std::to_string(next_id_++);
            DetectionGroup new_group(new_id, dets[i], temporal_window_);
            history_.insert({new_group.get_id(), new_group});
        }
    }
    
    // 3. Filter output and Cleanup
    std::vector<Detection3D> output;
    
    // We used a map, but we iterate and erase.
    // It's safer to just iterate and handle erasures carefully.
    
    for (auto it = history_.begin(); it != history_.end(); ) {
        DetectionGroup& group = it->second;
        
        // Remove if empty (decayed completely) or STALE (too old last update)
        // Python logic: if empty immediately remove.
        // Also check is_stale for cleanup (the python script doesn't check is_stale for removal explicitly inside the loop 
        //   but relies on group.empty() after pop(), 
        //   AND a final pass that checks stale?)
        // Wait, python script: "if detection_group.empty(): temporal_history.remove()"
        
        bool remove = false;
        if (group.is_empty()) {
            remove = true;
        } 
        // Also remove if stale (safeguard)
        else if (group.is_stale(current_time, 2.0)) { // 2.0s leeway
            remove = true;
        }
        
        if (remove) {
            it = history_.erase(it);
        } else {
             // Only output if confirmed
            if (group.is_confirmed(temporal_threshold_)) {
                output.push_back(group.get_average_detection());
            }
            ++it;
        }
    }
    
    return output;
}

} // namespace yolo_ros
