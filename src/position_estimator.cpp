#include "yolo_ros/position_estimator.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <rclcpp/logging.hpp>
#include <rclcpp/clock.hpp>


const int THROTTLE_LOG_INTERVAL_MS = 1000;
const int MILLIMETERS_IN_METER = 1000;

namespace yolo_ros {

/**
 * @brief Construct a new Position Estimator object
 * 
 * @param depth_samples Number of depth samples to take around detection center.
 */    
PositionEstimator::PositionEstimator(int depth_samples) : _depth_samples(depth_samples) {}


/**
 * @brief Set the map defining real-world radii for each class ID.
 * Used for fallback depth estimation when depth data is missing.
 * 
 * @param radii Map where Key = Class ID, Value = Radius in meters.
 */
void PositionEstimator::set_class_radii(const std::map<int, float>& radii) {
    _class_radii = radii;
}

void PositionEstimator::set_camera_parameters(float height_m, float pitch_deg) {
    _camera_height = height_m;
    _camera_pitch = pitch_deg;
}


/**
 * @brief Compute the 3D position of a detection.
 * 
 * @param detection The 2D detection result.
 * @param depth_image The depth image corresponding to the detection.
 * @param camera_info Camera intrinsic parameters.
 * @param header Message header containing timestamp and frame information.
 * @return Detection3D The computed 3D detection.
 */
Detection3D PositionEstimator::compute_3d(
    const Result2D& detection,
    const cv::Mat& depth_image,
    const sensor_msgs::msg::CameraInfo& camera_info,
    const std_msgs::msg::Header& header
) {
    Detection3D det_3d;
    det_3d.result2d = detection;
    det_3d.header = header;

    float fx = camera_info.k[0];
    float fy = camera_info.k[4];
    float cx = camera_info.k[2];
    float cy = camera_info.k[5];

    // Safety check for uninitialized camera info
    if (fx <= 0.0f || fy <= 0.0f) {
        RCLCPP_ERROR_ONCE(rclcpp::get_logger("PositionEstimator"), 
            "Invalid camera intrinsics (fx=%.2f, fy=%.2f). Cannot compute 3D position.", fx, fy);
        return det_3d;
    }

    float estimated_dist = 0.0f;
    std::string depth_source = "none";

    // 1. Try Depth Image
    if (!depth_image.empty()) {
        estimated_dist = get_depth_from_region(detection.bbox, depth_image);
        if (estimated_dist > 0.0f) depth_source = "depth_image";
    }
    
    // 3. Fallback to estimation from size
    if (estimated_dist <= 0.0f && depth_source != "ground_plane") {
        estimated_dist = estimate_depth_from_size(detection.bbox, camera_info, detection.class_id);
        if (estimated_dist > 0.0f) depth_source = "size_estimation";
    }
    
    // If still invalid, log warning and return zeroed position
    if (estimated_dist <= 0.0f && depth_source != "ground_plane") {
        static rclcpp::Clock clock;
        RCLCPP_WARN_STREAM_THROTTLE(rclcpp::get_logger("PositionEstimator"),
            clock, THROTTLE_LOG_INTERVAL_MS,
            "Unable to estimate depth for detection class_id=" << detection.class_id 
            << " bbox=(" << detection.bbox.x << ", " << detection.bbox.y 
            << ", " << detection.bbox.width << ", " << detection.bbox.height << ")");

        det_3d.position.x = 0.0f;
        det_3d.position.y = 0.0f;
        det_3d.position.z = 0.0f;
        return det_3d;
    }

    // 3. Project to 3D Camera Coordinates
    
    // Calculate center of the bounding box
    // Usually one projects the BOTTOM center of the bounding box to the ground plane, 
    // because that's where the object touches the ground.
    float u = detection.bbox.x + detection.bbox.width / 2.0f; 
    float v = detection.bbox.y + detection.bbox.height; // Bottom edge

    // Calculate normalized image coordinates
    float norm_u = (u - cx) / fx;
    float norm_v = (v - cy) / fy;

    // Resolve specific depth vs slant range
    float z_cam = 0.0f;

    if (depth_source == "depth_image") {
        // Depth image typically gives Z distance (orthogonal to image plane)
        z_cam = estimated_dist; 
    } else if (depth_source == "size_estimation") {
        // Size estimation gives Z distance (orthogonal to image plane)
        // because we use similar triangles: Z = (f * real_R) / image_R
        z_cam = estimated_dist;
    } else if (_camera_height > 0.0f) {
        // Ground Plane Projection (Monocular Depth Estimation)
        // Camera Frame: X-Right, Y-Down, Z-Forward
        // Tilted down by pitch (around X-axis)
        // The ray is R = (norm_u, norm_v, 1)
        // In the "Level" frame (where Y is straight down), the ray is R_level = RotX(-pitch) * R
        // We want to find scalar t such that the point on the ray intersects the ground plane Y_level = H
        
        float pitch_rad = _camera_pitch * M_PI / 180.0f;
        // Rotation matrix for pitch down around X axis:
        // [1, 0, 0]
        // [0, cos(p), -sin(p)]
        // [0, sin(p), cos(p)]
        // The camera frame is tilted DOWN relative to the horizon.
        // So to get back to level frame, we rotate UP by pitch.
        // Or simpler:
        // The ground plane equation in Camera Frame is:
        // Y_cam * cos(theta) + Z_cam * sin(theta) = H
        // Substituting ray equations:
        // (v_norm * Z) * cos(theta) + (1 * Z) * sin(theta) = H
        // Z * (v_norm * cos(theta) + sin(theta)) = H
        // Z = H / (v_norm * cos(theta) + sin(theta))

        float denominator = norm_v * std::cos(pitch_rad) + std::sin(pitch_rad);
        if (std::abs(denominator) > 1e-3) {
            z_cam = _camera_height / denominator;
            if (z_cam < 0.0f) {
                // Ray points above horizon or parallel
                static rclcpp::Clock ros_clock;
                 RCLCPP_WARN_THROTTLE(rclcpp::get_logger("PositionEstimator"), ros_clock, THROTTLE_LOG_INTERVAL_MS,
                    "Projected ground point is behind camera or above horizon. z_cam=%.2f", z_cam);
                 z_cam = 0.0f;
            } else {
                 depth_source = "ground_plane";
            }
        }
    }

    if (z_cam <= 0.0f && depth_source != "ground_plane") { // Only error out if ground plane failed too
        det_3d.position.x = 0.0f;
        det_3d.position.y = 0.0f;
        det_3d.position.z = 0.0f;
        return det_3d;
    }

    // Recompute projected point based on z_cam
    // Note: z_cam was computed using bottom edge for ground plane, but we want the center of the object 
    // for X, Y usually. However, using bottom edge Z effectively places the object "on the ground" at that distance.
    // If we use that Z for the center u, we get the X position correctly.
    // The Y (height) of the object center would be off the ground.
    
    // For ground robot navigation, we usually care about the ground position (X, Y on ground).
    // In camera frame:
    // X_cam = norm_u_center * z_cam
    // Y_cam = norm_v_center * z_cam
    
    // Let's use center U for X calculation, but keep the Z determined by the bottom edge contact point.
    // This assumes the object is standing on the ground.
    float u_center = detection.bbox.x + detection.bbox.width / 2.0f;
    float norm_u_center = (u_center - cx) / fx;
    
    // If we used bottom edge for distance, we are finding the point on the ground.
    // Y_cam (down) for that point is roughly H (in level frame).
    // In camera frame it corresponds to the bottom of the object.
    
    float x_cam = norm_u_center * z_cam;
    // For Y_cam, we can simply say it's on the ground? 
    // Let's just project using the ray.
    float v_center = detection.bbox.y + detection.bbox.height / 2.0f;
     // If we use the center v for Y_cam, we get the 3D position of the center.
    // But z_cam was derived from bottom.
    // If the object is vertical, Z is roughly constant for the whole surface (approx).
    // Let's stick to the projection of the center point using the distance found from the bottom.
    float norm_v_center = (v_center - cy) / fy;
    float y_cam = norm_v_center * z_cam;
    
    // 4. Map to Coordinate Frame
    // Since the header frame is "camera_optical_frame", we should return coordinates in that frame.
    // X-Right, Y-Down, Z-Forward.
    
    det_3d.position.x = x_cam;
    det_3d.position.y = y_cam;
    det_3d.position.z = z_cam;

    RCLCPP_INFO(rclcpp::get_logger("PositionEstimator"), 
        "Position estimated: x=%.3f y=%.3f z=%.3f (class_id=%d) [Source: %s]", 
        det_3d.position.x, det_3d.position.y, det_3d.position.z, detection.class_id, depth_source.c_str());

    return det_3d;
}


/**
 * @brief Get the depth from region object
 * 
 * @param bbox Bounding box of the detected object.
 * @param depth Depth image.
 * @return float Estimated depth in meters.
 */
float PositionEstimator::get_depth_from_region(const cv::Rect& bbox, const cv::Mat& depth) {
    if (depth.empty()) return 0.0f;
    
    std::vector<float> valid_depths;
    valid_depths.reserve(_depth_samples);
    
    // Sample from gaussian distribution around center
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    
    int center_x = bbox.x + bbox.width / 2;
    int center_y = bbox.y + bbox.height / 2;
    // Python legacy: bb_radius = ((detection.bbox.size_x + detection.bbox.size_y) / 2) / 2
    int radius = ((bbox.width + bbox.height) / 2) / 2;
    if (radius < 1) radius = 1;

    for (int i=0; i<_depth_samples; ++i) {
        float angle = distribution(generator) * 2 * M_PI;
        
        // Python legacy: r = random.uniform(0, 1) * bb_radius
        // (Without sqrt more points are sampled closer to the center.)
        float r = distribution(generator) * radius;
        
        int x = center_x + static_cast<int>(r * cos(angle));
        int y = center_y + static_cast<int>(r * sin(angle));
        
        if (x >= 0 && x < depth.cols && y >= 0 && y < depth.rows) {
            // Depth encoding depends on type. Usually 16UC1 (mm) or 32FC1 (m)
            float val = 0.0f;
            if (depth.type() == CV_16U) {
                unsigned short d = depth.at<unsigned short>(y, x);
                if (d > 0) val = d / static_cast<float>(MILLIMETERS_IN_METER); // mm to m
            } else if (depth.type() == CV_32F) {
                val = depth.at<float>(y, x);
            }
            
            if (val > 0.1f) { // Min 10cm
                valid_depths.push_back(val);
            }
        }
    }
    
    if (valid_depths.empty()) 
    {
        // Don't warn every single frame to avoid spam, but finding 0 samples is an issue.
        return 0.0f;
    }
    
    // Return mean
    return std::accumulate(valid_depths.begin(), valid_depths.end(), 0.0f) / valid_depths.size();
}

/**
 * @brief Estimate depth based on object size in the image and known real-world size.
 * 
 * @param bbox Bounding box of the detected object.
 * @param info Camera intrinsic parameters.
 * @param class_id Class ID of the detected object.
 * @return float Estimated depth in meters.
 */
// Reinterpret based on the provided Python legacy implementation
float PositionEstimator::estimate_depth_from_size(const cv::Rect& bbox, const sensor_msgs::msg::CameraInfo& info, int class_id) {
    
    if (_class_radii.find(class_id) == _class_radii.end())
    {
        RCLCPP_WARN(rclcpp::get_logger("PositionEstimator"), "No radius info for class_id=%d to estimate depth", class_id);
        return 0.0f; // No radius info
    }
    
    float real_radius = _class_radii.at(class_id);
    
    // Legacy Python implementation:
    // bb_radius = ((detection.bbox.size_x + detection.bbox.size_y) / 2) / 2
    // z = node.class_radii[class_index] / bb_radius * camera_info.k[0]
    
    float image_radius = ((bbox.width + bbox.height) / 2.0f) / 2.0f;

    float fx = info.k[0]; // fx camera intrinsic
    
    // Prevent division by zero
    if (image_radius < 1.0f) return 0.0f;
    
    // Z = fx * real_radius / image_radius
    return (fx * real_radius) / image_radius;
}

} // namespace yolo_ros
