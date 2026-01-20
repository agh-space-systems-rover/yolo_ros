#include "yolo_ros/position_estimator.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <rclcpp/logging.hpp>


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

    float z = 0.0f;
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

    // 1. Try Depth Image
    if (!depth_image.empty()) {
        z = get_depth_from_region(detection.bbox, depth_image);
    }

    // 2. Fallback to estimation from size
    if (z <= 0.0f) {
        z = estimate_depth_from_size(detection.bbox, camera_info, detection.class_id);
    }
    
    // If still invalid, default to something safe
    if (z <= 0.0f) {
        
        RCLCPP_WARN_STREAM_THROTTLE(rclcpp::get_logger("PositionEstimator"),
            *rclcpp::Clock().get_clock_handle(), THROTTLE_LOG_INTERVAL_MS,
            "Unable to estimate depth for detection class_id=" << detection.class_id 
            << " bbox=(" << detection.bbox.x << ", " << detection.bbox.y 
            << ", " << detection.bbox.width << ", " << detection.bbox.height << ")");

        det_3d.position.x = 0.0f;
        det_3d.position.y = 0.0f;
        det_3d.position.z = 0.0f;
        return det_3d;
    }

    // 3. Project to 3D
    // Using Pinhole Camera Model
    // X = (u - cx) * Z / fx
    // Y = (v - cy) * Z / fy
    float u = detection.bbox.x + detection.bbox.width / 2.0f;
    float v = detection.bbox.y + detection.bbox.height / 2.0f;

    det_3d.position.x = (u - cx) * z / fx;
    det_3d.position.y = (v - cy) * z / fy;
    det_3d.position.z = z;

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
    
    // Use random sampling logic from python or simple median of center ROI
    
    std::vector<float> valid_depths;
    valid_depths.reserve(_depth_samples);
    
    // Simple center implementation
    // Better: Sample from gaussian distribution around center
    
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    
    int center_x = bbox.x + bbox.width / 2;
    int center_y = bbox.y + bbox.height / 2;
    int radius = std::min(bbox.width, bbox.height) / 4; 

    for (int i=0; i<_depth_samples; ++i) {
        // Simple uniform within box for now, or just iterate center
        float angle = distribution(generator) * 2 * M_PI;
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
    
    // If no valid depths
    if (valid_depths.empty()) 
    {
        RCLCPP_WARN(rclcpp::get_logger("PositionEstimator"), "No valid depth samples found in bbox (%d, %d, %d, %d)", 
            bbox.x, bbox.y, bbox.width, bbox.height);
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
float PositionEstimator::estimate_depth_from_size(const cv::Rect& bbox, const sensor_msgs::msg::CameraInfo& info, int class_id) {
    
    
    if (_class_radii.find(class_id) == _class_radii.end())
    {
        RCLCPP_WARN(rclcpp::get_logger("PositionEstimator"), "No radius info for class_id=%d to estimate depth", class_id);
        return 0.0f; // No radius info
    }
    
    float real_radius = _class_radii.at(class_id);
    float image_radius = (bbox.width + bbox.height) / 4.0f; // Avg radius in pixels

    float fx = info.k[0]; // fx camera intrinsic
    
    // Prevent division by zero
    if (image_radius < 1.0f) return 0.0f;
    
    // Z = fx * real / image
    return (fx * real_radius) / image_radius;
}

} // namespace yolo_ros
