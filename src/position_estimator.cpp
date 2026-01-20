#include "yolo_ros/position_estimator.hpp"
#include <cmath>
#include <random>

namespace yolo_ros {

PositionEstimator::PositionEstimator() {}

void PositionEstimator::set_class_radii(const std::map<int, float>& radii) {
    class_radii_ = radii;
}

Detection3D PositionEstimator::compute_3d(
    const Result2D& detection,
    const cv::Mat& depth_image,
    const sensor_msgs::msg::CameraInfo& camera_info,
    const std_msgs::msg::Header& header
) {
    Detection3D det_3d;
    det_3d.result2d = detection;
    det_3d.header = header;

    float z = -1.0f;
    float fx = camera_info.k[0];
    float fy = camera_info.k[4];
    float cx = camera_info.k[2];
    float cy = camera_info.k[5];

    // 1. Try Depth Image
    if (!depth_image.empty()) {
        z = get_depth_from_region(detection.bbox, depth_image);
    }

    // 2. Fallback to estimation from size
    if (z <= 0.0f) {
        z = estimate_depth_from_size(detection.bbox, camera_info, detection.class_id);
    }
    
    // If still invalid, default to something safe or keep -1 to filter later
    if (z <= 0.0f) z = 1.0f; // Warn in logs

    // 3. Project to 3D
    // x = (u - cx) * z / fx
    // y = (v - cy) * z / fy
    float u = detection.bbox.x + detection.bbox.width / 2.0f;
    float v = detection.bbox.y + detection.bbox.height / 2.0f;

    det_3d.position.x = (u - cx) * z / fx;
    det_3d.position.y = (v - cy) * z / fy;
    det_3d.position.z = z;

    return det_3d;
}

float PositionEstimator::get_depth_from_region(const cv::Rect& bbox, const cv::Mat& depth) {
    if (depth.empty()) return 0.0f;
    
    // Use random sampling logic from python or simple median of center ROI
    // Python code used 100 samples in a circle.
    
    std::vector<float> valid_depths;
    valid_depths.reserve(POSITION_ESTIMATION_DEPTH_SAMPLES);
    
    // Simple center implementation
    // Better: Sample from gaussian distribution around center
    
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    
    int center_x = bbox.x + bbox.width / 2;
    int center_y = bbox.y + bbox.height / 2;
    int radius = std::min(bbox.width, bbox.height) / 4; 

    for (int i=0; i<POSITION_ESTIMATION_DEPTH_SAMPLES; ++i) {
        // Simple uniform within box for now, or just iterate center
        // Let's implement Python logic approx: random angle and radius
        float angle = distribution(generator) * 2 * M_PI;
        float r = distribution(generator) * radius;
        
        int x = center_x + static_cast<int>(r * cos(angle));
        int y = center_y + static_cast<int>(r * sin(angle));
        
        if (x >= 0 && x < depth.cols && y >= 0 && y < depth.rows) {
            // Depth encoding depends on type. Usually 16UC1 (mm) or 32FC1 (m)
            float val = 0.0f;
            if (depth.type() == CV_16U) {
                unsigned short d = depth.at<unsigned short>(y, x);
                if (d > 0) val = d / 1000.0f; // mm to m
            } else if (depth.type() == CV_32F) {
                val = depth.at<float>(y, x);
            }
            
            if (val > 0.1f) { // Min 10cm
                valid_depths.push_back(val);
            }
        }
    }
    
    if (valid_depths.empty()) return 0.0f;
    
    // Return mean
    float sum = 0.0f;
    for (float d : valid_depths) sum += d;
    return sum / valid_depths.size();
}

float PositionEstimator::estimate_depth_from_size(const cv::Rect& bbox, const sensor_msgs::msg::CameraInfo& info, int class_id) {
    if (class_radii_.find(class_id) == class_radii_.end()) return 0.0f;
    
    float real_radius = class_radii_.at(class_id);
    float image_radius = (bbox.width + bbox.height) / 4.0f; // Avg radius in pixels
    
    float fx = info.k[0];
    
    // Z = f * real / image
    if (image_radius < 1.0f) return 0.0f;
    return (fx * real_radius) / image_radius;
}

} // namespace yolo_ros
