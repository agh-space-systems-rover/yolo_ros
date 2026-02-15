#include "yolo_ros/visualizer.hpp"

namespace yolo_ros {

Visualizer::Visualizer(rclcpp_lifecycle::LifecycleNode* node) : node_(node) {}

Visualizer::~Visualizer() {}

void Visualizer::setup_publishers(const std::vector<std::string>& rgbd_ids) {
    annotated_pubs_.clear();
    for (size_t i = 0; i < rgbd_ids.size(); ++i) {
        std::string id = rgbd_ids[i];
        std::string prefix = "/" + id;
        if (id.front() == '/') prefix = id;
        std::string topic = prefix + "/yolo_annotated";
        
        // Create publisher
        auto pub = node_->create_publisher<sensor_msgs::msg::Image>(topic, 10);
        
        // Since we are setting up inside on_activate, activate immediately if node is active or transitioning
        // However, checking node state might be complex. 
        // Typically, if created in on_activate, we activate it.
        // If created in on_configure, we wait for on_activate.
        // The node calls this in on_activate.
        pub->on_activate();
        
        annotated_pubs_.push_back(pub);
    }
}

void Visualizer::set_class_names(const std::vector<std::string>& class_names) {
    class_names_ = class_names;
}

void Visualizer::publish_annotated_images(
    const std::vector<cv::Mat>& images, 
    const std::vector<int>& indices, 
    const std::vector<std::vector<Result2D>>& results_batch) 
{
    if (!publish_annotated_) return;

    for (size_t i = 0; i < images.size(); ++i) {
        int cam_idx = indices[i];
        if (cam_idx >= (int)annotated_pubs_.size()) continue;
        
        // Check if publisher is active
        if (!annotated_pubs_[cam_idx]->is_activated()) continue;

        cv::Mat annotated = images[i].clone();
        for (const auto& det : results_batch[i]) {
            cv::Scalar color = get_color(det.class_id);
            cv::rectangle(annotated, det.bbox, color, 2);
            
            std::string label = std::to_string(det.class_id); // Default
            if (det.class_id >= 0 && det.class_id < (int)class_names_.size()) {
                 label = class_names_[det.class_id];
            }
            label += ": " + std::to_string(det.score).substr(0, 4);

            int baseline;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(annotated, 
                cv::Point(det.bbox.x, det.bbox.y - textSize.height - 5), 
                cv::Point(det.bbox.x + textSize.width, det.bbox.y), 
                color, -1);
            cv::putText(annotated, label, cv::Point(det.bbox.x, det.bbox.y - 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
        }

        sensor_msgs::msg::Image::SharedPtr msg;
        try {
            msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", annotated).toImageMsg();
        } catch (cv_bridge::Exception& e) {
            continue;
        }

        msg->header.stamp = node_->get_clock()->now(); 
        
        annotated_pubs_[cam_idx]->publish(*msg);
    }
}

cv::Scalar Visualizer::get_color(int id) {
    const int colors[][3] = {
        {255,0,0}, {0,255,0}, {0,0,255}, {255,255,0}, {0,255,255}, {255,0,255}
    };
    int idx = std::abs(id) % 6;
    return cv::Scalar(colors[idx][0], colors[idx][1], colors[idx][2]);
}

} // namespace yolo_ros
