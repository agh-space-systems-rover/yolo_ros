#include "yolo_ros/yolo_detect_node.hpp"
#include "yolo_ros/position_estimator.hpp"
#include "yolo_ros/detection_tracker.hpp"
#include <cv_bridge/cv_bridge.h>

namespace yolo_ros {

YoloDetectNode::YoloDetectNode(const rclcpp::NodeOptions & options)
    : rclcpp_lifecycle::LifecycleNode("yolo_detect", options)
{
    declare_parameter("num_cameras", 1);
    declare_parameter("subscribe_depth", false);
    declare_parameter("color_transport", "compressed");
    declare_parameter("depth_transport", "raw");
    declare_parameter("model", "");
    declare_parameter("class_names", std::vector<std::string>());
    declare_parameter("class_radii", std::vector<double>());
    declare_parameter("world_frame", "odom");
    declare_parameter("merge_radius", 0.5);
    declare_parameter("temporal_window", 5);
    declare_parameter("temporal_threshold", 3);
    declare_parameter("confidence_threshold", 0.5);
    declare_parameter("publish_tf", true);
    declare_parameter("publish_annotated", true);
    declare_parameter("annotated_transport", "compressed");
    declare_parameter("debug_mode", false);
    declare_parameter("rgbd_ids", std::vector<std::string>());
    declare_parameter("rate", 10.0);
}

YoloDetectNode::CallbackReturn YoloDetectNode::on_configure(const rclcpp_lifecycle::State &) {
    num_cameras_ = get_parameter("num_cameras").as_int();
    subscribe_depth_ = get_parameter("subscribe_depth").as_bool();
    color_transport_ = get_parameter("color_transport").as_string();
    depth_transport_ = get_parameter("depth_transport").as_string();
    model_path_ = get_parameter("model").as_string();
    class_names_ = get_parameter("class_names").as_string_array();
    class_radii_param_ = get_parameter("class_radii").as_double_array();
    world_frame_ = get_parameter("world_frame").as_string();
    merge_radius_ = get_parameter("merge_radius").as_double();
    temporal_window_ = get_parameter("temporal_window").as_int();
    temporal_threshold_ = get_parameter("temporal_threshold").as_int();
    confidence_threshold_ = get_parameter("confidence_threshold").as_double();
    publish_tf_ = get_parameter("publish_tf").as_bool();
    publish_annotated_ = get_parameter("publish_annotated").as_bool();
    annotated_transport_ = get_parameter("annotated_transport").as_string();
    debug_mode_ = get_parameter("debug_mode").as_bool();
    rgbd_ids_ = get_parameter("rgbd_ids").as_string_array();
    
    if (!rgbd_ids_.empty()) {
        num_cameras_ = rgbd_ids_.size();
    }

    // Detector
    detector_ = std::make_unique<YoloOpenCVDetector>();
    ModelConfig cfg;
    cfg.confidence_threshold = confidence_threshold_;
    cfg.class_names = class_names_; // Not strictly needed by detector if only ID returned, but passed anyway
    
    if (!detector_->load(model_path_, cfg)) {
        RCLCPP_ERROR(get_logger(), "Failed to load model: %s", model_path_.c_str());
        return CallbackReturn::FAILURE;
    }

    // Estimator
    auto pos_estimator = std::make_unique<PositionEstimator>();
    std::map<int, float> radii_map;
    size_t count = std::min(class_names_.size(), class_radii_param_.size());
    for (size_t i = 0; i < count; ++i) {
        radii_map[i] = class_radii_param_[i];
    }
    pos_estimator->set_class_radii(radii_map);
    estimator_ = std::move(pos_estimator);

    // Tracker
    tracker_ = std::make_unique<DetectionTracker>(merge_radius_, temporal_window_, temporal_threshold_);

    // Camera Manager
    camera_man_ = std::make_unique<CameraManager>(this);
    
    // Visualizer
    visualizer_ = std::make_unique<Visualizer>(this);
    visualizer_->set_publish_annotated(publish_annotated_);
    visualizer_->set_class_names(class_names_);

    // TF
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Publisher
    detection_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);

    return CallbackReturn::SUCCESS;
}

YoloDetectNode::CallbackReturn YoloDetectNode::on_activate(const rclcpp_lifecycle::State & state) {
    LifecycleNode::on_activate(state); 
    if (detection_pub_) {
        detection_pub_->on_activate();
    }

    // Configure and Activate Camera Manager
    camera_man_->configure(num_cameras_, rgbd_ids_, color_transport_, depth_transport_, subscribe_depth_);
    camera_man_->activate();

    // Visualizer Publishers
    visualizer_->setup_publishers(rgbd_ids_);

    // Rate Timer
    double rate = get_parameter("rate").as_double();
    timer_ = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds((int)(1000.0/rate)), 
        std::bind(&YoloDetectNode::timer_callback, this));
    
    RCLCPP_INFO(get_logger(), "YOLO Node Activated");
    return CallbackReturn::SUCCESS;
}

YoloDetectNode::CallbackReturn YoloDetectNode::on_deactivate(const rclcpp_lifecycle::State & state) {
    timer_.reset();
    camera_man_->deactivate();
    if (detection_pub_) {
        detection_pub_->on_deactivate();
    }
    LifecycleNode::on_deactivate(state);
    return CallbackReturn::SUCCESS;
}

YoloDetectNode::CallbackReturn YoloDetectNode::on_cleanup(const rclcpp_lifecycle::State &) {
    detector_.reset();
    estimator_.reset();
    tracker_.reset();
    camera_man_.reset();
    visualizer_.reset();
    tf_buffer_.reset();
    tf_listener_.reset();
    detection_pub_.reset();
    return CallbackReturn::SUCCESS;
}

YoloDetectNode::CallbackReturn YoloDetectNode::on_shutdown(const rclcpp_lifecycle::State & state) {
    return CallbackReturn::SUCCESS;
}

void YoloDetectNode::timer_callback() {
    if (debug_mode_) {
        RCLCPP_INFO(get_logger(), "Timer callback start");
    }

    // 1. Gather Images
    // Returns vector of CameraData (struct with color, depth, info, index)
    auto gathered_data = camera_man_->gather_images();
    
    if (gathered_data.empty()) return;

    // Prepare infer batch
    std::vector<cv::Mat> infer_batch;
    std::vector<int> infer_indices;
    infer_batch.reserve(gathered_data.size());
    infer_indices.reserve(gathered_data.size());

    for (const auto& data : gathered_data) {
        infer_batch.push_back(data.color);
        infer_indices.push_back(data.camera_index);
    }

    // 2. Inference
    auto results_batch = detector_->detect(infer_batch);

    // 3. Process to 3D Detections
    std::vector<Detection3D> all_detections_3d = process_detections(gathered_data, results_batch);
    
    if (debug_mode_) {
        RCLCPP_INFO(get_logger(), "Processed %zu 3D detections", all_detections_3d.size());
    }

    rclcpp::Time tracking_time = get_clock()->now();
    if (!gathered_data.empty()) {
        tracking_time = gathered_data[0].header.stamp;
    }

    // 4. Tracker & TF
    auto final_detections = tracker_->process(all_detections_3d, tf_buffer_, world_frame_, tracking_time);
    
    // 5. Publish
    if (!final_detections.empty()) {
        publish_detections(final_detections);
    }
    
    // 6. Annotated Images (Visualization)
    if (publish_annotated_) {
        visualizer_->publish_annotated_images(infer_batch, infer_indices, results_batch);
    }
}

std::vector<Detection3D> YoloDetectNode::process_detections(
    const std::vector<CameraData>& camera_data, 
    const std::vector<std::vector<Result2D>>& results_batch) 
{
    std::vector<Detection3D> all_detections_3d;
    
    for (size_t i=0; i<camera_data.size(); ++i) {
        const auto& data = camera_data[i];
        const auto& results = results_batch[i];
        
        if (debug_mode_) {
            RCLCPP_INFO(get_logger(), "Camera %d: %zu raw detections", data.camera_index, results.size());
        }

        if (!data.info) continue;

        std_msgs::msg::Header header = data.header;

        for (const auto& res : results) {
            Detection3D d3d = estimator_->compute_3d(res, data.depth, *data.info, header);
            all_detections_3d.push_back(d3d);
        }
    }
    return all_detections_3d;
}

void YoloDetectNode::publish_detections(const std::vector<Detection3D>& final_detections) {
    if (debug_mode_) {
        RCLCPP_INFO(get_logger(), "Publishing %zu final detections", final_detections.size());
    }

    vision_msgs::msg::Detection2DArray msg;
    msg.header.frame_id = world_frame_;
    msg.header.stamp = final_detections.front().header.stamp;
    if (msg.header.stamp.sec == 0 && msg.header.stamp.nanosec == 0) {
        msg.header.stamp = get_clock()->now();
    }

    for (const auto& det : final_detections) {
        vision_msgs::msg::Detection2D ros_det;
        ros_det.header = msg.header;
        
        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        
        if (det.result2d.class_id >= 0 && det.result2d.class_id < (int)class_names_.size()) {
             hyp.hypothesis.class_id = class_names_[det.result2d.class_id];
        } else {
             hyp.hypothesis.class_id = std::to_string(det.result2d.class_id);
        }
        
        hyp.hypothesis.score = det.result2d.score;
        hyp.pose.pose.position = det.position;
        hyp.pose.pose.orientation.w = 1.0;
        
        ros_det.results.push_back(hyp);
        ros_det.bbox.center.position.x = det.result2d.bbox.x + det.result2d.bbox.width/2.0;
        ros_det.bbox.center.position.y = det.result2d.bbox.y + det.result2d.bbox.height/2.0;
        ros_det.bbox.size_x = det.result2d.bbox.width;
        ros_det.bbox.size_y = det.result2d.bbox.height;
        
        msg.detections.push_back(ros_det);
    }

    detection_pub_->publish(msg);
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Sent %zu detections on topic '%s' in frame '%s'",
        msg.detections.size(), detection_pub_->get_topic_name(), msg.header.frame_id.c_str());
}

} // namespace yolo_ros
