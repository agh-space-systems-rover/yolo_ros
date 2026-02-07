#include <stdbool.h>
#include "yolo_ros/yolo_detect_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <filesystem>

namespace yolo_ros {

/**
 * @brief Construct a new Yolo Detect Node object
 * 
 * @param options Node options
 */
YoloDetectNode::YoloDetectNode(const rclcpp::NodeOptions & options)
    : rclcpp_lifecycle::LifecycleNode("yolo_detect", options)
{
    // Params
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
}

/**
 * @brief Lifecycle on_configure callback
 * 
 * @param 
 * @return CallbackReturn 
 */
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

    // Init Logic Components
    detector_ = std::make_unique<YoloOpenCVDetector>();
    ModelConfig cfg;
    cfg.confidence_threshold = confidence_threshold_;
    cfg.class_names = class_names_;
    
    if (!detector_->load(model_path_, cfg)) {
        RCLCPP_ERROR(get_logger(), "Failed to load model: %s", model_path_.c_str());
        return CallbackReturn::FAILURE;
    }

    estimator_ = std::make_unique<PositionEstimator>();
    std::map<int, float> radii_map;
    
    // Map radii to class IDs ensuring bounds
    size_t count = std::min(class_names_.size(), class_radii_param_.size());
    for (size_t i = 0; i < count; ++i) {
        radii_map[i] = class_radii_param_[i];
    }
    estimator_->set_class_radii(radii_map);

    tracker_ = std::make_unique<DetectionTracker>(merge_radius_, temporal_window_, temporal_threshold_);

    // TF
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Publisher
    detection_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);

    return CallbackReturn::SUCCESS;
}


/**
 * @brief Lifecycle on_activate callback
 * 
 * @param state 
 * @return CallbackReturn 
 */
YoloDetectNode::CallbackReturn YoloDetectNode::on_activate(const rclcpp_lifecycle::State & state) {
    LifecycleNode::on_activate(state); // Activate publishers

    // Setup Subscribers & Sync
    cameras_.clear();
    RCLCPP_INFO(get_logger(), "Activating YOLO Node with %d cameras", num_cameras_);
    // RCLCPP_INFO(get_logger(), "Cameras ids: %s", rgbd_ids_.empty() ? "None" : std::accumulate(rgbd_ids_.begin(), rgbd_ids_.end(), std::string(),
    //     [](const std::string& a, const std::string& b) { return a + (a.empty() ? "" : ", ") + b; }));
    for (int i=0; i<num_cameras_; ++i) {
        auto cam = std::make_shared<CameraContext>();
        cam->index = i;
        
        std::string color_base;
        std::string depth_base;
        std::string info_base;
        std::string annotated_base;

        if (!rgbd_ids_.empty()) {
            std::string id = rgbd_ids_[i];
            // Assuming IDs like "d455_front"
            // Construct absolute topics based on user request
            // Color: /<id>/color/image_raw
            // Depth: /<id>/depth/image_raw
            // Info:  /<id>/color/camera_info
            
            // Ensure ID doesn't have leading slash for consistency if we prepend /
            // But usually ID is just "d455_front". 
            // We want "/d455_front/..."
            
            // If the ID passed is already absolute path-like (starts with /), handle that?
            // User launch file: "d455_front d455_back" -> clean strings.
            
            std::string prefix = "/" + id;
            if (id.front() == '/') prefix = id; // if already starts with /
            
            color_base = prefix + "/color/image_raw";
            depth_base = prefix + "/depth/image_raw";
            info_base = prefix + "/color/camera_info";
            annotated_base = prefix + "/yolo_annotated";
        } else {
            RCLCPP_ERROR(get_logger(), "Camera %d: No RGBD ID provided in 'rgbd_ids' parameter. Cannot construct topics.", i);
        }

        RCLCPP_INFO(get_logger(),"Setting up camera %d: color topic '%s', depth topic '%s', info topic '%s'", 
            i, color_base.c_str(), depth_base.c_str(), info_base.c_str());
        // QoS
        rmw_qos_profile_t custom_qos = rmw_qos_profile_default;

        // --- Color Subscription (Raw or Compressed) ---
        if (color_transport_ == "compressed") {
             // Use our wrapper
             auto sub = std::make_shared<CompressedSubscriberWrapper>(this, color_base + "/compressed", custom_qos);
             cam->color_sub = std::shared_ptr<message_filters::SimpleFilter<sensor_msgs::msg::Image>>(sub, sub.get());
             cam->color_sub_handle = sub; 
        } else {
             // Standard Raw
             auto sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>>(this, color_base, custom_qos);
             cam->color_sub = sub; 
             cam->color_sub_handle = sub;
        }

        // --- Depth Subscription (Raw or Compressed) ---
        if (depth_transport_ == "compressed") {
             // Usually depth compressed is "compressedDepth" png, cv_bridge handles closest match
             // User requested /d455_front/depth/image_raw/compressedDepth
             auto sub = std::make_shared<CompressedSubscriberWrapper>(this, depth_base + "/compressedDepth", custom_qos);
             cam->depth_sub = std::shared_ptr<message_filters::SimpleFilter<sensor_msgs::msg::Image>>(sub, sub.get());
             cam->depth_sub_handle = sub; 
        } else {
             auto sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>>(this, depth_base, custom_qos);
             cam->depth_sub = sub;
             cam->depth_sub_handle = sub;
        }

        // --- Info Subscription (Always Raw) ---
        cam->info_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo, rclcpp_lifecycle::LifecycleNode>>(this, info_base, custom_qos);
        
        // --- Publishers (Annotated) ---
        if (publish_annotated_) {
             // For simplicity in Lifecycle node, we use create_publisher<Image> (Raw only)
             // Implementing compressed publisher manually for annotated requires more code.
             // We stick to RAW output for annotated for C++ version MVP.
             cam->annotated_pub = this->create_publisher<sensor_msgs::msg::Image>(annotated_base, 1);
        }

        // Init Sync
        cam->sync = std::make_shared<Sync1>(SyncPolicy1(10), *cam->color_sub, *cam->depth_sub, *cam->info_sub);
        cam->sync->registerCallback(
            std::bind(&YoloDetectNode::on_camera_data, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, i)
        );
        
        cameras_.push_back(cam);
    }

    // Rate Timer
    // 10 Hz default
    double rate = 1.0; 
    get_parameter_or("rate", rate, 10.0);
    timer_ = create_wall_timer(std::chrono::milliseconds((int)(1000.0/rate)), 
        std::bind(&YoloDetectNode::timer_callback, this));
    
    RCLCPP_INFO(get_logger(), "YOLO Node Activated");
    return CallbackReturn::SUCCESS;
}

/**
 * @brief Lifecycle on_deactivate callback
 * 
 * @param state 
 * @return CallbackReturn 
 */
YoloDetectNode::CallbackReturn YoloDetectNode::on_deactivate(const rclcpp_lifecycle::State & state) {
    timer_.reset();
    cameras_.clear(); // Destroys subs
    LifecycleNode::on_deactivate(state);
    return CallbackReturn::SUCCESS;
}

/**
 * @brief Lifecycle on_cleanup callback
 * 
 * @param state 
 * @return CallbackReturn 
 */
YoloDetectNode::CallbackReturn YoloDetectNode::on_cleanup(const rclcpp_lifecycle::State &) {
    detector_.reset();
    estimator_.reset();
    tracker_.reset();
    tf_buffer_.reset();
    tf_listener_.reset();
    detection_pub_.reset();
    return CallbackReturn::SUCCESS;
}

/**
 * @brief Shutdown callback
 * 
 * @param state 
 * @return YoloDetectNode::CallbackReturn 
 */
YoloDetectNode::CallbackReturn YoloDetectNode::on_shutdown(const rclcpp_lifecycle::State & state) {
    return CallbackReturn::SUCCESS;
}

/**
 * @brief A callback for when camera data is received
 * 
 * @param color Color image from the camera
 * @param depth Depth image from the camera
 * @param info Camera info message
 * @param camera_index Index of the camera
 */
void YoloDetectNode::on_camera_data(
    const sensor_msgs::msg::Image::ConstSharedPtr& color,
    const sensor_msgs::msg::Image::ConstSharedPtr& depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info,
    int camera_index) 
{
    if (camera_index >= (int)cameras_.size()) return;
    auto& cam = cameras_[camera_index];
    cam->last_color = color;
    cam->last_depth = depth;
    cam->last_info = info;
    cam->has_new_data = true;

    if (debug_mode_) {
        RCLCPP_INFO(get_logger(), "Received data from camera %d", camera_index);
    }
}

/**
 * @brief Timer callback to process data and publish detections
 * 
 */
void YoloDetectNode::timer_callback() {
    if (debug_mode_) {
        RCLCPP_INFO(get_logger(), "Timer callback start");
    }
    // 1. Gather Images
    std::vector<cv::Mat> infer_batch;
    std::vector<int> infer_indices;
    
    
    if (!gather_images(infer_batch, infer_indices)) return;
    RCLCPP_INFO(get_logger(), "Gathering images for inference, current batch size: %zu", infer_batch.size());
    if (!infer_batch.empty()) {
        RCLCPP_INFO(get_logger(), "First image size: %dx%d, type: %d", 
            infer_batch[0].cols, infer_batch[0].rows, infer_batch[0].type());
    }


    // 2. Inference
    RCLCPP_INFO(get_logger(), "Running inference on batch of %zu images", infer_batch[0].empty() ? 0 : infer_batch.size());
    auto results_batch = detector_->detect(infer_batch);
    RCLCPP_INFO(get_logger(), "Publishing annotated images, received %zu inference results", results_batch.size());

    // 3. Process to 3D Detections
    std::vector<Detection3D> all_detections_3d = process_detections(infer_batch, infer_indices, results_batch);
    RCLCPP_INFO(get_logger(), "Processed %zu 3D detections", all_detections_3d.size());

    // 4. Tracker & TF
    auto final_detections = tracker_->process(all_detections_3d, tf_buffer_, world_frame_, get_clock()->now());
    RCLCPP_INFO(get_logger(), "Processed %zu final detections", final_detections.size());
    // 5. Publish
    if (!final_detections.empty()) {
        publish_detections(final_detections);
    }
    
    // 6. Annotated Images (Visualization)
    if (publish_annotated_) {
        RCLCPP_INFO(get_logger(), "Publishing annotated images, received %zu inference results", results_batch.size());
        publish_annotated_images(infer_batch, infer_indices, results_batch);
    }
}

/**
 * @brief Gather images from all cameras that have new data
 * 
 * @param images Vector to fill with gathered images
 * @param indices Vector to fill with corresponding camera indices
 * @return true if at least one image was gathered
 * @return false otherwise
 */
bool YoloDetectNode::gather_images(std::vector<cv::Mat>& images, std::vector<int>& indices) {
    for (auto& cam : cameras_) {
        if (!cam->has_new_data || !cam->last_color) continue;
        
        // Convert to CV
        try {
           cv::Mat img = cv_bridge::toCvCopy(cam->last_color, "bgr8")->image;
           images.push_back(img);
           indices.push_back(cam->index);
           cam->has_new_data = false; // Reset flag

           if (debug_mode_) {
               RCLCPP_INFO(get_logger(), "Gathered image from camera %d", cam->index);
           }
        } catch (cv_bridge::Exception& e) {
           RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    return !images.empty();
}

/**
 * @brief Process detections from the detector and estimate their 3D positions
 * 
 * @param images Vector of input images
 * @param indices Vector of corresponding camera indices
 * @param results_batch Vector of detection results for each image
 * @return std::vector<Detection3D> 
*/
std::vector<Detection3D> YoloDetectNode::process_detections(const std::vector<cv::Mat>& images, const std::vector<int>& indices, const std::vector<std::vector<Result2D>>& results_batch) {
    std::vector<Detection3D> all_detections_3d;
    
    for (size_t i=0; i<images.size(); ++i) {
        int cam_idx = indices[i];
        auto& cam_ctx = cameras_[cam_idx];
        auto& results = results_batch[i];
        
        if (debug_mode_) {
            RCLCPP_INFO(get_logger(), "Camera %d: %zu raw detections", cam_idx, results.size());
        }

        cv::Mat depth_img;
        if (cam_ctx->last_depth) {
            try {
                // Assuming raw 16UC1 or 32FC1
                 depth_img = cv_bridge::toCvCopy(cam_ctx->last_depth, sensor_msgs::image_encodings::TYPE_16UC1)->image;
            } catch (...) {
                // Try float
                 try {
                     depth_img = cv_bridge::toCvCopy(cam_ctx->last_depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
                 } catch(...) {}
            }
        }
        
        // Convert header
        std_msgs::msg::Header header = cam_ctx->last_color->header; // Frame ID
        
        for (const auto& res : results) {
            Detection3D d3d = estimator_->compute_3d(res, depth_img, *cam_ctx->last_info, header);
            all_detections_3d.push_back(d3d);
        }
    }
    return all_detections_3d;
}

/**
 * @brief Publish detections to ROS topic
 * 
 * @param detections Vector of 3D detections to publish
 */
void YoloDetectNode::publish_detections(const std::vector<Detection3D>& final_detections) {
    if (debug_mode_) {
        RCLCPP_INFO(get_logger(), "Publishing %zu final detections", final_detections.size());
    }

    vision_msgs::msg::Detection2DArray msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = world_frame_;
    
    for (const auto& d : final_detections) {
        vision_msgs::msg::Detection2D ros_det;
        ros_det.header = d.header; // Or world frame? Usually detections array is in world frame.
        
        // Format vision_msgs
        ros_det.bbox.center.position.x = d.result2d.bbox.x + d.result2d.bbox.width/2.0;
        ros_det.bbox.center.position.y = d.result2d.bbox.y + d.result2d.bbox.height/2.0;
        ros_det.bbox.size_x = d.result2d.bbox.width;
        ros_det.bbox.size_y = d.result2d.bbox.height;
        
        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = std::to_string(d.result2d.class_id); // we used int, msg uses string often?
        if (d.result2d.class_id < (int)class_names_.size()) {
            hyp.hypothesis.class_id = class_names_[d.result2d.class_id];
        } else {
            hyp.hypothesis.class_id = std::to_string(d.result2d.class_id);
        }
        
        hyp.hypothesis.score = d.result2d.score;
        hyp.pose.pose.position = d.position; // The 3D position
        hyp.pose.pose.orientation.w = 1.0;
        
        ros_det.results.push_back(hyp);
        msg.detections.push_back(ros_det);
    }
    
    detection_pub_->publish(msg);
}

/**
 * @brief Publish annotated images with detection results
 * 
 * @param images Vector of input images
 * @param indices Vector of corresponding camera indices
 * @param results_batch Vector of detection results for each image
 */
void YoloDetectNode::publish_annotated_images(const std::vector<cv::Mat>& images, const std::vector<int>& indices, const std::vector<std::vector<Result2D>>& results_batch) {
    for (size_t i=0; i<images.size(); ++i) {
        int cam_idx = indices[i];
        auto& cam_ctx = cameras_[cam_idx];
        
        // Draw
        // We need a writable copy
        cv::Mat annotated_img = images[i].clone();
        const auto& results = results_batch[i];
        
        for (const auto& r : results) {
            // Draw Box
            cv::rectangle(annotated_img, r.bbox, cv::Scalar(0, 255, 0), 2);
            
            // Draw Label
            std::string label = std::to_string(r.class_id);
            if (r.class_id < (int)class_names_.size()) {
                label = class_names_[r.class_id];
            }
            label += " " + std::to_string((int)(r.score * 100)) + "%";
            
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(annotated_img, cv::Point(r.bbox.x, r.bbox.y - labelSize.height),
                          cv::Point(r.bbox.x + labelSize.width, r.bbox.y + baseLine),
                          cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(annotated_img, label, cv::Point(r.bbox.x, r.bbox.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
        
        // Publish
        if (cam_ctx->annotated_pub && cam_ctx->annotated_pub->get_subscription_count() > 0) {
             sensor_msgs::msg::Image::SharedPtr out_msg = cv_bridge::CvImage(
                 cam_ctx->last_color->header, "bgr8", annotated_img).toImageMsg();
             cam_ctx->annotated_pub->publish(*out_msg);
        }
    }
}

} // namespace yolo_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(yolo_ros::YoloDetectNode)
