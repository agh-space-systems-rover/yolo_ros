#include "yolo_ros/camera_manager.hpp"

namespace yolo_ros {

const int MAX_QUEUE_SIZE = 3;

CompressedSubscriberWrapper::CompressedSubscriberWrapper(
    rclcpp_lifecycle::LifecycleNode* node, const std::string& topic, 
    const rmw_qos_profile_t& qos, const std::string& target_encoding)
{
    sub_ = node->create_subscription<sensor_msgs::msg::CompressedImage>(
        topic, 
        rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos), qos),
        [this, target_encoding](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
            cv_bridge::CvImagePtr cv_ptr;
            try {
                // Determine encoding. If empty, default to BGR8 (common for color compressed).
                // For depth, we usually want "passthrough" or specific type if known.
                std::string encoding = target_encoding.empty() ? sensor_msgs::image_encodings::BGR8 : target_encoding;
                if (encoding == "passthrough") encoding = ""; // specific behavior of cv_bridge? no, toCvCopy(msg, "") is safe.
                
                cv_ptr = cv_bridge::toCvCopy(msg, encoding);
            } catch (cv_bridge::Exception& e) {
                return;
            }
            this->signalMessage(cv_ptr->toImageMsg());
        }
    );
}

CameraManager::CameraManager(rclcpp_lifecycle::LifecycleNode* node) : node_(node) {}

CameraManager::~CameraManager() {
    deactivate();
}

void CameraManager::configure(
    int num_cameras,
    const std::vector<std::string>& rgbd_ids,
    const std::string& color_transport,
    const std::string& depth_transport,
    bool subscribe_depth
) {
    num_cameras_ = num_cameras;
    rgbd_ids_ = rgbd_ids;
    color_transport_ = color_transport;
    depth_transport_ = depth_transport;
    subscribe_depth_ = subscribe_depth;
}

void CameraManager::activate() {
    cameras_.clear();
    RCLCPP_INFO(node_->get_logger(), "Activating Camera Manager with %d cameras", num_cameras_);

    for (int i=0; i<num_cameras_; ++i) {
        auto cam = std::make_shared<CameraContext>();
        cam->index = i;
        
        std::string color_base;
        std::string depth_base;
        std::string info_base;

        if (!rgbd_ids_.empty()) {
            std::string id = rgbd_ids_[i];
            std::string prefix = "/" + id;
            if (id.front() == '/') prefix = id; 
            
            color_base = prefix + "/color/image_raw";
            depth_base = prefix + "/depth/image_raw";
            info_base = prefix + "/color/camera_info";
        } else {
             RCLCPP_ERROR(node_->get_logger(), "Camera %d: No ID provided.", i);
             continue; 
        }

        RCLCPP_INFO(node_->get_logger(),"Setting up camera %d: color topic '%s'", i, color_base.c_str());
        
        rmw_qos_profile_t custom_qos = rmw_qos_profile_default;

        // --- Color Subscription ---
        if (color_transport_ == "compressed") {
             // Default BGR8
             auto sub = std::make_shared<CompressedSubscriberWrapper>(node_, color_base + "/compressed", custom_qos, "bgr8");
             cam->color_sub = std::shared_ptr<message_filters::SimpleFilter<sensor_msgs::msg::Image>>(sub, sub.get());
             cam->color_sub_handle = sub; 
        } else {
             auto sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>>(node_, color_base, custom_qos);
             cam->color_sub = sub; 
             cam->color_sub_handle = sub;
        }

        // --- Depth Subscription ---
        // Always subscribe to satisfy sync policy (approximate time needs inputs)
        // If 'subscribe_depth' is false, we might receive nulls or just ignore in processing, 
        // but for sync to trigger, we need messages on all inputs unless we use optional policy.
        if (depth_transport_ == "compressed") {
            auto sub = std::make_shared<CompressedSubscriberWrapper>(node_, depth_base + "/compressedDepth", custom_qos, "passthrough");
            cam->depth_sub = std::shared_ptr<message_filters::SimpleFilter<sensor_msgs::msg::Image>>(sub, sub.get());
            cam->depth_sub_handle = sub; 
        } else {
            auto sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>>(node_, depth_base, custom_qos);
            cam->depth_sub = sub;
            cam->depth_sub_handle = sub;
        }


        // --- Info Subscription ---
        cam->info_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo, rclcpp_lifecycle::LifecycleNode>>(node_, info_base, custom_qos);
        
        // Init Sync
        cam->sync = std::make_shared<Sync1>(SyncPolicy1(MAX_QUEUE_SIZE), *cam->color_sub, *cam->depth_sub, *cam->info_sub);
        cam->sync->registerCallback(
            std::bind(&CameraManager::on_camera_data, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, i)
        );
        
        cameras_.push_back(cam);
    }
}

void CameraManager::deactivate() {
    cameras_.clear();
}

void CameraManager::on_camera_data(
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
}

std::vector<CameraData> CameraManager::gather_images() {
    std::vector<CameraData> gathered;
    for (auto& cam : cameras_) {
        if (!cam->has_new_data || !cam->last_color) continue;
        
        CameraData data;
        data.camera_index = cam->index;
        data.header = cam->last_color->header;
        data.info = cam->last_info;
        
        // Convert Color
        try {
            data.color = cv_bridge::toCvCopy(cam->last_color, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
            continue;
        }

        // Convert Depth
        if (cam->last_depth && subscribe_depth_) {
            try {
                 // Try to convert to 32FC1 or 16UC1
                 // If already correct encoding, toCvCopy handles it.
                 // If "passthrough" was used in subscription, it has correct encoding from msg.
                 if (cam->last_depth->encoding == "16UC1" || cam->last_depth->encoding == "mono16") {
                     data.depth = cv_bridge::toCvCopy(cam->last_depth, sensor_msgs::image_encodings::TYPE_16UC1)->image;
                 } else if (cam->last_depth->encoding == "32FC1") {
                     data.depth = cv_bridge::toCvCopy(cam->last_depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
                 } else {
                     // Try autodetect or force float
                     data.depth = cv_bridge::toCvCopy(cam->last_depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
                 }
            } catch (cv_bridge::Exception& e) {
                 RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                     "Failed to convert depth image for camera %d: %s", cam->index, e.what());
                 // Keep depth empty on conversion failure
            }
        }
        
        cam->has_new_data = false;
        gathered.push_back(data);
    }
    return gathered;
}

} // namespace yolo_ros
