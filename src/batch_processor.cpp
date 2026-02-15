#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include "yolo_ros/detector.hpp"
#include "yolo_ros/position_estimator.hpp"

namespace fs = std::filesystem;

class BatchProcessor : public rclcpp::Node {
public:
    BatchProcessor() : Node("batch_processor") {
        this->declare_parameter("input_folder", "");
        this->declare_parameter("model_path", "");
        this->declare_parameter("class_names", std::vector<std::string>({"coke"}));
        this->declare_parameter("class_radii_ids", std::vector<int64_t>({})); // Default empty
        this->declare_parameter("class_radii_values", std::vector<double>({0.03}));
        this->declare_parameter("confidence_threshold", 0.5);
        this->declare_parameter("nms_threshold", 0.4);
        this->declare_parameter("score_threshold", 0.5);

        this->declare_parameter("camera_fx", 320.0);
        this->declare_parameter("camera_fy", 320.0);
        this->declare_parameter("camera_cx", 320.0);
        this->declare_parameter("camera_cy", 180.0);
        this->declare_parameter("camera_height", 1.0); // meters
        this->declare_parameter("camera_pitch", 20.0); // degrees
        
        std::string input_folder = this->get_parameter("input_folder").as_string();
        std::string model_path = this->get_parameter("model_path").as_string();
        std::vector<std::string> class_names = this->get_parameter("class_names").as_string_array();
        
        std::vector<int64_t> radii_ids = this->get_parameter("class_radii_ids").as_integer_array();
        std::vector<double> radii_values = this->get_parameter("class_radii_values").as_double_array();
        
        double conf_thresh = this->get_parameter("confidence_threshold").as_double();
        double nms_thresh = this->get_parameter("nms_threshold").as_double();
        double score_thresh = this->get_parameter("score_threshold").as_double();

        double fx = this->get_parameter("camera_fx").as_double();
        double fy = this->get_parameter("camera_fy").as_double();
        double cx = this->get_parameter("camera_cx").as_double();
        double cy = this->get_parameter("camera_cy").as_double();
        
        double cam_h = this->get_parameter("camera_height").as_double();
        double cam_p = this->get_parameter("camera_pitch").as_double();

        class_names_ = class_names;

        if (input_folder.empty() || model_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Parameters input_folder and model_path are required. Provide them via config file or launch arguments.");
            return;
        }

        // Setup Detector
        detector_ = std::make_unique<yolo_ros::YoloOpenCVDetector>();
        yolo_ros::ModelConfig config;
        config.class_names = class_names;
        config.confidence_threshold = static_cast<float>(conf_thresh);
        config.score_threshold = static_cast<float>(score_thresh);
        config.nms_threshold = static_cast<float>(nms_thresh);

        if (!detector_->load(model_path, config)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load model from %s", model_path.c_str());
            return;
        }

        // Setup Position Estimator
        estimator_ = std::make_unique<yolo_ros::PositionEstimator>();
        std::map<int, float> radii_map;
        
        // If IDs are provided, use them. Otherwise assume 0..N
        if (!radii_ids.empty()) {
            for (size_t i = 0; i < radii_ids.size() && i < radii_values.size(); ++i) {
                radii_map[static_cast<int>(radii_ids[i])] = static_cast<float>(radii_values[i]);
            }
        } else {
            // Assume IDs match values order 0..N
            for (size_t i = 0; i < radii_values.size(); ++i) {
                radii_map[static_cast<int>(i)] = static_cast<float>(radii_values[i]);
            }
        }
        
        estimator_->set_class_radii(radii_map);
        estimator_->set_camera_parameters(static_cast<float>(cam_h), static_cast<float>(cam_p));

        // Setup Camera Info
        camera_info_.k[0] = fx;
        camera_info_.k[4] = fy;
        camera_info_.k[2] = cx;
        camera_info_.k[5] = cy;
        
        process_folder(input_folder);
    }

private:
    void process_folder(const std::string& folder_path) {
        if (!fs::exists(folder_path)) {
            RCLCPP_ERROR(this->get_logger(), "Folder %s does not exist.", folder_path.c_str());
            return;
        }

        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" || entry.path().extension() == ".jpeg") {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());

        RCLCPP_INFO(this->get_logger(), "Found %lu images in %s", files.size(), folder_path.c_str());

        // Create CSV file for batch results
        fs::path csv_path = fs::path(folder_path) / "detections.csv";
        std::ofstream csv_file(csv_path);
        if (csv_file.is_open()) {
            csv_file << "image_name,class,score,bbox_x,bbox_y,bbox_w,bbox_h,center_x,center_y,3d_x,3d_y,3d_z\n";
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to create CSV file %s", csv_path.c_str());
        }

        for (const auto& file : files) {
            process_image(file, csv_file);
        }
    }

    void process_image(const std::string& file_path, std::ofstream& csv_file) {
        cv::Mat image = cv::imread(file_path);
        if (image.empty()) {
            RCLCPP_WARN(this->get_logger(), "Failed to read image %s", file_path.c_str());
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Processing %s ...", fs::path(file_path).filename().c_str());

        // Detect
        auto detections = detector_->detect({image});
        if (detections.empty() || detections[0].empty()) {
            RCLCPP_INFO(this->get_logger(), "  No detections found.");
            return;
        }

        std_msgs::msg::Header header;
        header.frame_id = "camera_optical_frame";
        header.stamp = this->now();

        // Estimate
        for (const auto& det : detections[0]) {
            // Using empty depth image results in size-based estimation
            auto result3d = estimator_->compute_3d(det, cv::Mat(), camera_info_, header);
            
            std::string class_name = "Class " + std::to_string(det.class_id);
            if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names_.size())) {
                class_name = class_names_[det.class_id];
            }

            RCLCPP_INFO(this->get_logger(), 
                "  [%s] Score: %.2f | 2D: [x=%.1f, y=%.1f, w=%.1f, h=%.1f] | 3D: [x=%.3f, y=%.3f, z=%.3f]",
                class_name.c_str(), det.score,
                static_cast<float>(det.bbox.x), static_cast<float>(det.bbox.y), 
                static_cast<float>(det.bbox.width), static_cast<float>(det.bbox.height),
                result3d.position.x, result3d.position.y, result3d.position.z);

            float center_x = det.bbox.x + det.bbox.width / 2.0f;
            float center_y = det.bbox.y + det.bbox.height / 2.0f;

            if (csv_file.is_open()) {
                csv_file << fs::path(file_path).filename().string() << ","
                         << class_name << ","
                         << det.score << ","
                         << det.bbox.x << ","
                         << det.bbox.y << ","
                         << det.bbox.width << ","
                         << det.bbox.height << ","
                         << center_x << ","
                         << center_y << ","
                         << result3d.position.x << ","
                         << result3d.position.y << ","
                         << result3d.position.z << "\n";
            }

            // Draw bounding box
            cv::rectangle(image, det.bbox, cv::Scalar(0, 255, 0), 2);
            // Draw center point
            
            // Create label
            std::string label = "Class " + std::to_string(det.class_id);
            if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names_.size())) {
                label = class_names_[det.class_id];
            }
            label += ": " + std::to_string(det.score).substr(0, 4);
            
            // Draw label background
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(det.bbox.y, labelSize.height);
            cv::rectangle(image, cv::Point(det.bbox.x, top - labelSize.height),
                          cv::Point(det.bbox.x + labelSize.width, top + baseLine),
                          cv::Scalar(0, 255, 0), cv::FILLED);
            
            // Draw label text
            cv::putText(image, label, cv::Point(det.bbox.x, top),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // Save annotated image
        fs::path input_path(file_path);
        fs::path output_dir = input_path.parent_path() / "annotated";
        if (!fs::exists(output_dir)) {
            fs::create_directory(output_dir);
        }
        
        fs::path output_path = output_dir / input_path.filename();
        cv::imwrite(output_path.string(), image);
        RCLCPP_INFO(this->get_logger(), "  Saved annotated image to %s", output_path.c_str());
    }

    std::unique_ptr<yolo_ros::IDetector> detector_;
    std::unique_ptr<yolo_ros::PositionEstimator> estimator_;
    sensor_msgs::msg::CameraInfo camera_info_;
    std::vector<std::string> class_names_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BatchProcessor>();
    rclcpp::shutdown();
    return 0;
}
