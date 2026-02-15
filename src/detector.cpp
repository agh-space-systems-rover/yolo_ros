/**
 * @file detector.cpp
 * @author Mateusz Wójcik (mateuszwojcikv@gmail.com)
 * @brief Declaration of YOLO Object Detector using OpenCV DNN module
 * @version 0.1.0
 * @date 2026-02-01
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#include "yolo_ros/detector.hpp"
#include <fstream>
#include <iostream>
#include <rclcpp/rclcpp.hpp>

const int MAX_UINT8 = 255;


namespace yolo_ros {

/**
 * @brief Load the YOLO model

 * 
 * @param config model configuration parameters
 * @return true if the model was loaded successfully
 * @return false otherwise
 */
bool YoloOpenCVDetector::load(const std::string& model_path, const ModelConfig& config) {
    config_ = config;

    // Verify file existence
    std::ifstream f(model_path.c_str());
    if (!f.good()) {
        RCLCPP_ERROR(rclcpp::get_logger("yolo_opencv"), "YoloOpenCVDetector Error: Model file does not exist at: %s", model_path.c_str());
        return false;
    }
    f.close();

    try {
        RCLCPP_INFO(rclcpp::get_logger("yolo_opencv"), "YoloOpenCVDetector: Loading model from %s", model_path.c_str());
        
        net_ = cv::dnn::readNet(model_path);
        
        // Optimize for CUDA if available
#ifdef CV_CUDA
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        RCLCPP_INFO(rclcpp::get_logger("yolo_opencv"), "YoloOpenCVDetector: Using CUDA backend.");
#else
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        RCLCPP_INFO(rclcpp::get_logger("yolo_opencv"), "YoloOpenCVDetector: Using CPU backend.");
#endif

        RCLCPP_INFO(rclcpp::get_logger("yolo_opencv"), "YoloOpenCVDetector: Runtime OpenCV Version: %s", CV_VERSION);
        
        // Probe for layer info (optional, just for logging)
        out_names_ = net_.getUnconnectedOutLayersNames();
        // ... logging code ...
        
        return !net_.empty(); 
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("yolo_opencv"), "Failed to load YOLO model: %s", e.what());
        return false;
    }
}

/**
 * @brief Perform detection on a batch of images
 * 
 * @param images vector of input images
 * @return vector of detection results for each image
 */
std::vector<std::vector<Result2D>> YoloOpenCVDetector::detect(const std::vector<cv::Mat>& images) {
    if (images.empty()) return {};
    
    std::vector<std::vector<Result2D>> all_results;
    
    for (const auto& img : images) {
        std::vector<Result2D> results;
        if (img.empty()) {
            all_results.push_back(results);
            continue;
        }

        try {
            // Manual inference
            cv::Mat blob = preprocess(img);
            net_.setInput(blob);
            
            std::vector<cv::Mat> outs;
            net_.forward(outs, out_names_);
            
            if (!outs.empty()) {
                results = postprocess(outs[0], img.size());
            }

        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(rclcpp::get_logger("yolo_opencv"), "Detection error: %s", e.what());
        }

        all_results.push_back(results);
    }
    
    return all_results;
}

/**
 * @brief Preprocess the input image for YOLO model
 * 
 * @param img 
 * @return cv::Mat 
 */
cv::Mat YoloOpenCVDetector::preprocess(const cv::Mat& img) {
    cv::Mat blob;
    
    // Letterbox padding to preserve aspect ratio
    // Target is input_w_ x input_h_ (usually square 640x640)
    int w = img.cols;
    int h = img.rows;
    int max_dim = std::max(w, h);
    
    // Create a square image with max dimension
    cv::Mat square_img = cv::Mat::zeros(max_dim, max_dim, CV_8UC3);
    img.copyTo(square_img(cv::Rect(0, 0, w, h)));

    cv::Size input_size(input_w_, input_h_);
    
    try {
        cv::dnn::blobFromImage(square_img, blob, 1.0/255.0, input_size, cv::Scalar(), true, false);
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("yolo_opencv"), "Preprocess error: %s", e.what());
        throw;
    }

    return blob;
}

/**
 * @brief Postprocess the raw prediction from YOLO model
 * 
 * @param raw_prediction Raw output from the YOLO network
 * @param img_size Size of the original input image
 * @return std::vector<Result2D> 
 */
std::vector<Result2D> YoloOpenCVDetector::postprocess(const cv::Mat& raw_prediction, const cv::Size& img_size) {

    // 1. Transpose to ensure [Anchors, Channels] format
    cv::Mat prediction;
    try {
        prediction = sanitize_prediction_shape(raw_prediction);
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("yolo_opencv"), "Error in sanitize_prediction_shape: %s", e.what());
        RCLCPP_ERROR(rclcpp::get_logger("yolo_opencv"), "Raw dims: %d", raw_prediction.dims);
        return {};
    }
    // 2. Parse Detections
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    extract_detections(prediction, img_size, boxes, confidences, class_ids);
    // 3. Apply NMS and format results
    return apply_nms_and_format(boxes, confidences, class_ids);
}

/**
 * @brief Scale YOLO coordinates back to original image size
 * 
 * @param cx Center x coordinate in YOLO input scale
 * @param cy Center y coordinate in YOLO input scale
 * @param w Width in YOLO input scale
 * @param h Height in YOLO input scale
 * @param x_factor Scaling factor for x dimension
 * @param y_factor Scaling factor for y dimension
 * @return cv::Rect 
 */
cv::Rect YoloOpenCVDetector::scale_coords(float cx, float cy, float w, float h, float x_factor, float y_factor) {
    int left = int((cx - 0.5 * w) * x_factor);
    int top = int((cy - 0.5 * h) * y_factor);
    int width = int(w * x_factor);
    int height = int(h * y_factor);
    return cv::Rect(left, top, width, height);
}

/**
 * @brief Sanitize the shape of the raw prediction matrix to ensure consistent format
 * 
 * @param raw Raw output from the YOLO network
 * @return cv::Mat 
 */
cv::Mat YoloOpenCVDetector::sanitize_prediction_shape(const cv::Mat& raw) {
    cv::Mat dst;


    // Logic adapted from working test_onnx.cpp
    if (raw.dims == 3 && raw.size[0] == 1) {
        // [1, Channels, Anchors] or [1, Anchors, Channels]
        cv::Mat view(raw.size[1], raw.size[2], CV_32F, (void*)raw.data);
        if (view.rows < view.cols) {
            // [Channels, Anchors] -> Transpose -> [Anchors, Channels]
            cv::transpose(view, dst);
        } else {
            dst = view.clone();
        }
    } 
    else if (raw.dims == 2) {
        if (raw.rows < raw.cols) {
            // [Channels, Anchors] -> Transpose -> [Anchors, Channels]
            cv::transpose(raw, dst);
        } else {
            dst = raw.clone();
        }
    } else {
        // Already correct or unknown, just clone
        dst = raw.clone();
    }
    return dst;
}

/**
 * @brief Extract detections from the sanitized prediction matrix
 * 
 * @param prediction 
 * @param img_size 
 * @param boxes 
 * @param confidences 
 * @param class_ids 
 */
void YoloOpenCVDetector::extract_detections(const cv::Mat& prediction, const cv::Size& img_size, 
                        std::vector<cv::Rect>& boxes, 
                        std::vector<float>& confidences, 
                        std::vector<int>& class_ids) {
    
    int num_anchors = prediction.rows;
    // int num_channels = prediction.cols;
    int num_classes = prediction.cols - 4; // x, y, w, h are first 4

    // Scaling factors should use max dimension due to letterbox padding in preprocess
    float factor = 0.5f;

    for (int i = 0; i < num_anchors; i++) {
        const float* row_ptr = prediction.ptr<float>(i);
        const float* classes_scores = row_ptr + 4;
        
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(cv::Mat(1, num_classes, CV_32F, (void*)classes_scores), 0, &max_class_score, 0, &class_id_point);

        if (max_class_score > config_.confidence_threshold) {
            float w = row_ptr[0];
            float h = row_ptr[1];
            float cx = row_ptr[2];
            float cy = row_ptr[3];

            // // Debug first detection
            static bool first_log = true;
            if (first_log) {
                 RCLCPP_INFO(rclcpp::get_logger("yolo_opencv"), "Raw Box[0]: cx=%f cy=%f w=%f h=%f score=%f", cx, cy, w, h, max_class_score);
                 first_log = false;
            }

            

            // Assume x1, y1, x2, y2 format because w,h interpretation led to "too big" boxes
            // (If w is actually x2, it's a coordinate ~640, interpreted as width ~640)
            float x1 = cx;
            float y1 = cy;
            float x2 = w;
            float y2 = h;

            int left = int(x1 * factor);
            int top = int(y1 * factor);
            int width = int(h); // Note: Using h for width and w for height due to observed format
            int height = int(w);
            
            left = left - width;
            top = top - height;

            // Clip to image bounds
            left = std::max(0, left);
            top = std::max(0, top);
            if (left + width > img_size.width) width = img_size.width - left;
            if (top + height > img_size.height) height = img_size.height - top;

            boxes.push_back(cv::Rect(left, top, 2*width, 2*height));
            confidences.push_back((float)max_class_score);
            class_ids.push_back(class_id_point.x);
        }
    }
}

/**
 * @brief Apply Non-Maximum Suppression and format the final detection results
 * 
 * @param boxes 
 * @param confidences 
 * @param class_ids 
 * @return std::vector<Result2D> 
 */
std::vector<Result2D> YoloOpenCVDetector::apply_nms_and_format(const std::vector<cv::Rect>& boxes, 
                                            const std::vector<float>& confidences, 
                                            const std::vector<int>& class_ids) {
    std::vector<Result2D> results;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config_.confidence_threshold, config_.nms_threshold, indices);

    for (int idx : indices) {
        Result2D res;
        res.id = -1; 
        res.class_id = class_ids[idx];
        res.score = confidences[idx];
        res.bbox = boxes[idx];
        results.push_back(res);
    }
    return results;
}

} // namespace yolo_ros
