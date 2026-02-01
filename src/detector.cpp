/**
 * @file detector.cpp
 * @author Mateusz Wójcik (mateuszwojcikv@gmail.com)
 * @brief Declaration of YOLO Object Detector using OpenCV DNN module
 * @version 0.1
 * @date 2026-02-01
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#include "yolo_ros/detector.hpp"
#include <fstream>
#include <iostream>

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const int MAX_UINT8 = 255;


namespace yolo_ros {

/**
 * @brief Load the YOLO model
 * 
 * @param model_path path to the model file
 * @param classes_path path to the classes file
 * @param config model configuration parameters
 * @return true if the model was loaded successfully
 * @return false otherwise
 */
bool YoloOpenCVDetector::load(const std::string& model_path, const std::string& classes_path, const ModelConfig& config) {
    config_ = config;
    try {
        net_ = cv::dnn::readNet(model_path);
        
        // Optimize for CUDA if available
#ifdef CV_CUDA
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "YoloOpenCVDetector: Using CUDA backend." << std::endl;
#else
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "YoloOpenCVDetector: Using CPU backend." << std::endl;
#endif
        
        out_names_ = net_.getUnconnectedOutLayersNames();
        
        // If classes are not provided in config, try to load from file
        if (config_.class_names.empty() && !classes_path.empty()) {
             std::ifstream ifs(classes_path.c_str());
             std::string line;
             while (std::getline(ifs, line)) {
                 if (!line.empty()) config_.class_names.push_back(line);
             }
        }
        
        return !net_.empty();
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to load YOLO model: " << e.what() << std::endl;
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

        // 1. Preprocess
        cv::Mat blob = preprocess(img);
        
        // 2. Inference
        net_.setInput(blob);
        std::vector<cv::Mat> outs;
        try {
            net_.forward(outs, out_names_);
        } catch (const cv::Exception& e) {
            std::cerr << "YOLO Inference error: " << e.what() << std::endl;
            all_results.push_back(results);
            continue;
        }

        if (outs.empty()) {
            all_results.push_back(results);
            continue;
        }

        // 3. Postprocess
        results = postprocess(outs[0], img.size());
        all_results.push_back(results);
    }
    
    return all_results;
}

cv::Mat YoloOpenCVDetector::preprocess(const cv::Mat& img) {
    cv::Mat blob;
    cv::Size input_size(INPUT_WIDTH, INPUT_HEIGHT);
    // YOLOv11/v8 standard: 1/255 scaling, swapRB=true, crop=false
    cv::dnn::blobFromImage(img, blob, 1.0/MAX_UINT8, input_size, cv::Scalar(), true, false);
    return blob;
}

std::vector<Result2D> YoloOpenCVDetector::postprocess(const cv::Mat& raw_prediction, const cv::Size& img_size) {
    // 1. Transpose to ensure [Anchors, Channels] format
    cv::Mat prediction = sanitize_prediction_shape(raw_prediction);

    // 2. Parse Detections
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    extract_detections(prediction, img_size, boxes, confidences, class_ids);

    // 3. Apply NMS and format results
    return apply_nms_and_format(boxes, confidences, class_ids);
}

cv::Rect YoloOpenCVDetector::scale_coords(float cx, float cy, float w, float h, float x_factor, float y_factor) {
    int left = int((cx - 0.5 * w) * x_factor);
    int top = int((cy - 0.5 * h) * y_factor);
    int width = int(w * x_factor);
    int height = int(h * y_factor);
    return cv::Rect(left, top, width, height);
}

cv::Mat YoloOpenCVDetector::sanitize_prediction_shape(const cv::Mat& raw) {
    cv::Mat dst;
    if (raw.dims == 3 && raw.size[0] == 1) {
        // [1, Channels, Anchors] -> [Anchors, Channels]
        cv::Mat view(raw.size[1], raw.size[2], CV_32F, (void*)raw.data);
        cv::transpose(view, dst); 
    } else if (raw.dims == 2 && raw.rows < raw.cols) {
        // [Channels, Anchors] -> [Anchors, Channels]
        cv::transpose(raw, dst);
    } else {
        dst = raw;
    }
    return dst;
}

void YoloOpenCVDetector::extract_detections(const cv::Mat& prediction, const cv::Size& img_size, 
                        std::vector<cv::Rect>& boxes, 
                        std::vector<float>& confidences, 
                        std::vector<int>& class_ids) {
    
    int num_anchors = prediction.rows;
    int num_channels = prediction.cols;
    int num_classes = num_channels - 4; // x, y, w, h are first 4

    // Scaling factors to map network input (640x640) back to input image size
    float x_factor = (float)img_size.width / INPUT_WIDTH;
    float y_factor = (float)img_size.height / INPUT_HEIGHT;

    for (int i = 0; i < num_anchors; i++) {
        const float* row_ptr = prediction.ptr<float>(i);
        const float* classes_scores = row_ptr + 4;
        
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(cv::Mat(1, num_classes, CV_32F, (void*)classes_scores), 0, &max_class_score, 0, &class_id_point);

        if (max_class_score > config_.confidence_threshold) {
            float cx = row_ptr[0];
            float cy = row_ptr[1];
            float w = row_ptr[2];
            float h = row_ptr[3];

            boxes.push_back(scale_coords(cx, cy, w, h, x_factor, y_factor));
            confidences.push_back((float)max_class_score);
            class_ids.push_back(class_id_point.x);
        }
    }
}

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
