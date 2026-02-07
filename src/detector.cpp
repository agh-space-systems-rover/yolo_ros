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
        std::cerr << "YoloOpenCVDetector Error: Model file does not exist at: " << model_path << std::endl;
        return false;
    }
    f.close();

    try {
        std::cout << "YoloOpenCVDetector: Loading model from " << model_path << std::endl;
        net_ = cv::dnn::readNet(model_path);
        
        // Optimize for CUDA if available
#ifdef CV_CUDA
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "YoloOpenCVDetector: Using CUDA backend." << std::endl;
#else
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        // Workaround attempts for OpenCV 4.5.4 assertion failures:
        // winograd/fusion options are not available in this OpenCV version's C++ API.
        // We rely on standard CPU backend invocation.
        
        std::cout << "YoloOpenCVDetector: Using CPU backend." << std::endl;
#endif
        
        out_names_ = net_.getUnconnectedOutLayersNames();

        // Try to detect input size from the model (works for ONNX with fixed shapes)
        std::vector<cv::dnn::MatShape> inLayerShapes, outLayerShapes;
        // Layer 0 is usually the input layer
        net_.getLayerShapes(cv::dnn::MatShape(), 0, inLayerShapes, outLayerShapes);
        if (!inLayerShapes.empty() && !inLayerShapes[0].empty()) {
            std::cout << "YoloOpenCVDetector: Model input shape: [";
            for (int s : inLayerShapes[0]) std::cout << s << " ";
            std::cout << "]" << std::endl;

            // Usually [Batch, Channels, Height, Width]
            if (inLayerShapes[0].size() == 4) {
                input_w_ = inLayerShapes[0][3];
                input_h_ = inLayerShapes[0][2];
                std::cout << "YoloOpenCVDetector: Detected input size: " << input_w_ << "x" << input_h_ << std::endl;
            }
        } else {
             std::cout << "YoloOpenCVDetector: Using default input size: " << input_w_ << "x" << input_h_ << std::endl;
        }
        
        std::cout << "YoloOpenCVDetector: Output layers: ";
        for (const auto& name : out_names_) std::cout << name << " ";
        std::cout << std::endl;

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
            // Standard forward pass, should work now with Opset 12 model
            if (!out_names_.empty()) {
                net_.forward(outs, out_names_);
            } else {
                cv::Mat out = net_.forward();
                outs.push_back(out);
            }
        } catch (const cv::Exception& e) {
            std::cerr << "YOLO Inference error: OpenCV exception during forward pass." << std::endl;
            std::cerr << "Details: " << e.what() << std::endl;
            std::cerr << "Blob shape: " << blob.size[0] << "x" << blob.size[1] << "x" << blob.size[2] << "x" << blob.size[3] << std::endl;
            // Return empty results for this frame but keep running
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
    // YOLOv11/v8 standard: 1/255 scaling, swapRB=true, crop=false
    // blobFromImage handles the resize from max_dim to input_size
    // cv::dnn::blobFromImage(square_img, blob, 1.0/MAX_UINT8, input_size, cv::Scalar(), true, false);
    
    // Debug logging for preprocessing
    try {
        cv::dnn::blobFromImage(square_img, blob, 1.0/MAX_UINT8, input_size, cv::Scalar(), true, false);
    } catch (const cv::Exception& e) {
        std::cerr << "Preprocess error in blobFromImage: " << e.what() << std::endl;
        std::cerr << "Input img size: " << square_img.size() << " channels: " << square_img.channels() << std::endl;
        throw;
    }

    // Ensure blob is continuous - critical for some OpenCV versions
    if (!blob.isContinuous()) {
        blob = blob.clone();
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
    // Debug logging
    // std::cout << "Postprocess raw shape: dims=" << raw_prediction.dims << " size=[";
    // for(int i=0; i<raw_prediction.dims; i++) std::cout << raw_prediction.size[i] << (i==raw_prediction.dims-1?"":", ");
    // std::cout << "]" << std::endl;

    // 1. Transpose to ensure [Anchors, Channels] format
    cv::Mat prediction;
    try {
        prediction = sanitize_prediction_shape(raw_prediction);
    } catch (const cv::Exception& e) {
        std::cerr << "Error in sanitize_prediction_shape: " << e.what() << std::endl;
        std::cerr << "Raw dims: " << raw_prediction.dims << std::endl;
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
    
    // Debug info
    // std::cout << "Sanitize input: dims=" << raw.dims << " [";
    // for(int i=0; i<raw.dims; ++i) std::cout << raw.size[i] << " ";
    // std::cout << "]" << std::endl;

    // Logic adapted from working test_onnx.cpp
    if (raw.dims == 3 && raw.size[0] == 1) {
        // [1, Channels, Anchors] -> [Channels, Anchors]
        // Create a 2D view of the data
        cv::Mat view(raw.size[1], raw.size[2], CV_32F, (void*)raw.data);
        // Transpose to [Anchors, Channels]
        cv::transpose(view, dst);
    } 
    else if (raw.dims == 2 && raw.rows < raw.cols) {
        // [Channels, Anchors] -> Transpose -> [Anchors, Channels]
        cv::transpose(raw, dst);
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
    int num_channels = prediction.cols;
    int num_classes = num_channels - 4; // x, y, w, h are first 4

    // Scaling factors. We padded the image to be square (max_dim x max_dim).
    // The network output corresponds to that square image resized to input_w_ x input_h_.
    int max_dim = std::max(img_size.width, img_size.height);
    float x_factor = (float)max_dim / input_w_;
    float y_factor = (float)max_dim / input_h_;

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
