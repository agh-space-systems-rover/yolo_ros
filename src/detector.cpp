#include "yolo_ros/detector.hpp"
#include <fstream>
#include <iostream>

namespace yolo_ros {

bool YoloOpenCVDetector::load(const std::string& model_path, const std::string& classes_path, const ModelConfig& config) {
    config_ = config;
    try {
        net_ = cv::dnn::readNet(model_path);
        // Optimize for CUDA if available
#ifdef CV_CUDA
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#else
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif
        
        out_names_ = net_.getUnconnectedOutLayersNames();
        
        // If classes are not provided in config, try to load from file
        if (!config_.class_names.empty()) {
            // Used provided names
        } else if (!classes_path.empty()) {
             std::ifstream ifs(classes_path.c_str());
             std::string line;
             while (std::getline(ifs, line)) {
                 config_.class_names.push_back(line);
             }
        }
        
        return !net_.empty();
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to load Copy of YOLO model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::vector<Result2D>> YoloOpenCVDetector::detect(const std::vector<cv::Mat>& images) {
    if (images.empty()) return {};
    
    // For simplicity in OpenCV DNN, we usually process images one by one or as a blob.
    // Batch processing is possible with blobFromImages.
    
    std::vector<std::vector<Result2D>> all_results;
    
    for (const auto& img : images) {
        std::vector<Result2D> results;
        if (img.empty()) {
            all_results.push_back(results);
            continue;
        }

        cv::Mat blob;
        // YOLOv8/v5/v11 usually expects [0, 1] scaling, swapRB=true, crop=false
        // Size often 640x640. We assume the user provides a model that fits.
        // Or we should parameterize input size.
        cv::dnn::blobFromImage(img, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        
        net_.setInput(blob);
        
        std::vector<cv::Mat> outs;
        net_.forward(outs, out_names_);
        
        // Post-processing logic for YOLO
        // This heavily depends on the specific YOLO version (output layout).
        // Standard YOLOv8 output: [1, 84, 8400] -> [batch, classes+4, anchors]
        // 4 coords (cx, cy, w, h) + 80 class scores.

        // Assuming YOLOv8 format for this example.
        // float* data = (float*)outs[0].data;
        // int rows = outs[0].size[1]; // 84
        // int dimensions = outs[0].size[2]; // 8400 (anchors)

        // Need to transpose for easier iterating if formatted as above.
        // But let's check generic "ultralytics export onnx" format.
        // Usually it IS [1, 4+nc, N].
        
        // Handling this generically requires robust parsing or just assuming one standard.
        // For MVP we assume standard YOLOv8 ONNX export.
        
        // Simplified parsing loop (Psuedo-implementation)
        // Implementation details omitted for brevity, would need specific layout check
        
        // TODO: Implement full YOLOv8 parsing here.
        all_results.push_back(results);
    }
    
    return all_results;
}

} // namespace yolo_ros
