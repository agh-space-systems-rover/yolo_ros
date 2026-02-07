/**
 * @file detector.hpp
 * @author Mateusz Wójcik (mateuszwojcikv@gmail.com)
 * @brief Definition of YOLO Object Detector using OpenCV DNN module
 * @version 0.1
 * @date 2026-02-01
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Uncomment if GPU avaible
// #define CV_CUDA

namespace yolo_ros {

    /**
     * @brief Structure to hold 2D detection results
     * 
     */
    struct Result2D {
        int id; 
        int class_id;
        float score;
        cv::Rect bbox;
    };

    /**
     * @brief Structure to hold model configuration parameters
     * 
     */
    struct ModelConfig {
        float confidence_threshold = 0.5f;
        float score_threshold = 0.5f;
        float nms_threshold = 0.4f;
        std::vector<std::string> class_names;
    };

    /**
     * @brief Interface for object detectors
     * 
     */
    class IDetector {
    public:
        using Ptr = std::unique_ptr<IDetector>;
        virtual ~IDetector() = default;

        /**
         * @brief Load the model
         * 
         * @param model_path path to the model file
         * @param config model configuration parameters
         * @return true if the model was loaded successfully
         * @return false otherwise
         */
        virtual bool load(const std::string& model_path, const ModelConfig& config) = 0;

        /**
         * @brief Perform detection on a batch of images
         * 
         * @param images vector of input images
         * @return vector of detection results for each image
         */
        virtual std::vector<std::vector<Result2D>> detect(const std::vector<cv::Mat>& images) = 0;
    };

    /**
     * @brief YOLO Object Detector using OpenCV DNN module
     * 
     */
    class YoloOpenCVDetector : public IDetector {
    public:
        
        /**
         * @brief Load the YOLO model
         * 
         * @param model_path path to the model file
         * @param config model configuration parameters
         * @return true if the model was loaded successfully
         * @return false otherwise
         */
        bool load(const std::string& model_path, const ModelConfig& config) override;
        
        /**
         * @brief Perform detection on a batch of images
         * 
         * @param images vector of input images
         * @return vector of detection results for each image
         */
        std::vector<std::vector<Result2D>> detect(const std::vector<cv::Mat>& images) override;

    private:
        cv::dnn::Net net_;
        ModelConfig config_;
        std::vector<std::string> out_names_;
        std::string input_name_;
        
        // Input dimensions
        int input_w_ = 640;
        int input_h_ = 640;

        cv::Mat preprocess(const cv::Mat& img);
        std::vector<Result2D> postprocess(const cv::Mat& prediction, const cv::Size& img_size);
        
        static cv::Rect scale_coords(float cx, float cy, float w, float h, float x_factor, float y_factor);
        static cv::Mat sanitize_prediction_shape(const cv::Mat& raw);

        void extract_detections(const cv::Mat& prediction, const cv::Size& img_size, 
                                std::vector<cv::Rect>& boxes, 
                                std::vector<float>& confidences, 
                                std::vector<int>& class_ids);
                                
        std::vector<Result2D> apply_nms_and_format(const std::vector<cv::Rect>& boxes, 
                                                   const std::vector<float>& confidences, 
                                                   const std::vector<int>& class_ids);
    };

} // namespace yolo_ros
