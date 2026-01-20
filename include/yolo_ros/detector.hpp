#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace yolo_ros {

    struct Result2D {
        int id; // unique tracking id (not class id) assigned later? No, usually just detection index
        int class_id;
        float score;
        cv::Rect bbox;
        // Optionally mask for segmentation but we focus on det for now
    };

    struct ModelConfig {
        float confidence_threshold = 0.5f;
        float score_threshold = 0.5f;
        float nms_threshold = 0.4f;
        std::vector<std::string> class_names;
    };

    class IDetector {
    public:
        using Ptr = std::unique_ptr<IDetector>;
        virtual ~IDetector() = default;

        virtual bool load(const std::string& model_path, const std::string& classes_path, const ModelConfig& config) = 0;
        virtual std::vector<std::vector<Result2D>> detect(const std::vector<cv::Mat>& images) = 0;
    };

    class YoloOpenCVDetector : public IDetector {
    public:
        bool load(const std::string& model_path, const std::string& classes_path, const ModelConfig& config) override;
        std::vector<std::vector<Result2D>> detect(const std::vector<cv::Mat>& images) override;

    private:
        cv::dnn::Net net_;
        ModelConfig config_;
        std::vector<std::string> out_names_;
    };

} // namespace yolo_ros
