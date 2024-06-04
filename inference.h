#ifndef INFERENCE_H
#define INFERENCE_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <map>
// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <openvino/openvino.hpp>

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

class Inference
{
public:
    Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640}, const std::string &classesTxtFile = "", const bool &runWithCuda = true, const bool &runWithOpenvino=true);
    std::vector<Detection> runInference(const cv::Mat &input, double& infTime);
    std::vector<Detection> runOpenvinoInference(const cv::Mat &input, double& infTime);
    void UpdateDetectMap(int id, bool use);
    void loadClassesFromFile(std::string clsfile);
    void loadNetwork(std::string modelPath, bool runOnGpu, bool runOnOv);
    void UpdateInputShape(cv::Size inputsize);
    void UpdateThreshold(float confidThr, float scoreThr, float nmsThr);
    void loadClasses(std::vector<std::string> &cls);
private:
    cv::Mat formatToSquare(const cv::Mat &source);
    void GenerateColorMaps();
    std::string modelPath{};
    std::string classesPath{};

    // std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    std::vector<std::string> classes;
    std::map<int, bool> detectMap;
    cv::Size2f modelShape{};

    float modelConfidenceThreshold {0.25};
    float modelScoreThreshold      {0.45};
    float modelNMSThreshold        {0.50};

    bool letterBoxForSquare = true;

    std::vector<cv::Scalar> colorsmaps;
    cv::dnn::Net net;
    std::vector<double> layersTimes;

    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
};

#endif // INFERENCE_H
