#pragma once
// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <thread>
#include <mutex>
// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "inference.h"

struct Config {
    std::string onnxPath="";
    cv::Size inputSize=cv::Size(0, 0);
	std::string classfile = "";
	bool useCamera = false;
	bool runOnGPU = false;
    bool runOnOpenvino = false;
    float ConfidenceThreshold = 0.25;
    float ScoreThreshold = 0.45;
    float NMSThreshold = 0.50;
    bool displayClassName = true;
    bool displayScore = true;
};

class DetectPool
{
public:
    DetectPool(struct Config cfg);
	~DetectPool() {};
    bool OpenCamera(cv::Mat &frame, int cid=0, bool isfile=true, std::string filename="");
    void CloseCamera();
    bool SendFrame(cv::Mat &frame, double infTime, int& numDet);
    cv::Mat GetOutPut(cv::Mat frame, double& infTime, int& numDet);
    unsigned char* CV2QT(cv::Mat img);
    void UpdateMap(int id, bool use);
    void UpdateConfig(struct Config cfg);
    void GetClasses(std::vector<std::string>& classes);
    Config GetConfig();
private:
	Inference *Inf;
	cv::VideoCapture Cap;
    Config config;
    std::mutex mtx;

	
};

