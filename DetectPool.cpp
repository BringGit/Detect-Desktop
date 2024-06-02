#include "DetectPool.h"

DetectPool::DetectPool(Config cfg) : config(cfg)
{

	Inf = new Inference(config.onnxPath, config.inputSize, config.classfile, config.runOnGPU);
    // if (config.useCamera)
 //        OpenCamera(0, false);
	
}

bool DetectPool::OpenCamera(cv::Mat &frame, int cid, bool isfile, std::string filename)
{
    if (isfile)
    {
        Cap.open(filename);
        if (!Cap.isOpened())
            return false;
        Cap.grab();
        Cap.retrieve(frame);
    }
    else
    {
        Cap.open(cid);
        if (!Cap.isOpened())
            return false;
        Cap.grab();
        Cap.retrieve(frame);
    }

	return true;
}

void DetectPool::CloseCamera()
{
    Cap.release();
}

bool DetectPool::SendFrame(cv::Mat &frame, double infTime, int& numDet)
{
    if(Cap.grab())
    {

        // mtx.lock();
        Cap.retrieve(frame);
        // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        // mtx.unlock();
        frame = GetOutPut(frame, infTime, numDet);
        cv::waitKey(1);
        return true;
    }
    else
    {
        return false;
    }
}

cv::Mat DetectPool::GetOutPut(cv::Mat frame, double& infTime, int& numDet)
{
    cv::Mat outimg = frame.clone();


    std::vector<Detection> output = Inf->runInference(outimg, infTime);

    int detections = output.size();
    numDet = detections;
    std::cout << "Number of detections:" << detections << std::endl;
    for (int i = 0; i < detections; ++i)
    {
        Detection detection = output[i];

        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;

        // Detection box
        cv::rectangle(outimg, box, color, 2);
        // Detection box text
        std::string classnameString = "";
        std::string scoreString = "";
        if (config.displayClassName)
            classnameString = detection.className + ' ';
        if (config.displayScore)
            scoreString = std::to_string(detection.confidence).substr(0, 4);

        if (config.displayClassName || config.displayScore)
        {
            cv::Size textSize = cv::getTextSize(classnameString + scoreString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
            cv::rectangle(outimg, textBox, color, cv::FILLED);
            cv::putText(outimg, classnameString + scoreString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
    }
    // Inference ends here...s
    return outimg;
}

unsigned char* DetectPool::CV2QT(cv::Mat img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    return (unsigned char*)(img.data);
}

void DetectPool::UpdateMap(int id, bool use){
    Inf->UpdateDetectMap(id, use);
}

void DetectPool::UpdateConfig(struct Config cfg)
{
    if ((!cfg.onnxPath.empty() && config.onnxPath != cfg.onnxPath) || (config.runOnGPU != cfg.runOnGPU))
    {
        config.onnxPath = cfg.onnxPath;
        config.runOnGPU = cfg.runOnGPU;
        Inf->loadOnnxNetwork(config.onnxPath, config.runOnGPU);
        if ((cfg.inputSize.width !=0 && cfg.inputSize.height !=0) && (cfg.inputSize != config.inputSize))
        {
            config.inputSize = cfg.inputSize;
            Inf->UpdateInputShape(config.inputSize);
        }
    }
    if (!cfg.classfile.empty() && config.classfile != cfg.classfile)
    {
        config.classfile = cfg.classfile;
        Inf->loadClassesFromFile(config.classfile);
    }

    config.useCamera = cfg.useCamera;
    config.ConfidenceThreshold = cfg.ConfidenceThreshold;
    config.ScoreThreshold = cfg.ScoreThreshold;
    config.NMSThreshold = cfg.NMSThreshold;
    Inf->UpdateThreshold(cfg.ConfidenceThreshold, cfg.ScoreThreshold, cfg.NMSThreshold);

    config.displayClassName = cfg.displayClassName;
    config.displayScore = cfg.displayScore;
}

Config DetectPool::GetConfig()
{
    return config;
}

void DetectPool::GetClasses(std::vector<std::string>& classes)
{
    Inf->loadClasses(classes);
}
