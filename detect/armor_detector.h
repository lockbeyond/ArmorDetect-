#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H

#include "stdio.h"
#include<iostream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/dnn.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv::ml;

using namespace std;
using namespace cv;

class ArmorDetector {
public:
    
    ~ArmorDetector();
float width, length, angle, area;
      cv::Point2f center;

    ArmorDetector() ;
    //让得到的灯条套上一个旋转矩形，以方便之后对角度这个特殊因素作为匹配标准
    ArmorDetector(const cv::RotatedRect& light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
    void processFrame(const cv::Mat& frame);
    void displayResults(const cv::Mat& frame);

private:
    void detectLampBars(const cv::Mat& frame);
    void detectNumbers(const cv::Mat& frame);
};

#endif // ARMOR_DETECTOR_H