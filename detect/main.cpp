//葛文扬 遥感2401

#include <opencv2/opencv.hpp>
#include "armor_detector.h"

int main() {
    VideoCapture video; //VC类对象化
    video.open("test1.avi");
    

    ArmorDetector detector;
    cv::Mat frame;

    while (true) {
        video>> frame;
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }

        detector.processFrame(frame);
        detector.displayResults(frame);

        if (cv::waitKey(30) >= 0) break;
    }

    return 0;
}