#include "armor_detector.h"

ArmorDetector::ArmorDetector() {
    // 初始化代码
}

ArmorDetector::~ArmorDetector() {
    // 清理代码
}

void ArmorDetector::processFrame(const cv::Mat& frame) {
    detectLampBars(frame);
    detectNumbers(frame);
}

void ArmorDetector::displayResults(const cv::Mat& frame) {
    // 显示结果
    cv::imshow("Armor Detection", frame);
    cv::waitKey(30);
}

void ArmorDetector::detectLampBars(const cv::Mat& frame) {
    Mat channels[3], binary, Gaussian, dilatee,gray;
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Rect boundRect;
    RotatedRect box;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point2f> boxPts(4);
      cvtColor(frame, gray, COLOR_BGR2GRAY);
        threshold(gray, binary, 220, 255, 0);//二值化
        GaussianBlur(binary, Gaussian, Size(5, 5), 0);//滤波
        dilate(Gaussian, dilatee, element);
        // dilate(Gaussian, dilate, element, Point(-1, -1));//膨胀，把滤波得到的细灯条变宽
        findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//轮廓检测
        vector<ArmorDetector> lightInfos;//创建一个灯条类的动态数组
        for (int i = 0; i < contours.size(); i++) {
            // 求轮廓面积
            double area = contourArea(contours[i]);
            // 去除较小轮廓&fitEllipse的限制条件
            if (area < 5|| contours[i].size() <= 1)
                continue;//相当于就是把这段轮廓去除掉
            // 用椭圆拟合区域得到外接矩形（特殊的处理方式：因为灯条是椭圆型的，所以用椭圆去拟合轮廓，再直接获取旋转外接矩形即可）
            RotatedRect Light_Rec = fitEllipse(contours[i]);
 
            // 长宽比和轮廓面积比限制（由于要考虑灯条的远近都被识别到，所以只需要看比例即可）
            if (Light_Rec.size.width / Light_Rec.size.height > 4)
                continue;
            lightInfos.push_back(ArmorDetector(Light_Rec));
        }

        for (size_t i = 0; i < lightInfos.size(); i++) {
            for (size_t j = i + 1; (j < lightInfos.size()); j++) {
                ArmorDetector& leftLight = lightInfos[i];
                ArmorDetector& rightLight = lightInfos[j];
                float angleGap_ = abs(leftLight.angle - rightLight.angle);
                //由于灯条长度会因为远近而受到影响，所以按照比值去匹配灯条
                float LenGap_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
                float dis = pow(pow((leftLight.center.x - rightLight.center.x), 2) + pow((leftLight.center.y - rightLight.center.y), 2), 0.5);
                //均长
                float meanLen = (leftLight.length + rightLight.length) / 2;
                float lengap_ratio = abs(leftLight.length - rightLight.length) / meanLen;
                float yGap = abs(leftLight.center.y - rightLight.center.y);
                float yGap_ratio = yGap / meanLen;
                float xGap = abs(leftLight.center.x - rightLight.center.x);
                float xGap_ratio = xGap / meanLen;
                float ratio = dis / meanLen;
                //匹配不通过的条件
                if (angleGap_ > 5 ||
                    LenGap_ratio > 1.0 ||
                    lengap_ratio > 0.8 ||
                    yGap_ratio > 1.5 ||
                    xGap_ratio > 2.2 ||
                    xGap_ratio < 0.8 ||
                    ratio > 3 ||
                    ratio < 0.8) {
                    continue;
                }
                //绘制矩形
                Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y) / 2);
                RotatedRect rect = RotatedRect(center, Size(dis, meanLen), (leftLight.angle + rightLight.angle) / 2);
                Point2f vertices[4];
                rect.points(vertices);
                for (int i = 0; i < 4; i++) {
                    line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2.2);
                }
            }
        }

        
}




void  ArmorDetector::detectNumbers ( const cv::Mat& frame)
{
Mat thr, gray, con;
   
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    threshold(gray, thr, 200, 255, THRESH_BINARY_INV); // Threshold to create input
    thr.copyTo(con);


    // Read stored sample and label for training
    Mat sample;
    Mat response, tmp;
    FileStorage Data("TrainingData.yml", FileStorage::READ); // Read traing data to a Mat
    Data["data"] >> sample;
    Data.release();

    FileStorage Label("LabelData.yml", FileStorage::READ); // Read label data to a Mat
    Label["label"] >> response;
    Label.release();

    Ptr<ml::KNearest>  knn(ml::KNearest::create());
    knn->train(sample, ml::ROW_SAMPLE,response); // Train with sample and responses
    cout << "Training compleated.....!!" << endl;

    vector< vector <Point> > contours; // Vector for storing contour
    vector< Vec4i > hierarchy;

    // 在图片中寻找轮廓
    findContours(con, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    Mat dst(frame.rows, frame.cols, CV_8UC3, Scalar::all(0));

    for (int i = 0; i< contours.size(); i = hierarchy[i][0]) // 逐个识别轮廓内数字
    {
        // cout << "begin analyze" << endl;

        Rect r = boundingRect(contours[i]);
       
        
        rectangle(dst, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 0, 255), 2, 8, 0);
        Mat ROI = thr(r);
        Mat tmp1, tmp2;
        resize(ROI, tmp1, Size(10, 10), 0, 0, INTER_LINEAR);
        tmp1.convertTo(tmp2, CV_32FC1);
        Mat response;

        float p = knn->findNearest(tmp2.reshape(1, 1), 1, response); // 识别数字
        
        
        char name[4];
        sprintf(name, "%d", (int)p);
        // 将识别到的数字标识到输出图片上
        putText(dst, name, Point(r.x, r.y + r.height), 0, 1, Scalar(0, 255, 0), 2, 8);

    }
    
    

       
        

 
    
    

}








  



 
    





 
   
 
    