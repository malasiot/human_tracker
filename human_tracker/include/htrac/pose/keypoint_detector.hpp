#pragma once

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <map>

using KeyPoint = std::pair<Eigen::Vector2f, float> ;
using KeyPoints = std::map<std::string, KeyPoint> ;

class KeyPointDetector {
public:
    virtual void init() = 0 ;
    virtual KeyPoints findKeyPoints(const cv::Mat &img) = 0 ;

    void drawKeyPoints(cv::Mat &rgb, const KeyPoints &kp, float thresh = 0.5, float thickness = 3.f, float radius = 4.f);
};

class KeyPointDetectorException: public std::runtime_error {
public:
    KeyPointDetectorException(const std::string &what): std::runtime_error(what) {}
};

