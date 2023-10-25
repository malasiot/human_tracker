#pragma once

#include <opencv2/opencv.hpp>
#include <cvx/camera/camera.hpp>
#include <htrac/pose/dataset.hpp>

class DepthProvider {
public:
    virtual cv::Size imageSize() const = 0;
    virtual cvx::PinholeCamera camera() const = 0 ;
    virtual cv::Mat getDepth() const = 0;
};

class DatasetProvider: public DepthProvider {
public:

};
