#pragma once

#include <cvx/misc/variant.hpp>
#include <cvx/camera/camera.hpp>

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

class PoseEstimationAlgorithm ;
using Config = cvx::Variant ;

class PoseEstimator {
public:
    PoseEstimator(const Config &config) ;
    ~PoseEstimator() ;

    bool init() ;

    std::vector<Eigen::Vector3f> predict(const cv::Mat &depth, const cvx::PinholeCamera &cam, const cv::Rect &roi) ;

private:
    std::unique_ptr<PoseEstimationAlgorithm> alg_ ;

};


class PoseEstimationAlgorithm {
public:
    PoseEstimationAlgorithm() = default;

    virtual bool init() =0 ;
    virtual std::vector<Eigen::Vector3f> predict(const cv::Mat &depth, const cvx::PinholeCamera &cam, const cv::Rect &roi) = 0;

protected:

    std::tuple<cv::Mat, Eigen::Affine2f> cropScaleImage(const cv::Mat &img, const cv::Rect &box,
                                                        const cv::Size &sz, float maxz, int padding=4) ;
};
