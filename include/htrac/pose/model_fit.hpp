#pragma once

#include <cvx/misc/variant.hpp>
#include <cvx/camera/camera.hpp>

#include <htrac/model/pose.hpp>
#include <opencv2/opencv.hpp>

using Config = cvx::Variant ;
class HumanModelFitImpl ;
class KeyPointsDistanceField ;
using Plane = Eigen::Hyperplane<float, 3> ;

class HumanModelFit {
public:
    struct Parameters {
        Parameters() = default ;
        Parameters(const Config &) ;

        float lambda_im_ = 1.0e3 ; // factor for closest point term
        float lambda_mi_ = 0.005 ; // factor for background penalty term
        float lambda_col_ = 1.0e2 ; // factor for collision term
        float lambda_jl_ = 1.0e2 ; // factor for joint limits violation
        float lambda_kp_ = 1.0e3 ;
        float outlier_threshold_ = 0.5f ; // maximum distance of closest point to consider as inlier
        uint sample_step_ = 2 ;
    };

    HumanModelFit(const Parameters &p) ;
    ~HumanModelFit() ;

    const Skeleton &skeleton() const ;

    void setClippingPlanes(const std::vector<Plane> &planes) ;

    Pose fit(const std::vector<Eigen::Vector3f> &cloud, const cv::Mat &mask, const cvx::PinholeCamera &cam,
             KeyPointsDistanceField *kpts, const Pose &orig) ;
    Pose fit(const cv::Mat &im, const cv::Mat &mask, const cvx::PinholeCamera &cam,
             KeyPointsDistanceField *kpts, const Pose &orig) ;
private:
    std::unique_ptr<HumanModelFitImpl> impl_ ;
    std::vector<Plane> clipping_planes_ ;
};
