#pragma once

#include <htrac/model/skeleton.hpp>
#include <htrac/pose/keypoint_detector.hpp>
#include <cvx/camera/camera.hpp>

class PoseFromKeyPoints {

public:
    PoseFromKeyPoints(const Skeleton &sk, float kp_conf_thresh = 0.8): skeleton_(sk), thresh_(kp_conf_thresh) {}

    bool estimate(const KeyPoints &kps, const cvx::PinholeCamera &cam, const cv::Mat &depth_img, Pose &pose) ;

private:

    using KeyPoints3 = std::map<std::string, Eigen::Vector3f> ;

    KeyPoints3 getKeyPoints3d(const KeyPoints &kpts, const cvx::PinholeCamera &cam, const cv::Mat &depth);

    static bool hasChest(const KeyPoints3 &kpts) ;
    static bool hasHips(const KeyPoints3 &kpts) ;
    static Eigen::Vector3f approxHipsFromChest(const KeyPoints3 &kpts) ;


    const Skeleton &skeleton_ ;
    float thresh_ = 0.8 ;
};
