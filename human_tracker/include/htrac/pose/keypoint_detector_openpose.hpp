#pragma once

#include <htrac/pose/keypoint_detector.hpp>

class OpenPoseAsync ;

class KeyPointDetectorOpenPose: public KeyPointDetector {
public:
    struct Parameters {
        std::string data_folder_ ;
    } ;

    KeyPointDetectorOpenPose(const Parameters &params = {});
    ~KeyPointDetectorOpenPose() ;

    void init() override ;
    KeyPoints findKeyPoints(const cv::Mat &img) override ;

private:

    Parameters params_ ;
    std::unique_ptr<OpenPoseAsync> impl_ ;
};
