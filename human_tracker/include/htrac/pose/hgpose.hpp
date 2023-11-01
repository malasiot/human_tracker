#pragma once

#include <htrac/pose/pose_estimation.hpp>

#include <Eigen/Geometry>
#include <cvx/camera/camera.hpp>

#include <torch/script.h>

class PoseStackedHourGlass: public PoseEstimationAlgorithm {
public:
    struct Parameters {
        Parameters();
        int lft_wnd_size_ = 5 ;
        std::string data_folder_ ;
    };


    PoseStackedHourGlass(const Parameters &params = Parameters{}) ;
    PoseStackedHourGlass(const Config &cfg) ;

    std::vector<Eigen::Vector3f> predict(const cv::Mat &im, const cvx::PinholeCamera &cam, const cv::Rect &roi) override ;

    struct HeatmapData {
        std::vector<cv::Mat> heatmaps_, heatmap_grad_x_, heatmap_grad_y_ ;
        Eigen::Affine2f tr_ ;
    };

    HeatmapData predictHeatmaps(const cv::Mat &im, const cv::Rect &roi) ;

    const Parameters &params() const { return params_ ; }

    bool init() override ;
private:

    using Predictions2d = std::map<std::string, std::pair<Eigen::Vector2f, float>> ;
    using Predictions3d = std::map<std::string, std::pair<Eigen::Vector3f, float>> ;

    torch::Tensor runNetwork(const cv::Mat &img);
    Predictions2d predictJoints(const cv::Mat &im) ;
    Predictions3d liftPredictions(const Predictions2d &p, const cv::Mat &depth, const cvx::PinholeCamera &cam) ;
    std::vector<Eigen::Vector3f> jointRegression(const Predictions3d &p) ;


    void drawSkeleton(cv::Mat &im, Predictions2d &joints) ;

    torch::jit::script::Module module_, regressor_;
    Parameters params_ ;

};
