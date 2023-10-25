#pragma once

#include <htrac/model/skeleton.hpp>
#include <htrac/pose/keypoint_distance_field.hpp>
#include <cvx/camera/camera.hpp>
#include <opencv2/opencv.hpp>

#include "context.hpp"


class KeyPoints2DTerm {
public:

    KeyPoints2DTerm(const Context &ctx, const cvx::PinholeCamera &cam, KeyPointsDistanceField *kdf): ctx_(ctx), cam_(cam), kdf_(kdf) {
        scale_ = sqrt(cam_.sz().width * cam_.sz().width +  cam_.sz().height * cam_.sz().height);
    }

    float energy(const Pose &pose) ;
    std::pair<float, Eigen::VectorXf> energyGradient(const Pose &pose) ;

    void energy(Eigen::VectorXf &e) const  ;
    void jacobian(Eigen::MatrixXf &jac) const ;
    void norm(Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda) const ;
    size_t nTerms() const { return kdf_ == nullptr ? 0 : kdf_->nBones() ; }

private:
    std::unique_ptr<KeyPointsDistanceField> kdf_ ;
    const Context &ctx_ ;
    const cvx::PinholeCamera &cam_ ;
    float scale_ ;
};
