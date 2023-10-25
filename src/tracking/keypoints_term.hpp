#pragma once

#include <htrac/model/skeleton.hpp>
#include <htrac/pose/keypoint_detector.hpp>

#include "context.hpp"

using KeyPoint3 = std::pair<Eigen::Vector3f, float> ;
using KeyPoints3 = std::map<std::string, KeyPoint3> ;

class KeyPointsTerm {
public:

    KeyPointsTerm(const Context &ctx, const KeyPoints3 &kpts): ctx_(ctx), kpts_(kpts) {}

    float energy(const Pose &pose) ;
    std::pair<float, Eigen::VectorXf> energyGradient(const Pose &pose) ;

    void energy(Eigen::VectorXf &e) const  ;
    void jacobian(Eigen::MatrixXf &jac) const ;
    void norm(Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda) const ;
    size_t nTerms() const { return kpts_.size() ; }

private:
    const KeyPoints3 &kpts_ ;
    const Context &ctx_ ;
};

