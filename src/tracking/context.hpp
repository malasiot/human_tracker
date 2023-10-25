#pragma once

#include <Eigen/Core>
#include "energy_term.hpp"
struct Context {

    Context(const Skeleton &sk);

    void computeTransforms(const Pose &p) ;
    void computeDerivatives(const Pose &p) ;

    std::vector<Eigen::Matrix4f> bder_, gder_, trans_, itrans_, ioffset_ ;
    const Skeleton &skeleton_ ;

};
