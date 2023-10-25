#pragma once

#include "energy_term.hpp"
#include "collision_data.hpp"
#include <htrac/model/sdf_model.hpp>


class CollisionTerm {
public:

    CollisionTerm(const Skeleton &sk, const CollisionData &cd, float gamma);

    float energy(const Pose &pose) ;
    std::pair<float, Eigen::VectorXf> energyGradient(const Pose &pose) ;

private:

    std::vector<std::pair<size_t, size_t>> pairs_ ;

    const Skeleton &skeleton_ ;
    const CollisionData &cd_ ;
    float gamma_ ;

};

