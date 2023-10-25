#ifndef ENERGY_TERM_HPP
#define ENERGY_TERM_HPP

#include <htrac/model/skeleton.hpp>
#include <vector>
#include <memory>

class EnergyTerm {
public:
    static void compute_transform_derivatives(const Skeleton &skeleton, const Pose &pose,
                                       std::vector<Eigen::Matrix4f> &der, std::vector<Eigen::Matrix4f> &gder) ;
};

#endif
