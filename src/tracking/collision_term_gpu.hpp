#pragma once

#include <htrac/model/skeleton.hpp>
#include "collision_data.hpp"

#include "collision_data.hpp"
#include "cuda_util.hpp"
#include "context_gpu.hpp"
#include <htrac/util/matrix4x4.hpp>

#include <thrust/device_vector.h>

class CollisionTermGPU {
public:
    CollisionTermGPU(ContextGPU &ctx, const Skeleton &sk, const CollisionData &cd);
    ~CollisionTermGPU() {}

    float energy(const Pose &pose) const ;
    void energy(Eigen::VectorXf &e) const ;

    std::pair<float, Eigen::VectorXf> energyGradient(const Pose &pose) const ;
    void jacobian(Eigen::MatrixXf &jac) const ;

    void norm(Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda) ;

    size_t nTerms() const { return ps_.size() ; }

private:
    void compute_errors(thrust::device_vector<float> &errors) const;
    void compute_gradient(thrust::device_vector<float> &grad) const;
    void compute_jacobian(thrust::device_vector<float> &jac) const;

    thrust::device_vector<Vec3> centers_ ;
    thrust::device_vector<float> radius_ ;
    thrust::device_vector<uint> bones_, ps_, pt_ ;

    const Skeleton &skeleton_ ;
    const CollisionData &cd_ ;
    ContextGPU &ctx_ ;
};
