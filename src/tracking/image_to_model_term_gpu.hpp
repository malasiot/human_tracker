#pragma once

#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include <cublas_v2.h>
#include <cublas_api.h>

#include "cuda_util.hpp"
#include "context_gpu.hpp"
#include "energy_term.hpp"
#include "primitive_sdf_gpu.hpp"
#include <htrac/model/primitive_sdf.hpp>

#include <cvx/camera/camera.hpp>

#include <thrust/host_vector.h>

class ImageToModelTermGPU {
public:

    ImageToModelTermGPU(ContextGPU &ctx, const Skeleton &sk, const PrimitiveSDF &sdf, float outlier_threshold):
        ctx_(ctx), skeleton_(sk), sdf_(sdf), outlier_threshold_(outlier_threshold) {
        cudaStreamCreate(&stream_);
    }

    ~ImageToModelTermGPU() {
    }

    double energy(const Pose &pose) const ;
    std::pair<float, Eigen::VectorXf> energyGradient(const Pose &pose) const ;

    void energy(Eigen::VectorXf &e) const  ;
    void jacobian(Eigen::MatrixXf &jac) const ;
    void norm(Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda) ;
    size_t nTerms() const { return ctx_.n_pts_ ; }

private:

    void findBestPart(const thrust::device_vector<float> &samples, thrust::device_vector<float> &distances,
                        thrust::device_vector<size_t> &labels) const ;

    void compute_gradient_sdf(const thrust::device_vector<float> &distances,
                              const thrust::device_vector<size_t> &labels,
                              const thrust::device_vector<Vec3> &gradients,
                              uint count,
                              thrust::device_vector<float> &grad) const ;

    void compute_jacobian_sdf(const thrust::device_vector<float> &distances,
                                                   const thrust::device_vector<size_t> &labels,
                                                   const thrust::device_vector<Vec3> &gradients,
                                                   thrust::device_vector<float> &g) const ;

    float outlier_threshold_ ;
    const Skeleton &skeleton_ ;

    uint n_pts_ ;
    PrimitiveSDFGPU sdf_ ;
    ContextGPU &ctx_ ;
    cudaStream_t stream_ ;

};
