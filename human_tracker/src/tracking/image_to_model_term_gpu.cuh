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

#include "energy_term.hpp"
#include "primitive_sdf_gpu.hpp"
#include <htrac/model/primitive_sdf.hpp>

#include <thrust/host_vector.h>


namespace impl {

class ImageToModelTermGPU {
public:

    ImageToModelTermGPU(const Skeleton &sk, const PrimitiveSDF &sdf,  float outlier_threshold):
        skeleton_(sk), sdf_(sdf), outlier_threshold_(outlier_threshold) {
        upload(sk, sdf) ;
        cublasCreate(&cublas_handle_);
    }

    ~ImageToModelTermGPU() {
        cublasDestroy(cublas_handle_) ;
    }

    void setImageCoords(const std::vector<Eigen::Vector3f> &icoords) {
        std::vector<Vec3> ic_gpu(icoords.size()) ;
        std::transform(icoords.begin(), icoords.end(), ic_gpu.begin(), [](const Eigen::Vector3f &src) { return Vec3(src) ; }) ;
        icoords_ = ic_gpu ;

    }

    double energy(const Pose &pose) ;
    std::pair<float, Eigen::VectorXf> energyGradient(const Pose &pose) ;

    void energy(const Pose &p, Eigen::VectorXf &e)  ;
    void jacobian(const Pose &p, Eigen::MatrixXf &jac)  ;

private:

    void upload(const Skeleton &sk, const PrimitiveSDF &psdf) ;
    void computeTransforms(const Pose &p) ;
    void updateDerivatives(const Pose &pose);
    void findBestPart(const thrust::device_vector<float> &samples, thrust::device_vector<float> &distances,
                        thrust::device_vector<size_t> &labels);

    void compute_gradient_sdf(const thrust::device_vector<float> &distances,
                              const thrust::device_vector<size_t> &labels,
                              const thrust::device_vector<Vec3> &gradients,
                              uint count,
                              thrust::device_vector<float> &grad);

    void compute_jacobian_sdf(const thrust::device_vector<float> &distances,
                                                   const thrust::device_vector<size_t> &labels,
                                                   const thrust::device_vector<Vec3> &gradients,
                                                   thrust::device_vector<float> &g);

    float outlier_threshold_ ;
    const Skeleton &skeleton_ ;

    size_t n_bones_, n_vars_, n_pose_bones_, n_global_params_ ;
    thrust::device_vector<Vec3> icoords_ ;
    thrust::device_vector<Matrix4x4> der_, gder_, trans_, itrans_ ;
    thrust::device_vector<size_t> pbone_idx_, pbone_dim_, pbone_offset_ ;
    thrust::device_vector<float> d_ones_ ;
    PrimitiveSDFGPU sdf_ ;
    cublasHandle_t cublas_handle_ ;

};
}
