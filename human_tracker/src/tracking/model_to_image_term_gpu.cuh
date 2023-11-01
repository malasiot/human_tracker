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

#include <htrac/util/matrix4x4.hpp>
#include <htrac/util/vector3.hpp>
#include <htrac/util/vector2.hpp>

namespace impl {

#define MAX_BONES_PER_VERTEX 4

struct VertexBoneDataGPU {
    int id_[MAX_BONES_PER_VERTEX];
    float weight_[MAX_BONES_PER_VERTEX];

    __device__ __host__
    VertexBoneDataGPU(const SkinnedMesh::VertexBoneData &src) {
        for( size_t i=0 ; i<MAX_BONES_PER_VERTEX; i++ ) {
            id_[i] = src.id_[i] ;
            weight_[i] = src.weight_[i];
        }
    }
};

class ModelToImageTermGPU {
public:
    ModelToImageTermGPU(const SkinnedMesh &mesh, const cvx::PinholeCamera &cam):
        mesh_(mesh), cam_(cam), skeleton_(mesh.skeleton_) {
        cublasCreate(&cublas_handle_);
        upload(mesh) ;
    }


    ~ModelToImageTermGPU() {
        cublasDestroy(cublas_handle_) ;
        cudaDestroyTextureObject(tex_) ;
        cudaFreeArray(dt_) ;
    }

    void setDistanceTransform(const cv::Mat &dt, const cv::Mat &grad) ;

    double energy(const Pose &pose) ;
    std::pair<float, Eigen::VectorXf> energyGradient(const Pose &pose) ;

    void energy(const Pose &p, Eigen::VectorXf &e)  ;
    void jacobian(const Pose &p, Eigen::MatrixXf &jac)  ;

private:

    void upload(const SkinnedMesh &sk) ;

    void sampleVertices(const Pose &p, thrust::device_vector<Vec3> &mpos,
                        thrust::device_vector<float> &errors, thrust::device_vector<Vec2> &gradients) ;


    void updateDerivatives(const Pose &pose);

    void compute_gradient_sdf(const thrust::device_vector<Vec3> &mpos,
                              const thrust::device_vector<float> &distances,
                              const thrust::device_vector<Vec2> &gradients,
                              thrust::device_vector<float> &grad);


    void compute_jacobian_sdf(const thrust::device_vector<Vec3> &mpos, const thrust::device_vector<float> &distances,
                              const thrust::device_vector<Vec2> &gradients, thrust::device_vector<float> &grad);

    const Skeleton &skeleton_ ;

    size_t n_bones_, n_vars_, n_pose_bones_, n_global_params_, n_pts_ ;
    size_t width_, height_ ;
    thrust::device_vector<Vec3> mesh_pos_ ;
    thrust::device_vector<VertexBoneDataGPU> mesh_vbd_ ;

    thrust::device_vector<Matrix4x4> der_, gder_, ioffset_ ;

    thrust::device_vector<size_t> pbone_idx_, pbone_dim_, pbone_offset_ ;
    thrust::device_vector<float> d_ones_ ;

    cudaArray *dt_ = nullptr ;
    cudaTextureObject_t tex_ ;

    thrust::device_vector<float> dtv_ ;
    thrust::device_vector<Vec2> dtg_ ;

    cublasHandle_t cublas_handle_ ;
    const SkinnedMesh &mesh_ ;
    const cvx::PinholeCamera &cam_ ;
};

}
