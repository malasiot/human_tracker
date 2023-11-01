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

#include <htrac/util/matrix4x4.hpp>
#include <htrac/util/vector3.hpp>
#include <htrac/util/vector2.hpp>
#include <htrac/model/skinned_mesh.hpp>

#include <cvx/camera/camera.hpp>

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
    ModelToImageTermGPU(ContextGPU &ctx, const SkinnedMesh &mesh, const cvx::PinholeCamera &cam):
        ctx_(ctx), mesh_(mesh), cam_(cam), skeleton_(mesh.skeleton_) {
        upload(mesh) ;
    }


    ~ModelToImageTermGPU() {
    }

    float energy(const Pose &pose) const ;
    std::pair<float, Eigen::VectorXf> energyGradient(const Pose &pose) const ;

    void energy(Eigen::VectorXf &e)  const;
    void jacobian(Eigen::MatrixXf &jac) const ;
    void norm(Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda) ;

    size_t nTerms() const { return mesh_.positions_.size() ; }
private:

    void upload(const SkinnedMesh &sk) ;

    void sampleVertices(thrust::device_vector<Vec3> &mpos,
                        thrust::device_vector<float> &errors, thrust::device_vector<Vec2> &gradients) const;



    void compute_gradient_sdf(const thrust::device_vector<Vec3> &mpos,
                              const thrust::device_vector<float> &distances,
                              const thrust::device_vector<Vec2> &gradients,
                              thrust::device_vector<float> &grad) const;


    void compute_jacobian_sdf(const thrust::device_vector<Vec3> &mpos, const thrust::device_vector<float> &distances,
                              const thrust::device_vector<Vec2> &gradients, thrust::device_vector<float> &grad) const;

    const Skeleton &skeleton_ ;

    size_t n_pts_ ;

    thrust::device_vector<Vec3> mesh_pos_ ;
    thrust::device_vector<VertexBoneDataGPU> mesh_vbd_ ;

    thrust::device_vector<float> d_ones_ ;

    const SkinnedMesh &mesh_ ;
    const cvx::PinholeCamera &cam_ ;
    ContextGPU &ctx_ ;
};


