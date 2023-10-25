#include "collision_term_gpu.hpp"
#include "collision_term_gpu.cuh"

#include "energy_term.hpp"
#include "collision_data.hpp"
#include "cuda_util.hpp"
#include "context_gpu.hpp"
#include <htrac/util/matrix4x4.hpp>

#include <thrust/device_vector.h>

using namespace std ;
using namespace Eigen ;

CollisionTermGPU::CollisionTermGPU(ContextGPU &ctx, const Skeleton &sk, const CollisionData &cd):
    ctx_(ctx), skeleton_(sk), cd_(cd) {

    vector<Vec3> centers ;
    vector<float> rad ;
    vector<uint> bones, ps, pt ;

    for( size_t i=0 ; i<cd.spheres_.size() ; i++ ) {
        const auto &p = cd.spheres_[i] ;
        const auto &s1 = p.group_ ;

        centers.emplace_back(p.c_) ;
        rad.emplace_back(p.r_) ;
        bones.push_back(p.bone_) ;

        for( size_t j=0 ; j<i ; j++) {
            const auto &s2 = cd.spheres_[j].group_ ;

            if ( s1 == s2 ) continue ;
            ps.emplace_back(i) ;
            pt.emplace_back(j) ;
        }
    }

    centers_ = centers ;
    radius_ = rad ;
    ps_ = ps ;
    pt_ = pt ;
    bones_ = bones ;
}

__global__ void collisionEnergyKernel(size_t npairs, const Matrix4x4 *trans, const Vec3 *centers, const float *radius,
                                      const uint *ps, const uint *pt, const uint *bones,
                                      float *errors)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x ;

    if ( idx < npairs ) {
        uint idx1 = ps[idx] ;
        uint idx2 = pt[idx] ;

        uint bone_s = bones[idx1] ;
        uint bone_t = bones[idx2] ;

        const Matrix4x4 &bts = trans[bone_s] ;
        const Matrix4x4 &btt = trans[bone_t] ;

        const Vec3 &c0s = centers[idx1] ;
        const Vec3 &c0t = centers[idx2] ;

        const Vec3 cs = bts * c0s ;
        const Vec3 ct = btt * c0t ;

        float rs = radius[idx1] ;
        float rt = radius[idx2] ;

        Vec3 cst = cs - ct ;
        float g = cst.norm() ;
        float rst = (rs + rt)  ;
        float rstg = rst - g ;

        float e = std::max(0.f, rstg);

        errors[idx] = e ;
    }
}

void CollisionTermGPU::compute_errors(thrust::device_vector<float> &errors) const{

    size_t blockSize = CUDA_BLOCK_SIZE ;
    size_t n = ps_.size() ;

    uint gridSize = (n + blockSize - 1) / blockSize;

    errors.resize(n) ;

    collisionEnergyKernel<<<gridSize, blockSize>>>(n, TPTR(ctx_.trans_), TPTR(centers_), TPTR(radius_),
                                                   TPTR(ps_), TPTR(pt_), TPTR(bones_), TPTR(errors)) ;

    CudaSafeCall(cudaDeviceSynchronize()) ;
}

float CollisionTermGPU::energy(const Pose &pose) const {

    ctx_.computeTransforms(pose);

    thrust::device_vector<float> errors ;
    compute_errors(errors) ;

    float total = thrust::transform_reduce(errors.begin(), errors.end(), thrust::square<float>(), (float)0, thrust::plus<float>()) ;
    return total/errors.size() ;
}

void CollisionTermGPU::energy(VectorXf &e) const
{
    thrust::device_vector<float> errors ;
    compute_errors(errors) ;

    thrust::copy(errors.begin(), errors.end(), e.data());
}


#define IDX3(i, j, k) (n_pose_bones * 4 * (i) + 4 * (j) + (k))

__global__ void collisionEnergyGradientKernel(size_t npairs, const Matrix4x4 *trans, const Vec3 *centers, const float *radius,
                                                const uint *ps, const uint *pt, const uint *bones,
                                              const Matrix4x4 *bder,
                                              uint n_pose_bones, uint n_global_vars,
                                              const size_t *pbone_dim, const size_t *pbone_offset,
                                              float *grad, bool square) {
    uint idx = blockIdx.x * blockDim.x  + threadIdx.x ;
    uint var = blockIdx.y ;
    uint nvars = n_pose_bones + n_global_vars ;

    if ( var < n_global_vars ) return ;

    var -= n_global_vars ;
    uint k = var / 4 ;// pose bone index
    uint r = var % 4 ;

    if ( r >= pbone_dim[k] ) return ;

    if ( idx < npairs ) {
        uint idx1 = ps[idx] ;
        uint idx2 = pt[idx] ;

        uint bone_s = bones[idx1] ;
        uint bone_t = bones[idx2] ;

        const Matrix4x4 &bts = trans[bone_s] ;
        const Matrix4x4 &btt = trans[bone_t] ;

        const Vec3 &c0s = centers[idx1] ;
        const Vec3 &c0t = centers[idx2] ;

        const Vec3 cs = bts * c0s ;
        const Vec3 ct = btt * c0t ;

        float rs = radius[idx1] ;
        float rt = radius[idx2] ;

        Vec3 cst = cs - ct ;
        float g = cst.norm() ;
        float rst = (rs + rt)  ;
        float rstg = rst - g ;
        Vec3 dg0 = cst / g ;

        if ( rstg < 0 ) return ;

        Matrix4x4 dQs = bder[IDX3(bone_s, k, r)]  ;
        Vec3 dps =  dQs * c0s ;
        Matrix4x4 dQt = bder[IDX3(bone_t, k, r)] ;
        Vec3 dpt =  dQt * c0t ;

        float dG = (dps - dpt).dot(dg0) ;

        uint ov = n_global_vars + pbone_offset[k] + r ;

        if ( square )
             grad[ov * npairs + idx] = -2*rstg*dG ;
        else
            grad[ov * npairs + idx] = -dG ;
    }

}

void CollisionTermGPU::compute_gradient(thrust::device_vector<float> &grad) const
{
    uint npairs = ps_.size() ;
    uint nv = ctx_.n_pose_bones_ * 4 + ctx_.n_global_params_ ;
    uint nvars = ctx_.n_vars_ + ctx_.n_global_params_ ;

    grad.resize(nvars) ;

    thrust::device_vector<float> g(npairs * nvars) ;

    thrust::fill(g.begin(), g.end(), (float)0) ;

    const uint block_size = CUDA_BLOCK_SIZE ;
    uint grid_size = (npairs + block_size - 1) / block_size;

    collisionEnergyGradientKernel<<<dim3(grid_size, nv), block_size>>>(npairs, TPTR(ctx_.trans_), TPTR(centers_), TPTR(radius_),
                                 TPTR(ps_), TPTR(pt_), TPTR(bones_), TPTR(ctx_.bder_), ctx_.n_pose_bones_, ctx_.n_global_params_,
                                 TPTR(ctx_.pbone_dim_), TPTR(ctx_.pbone_offset_), TPTR(g), true ) ;

    CudaSafeCall(cudaDeviceSynchronize())  ;

    float alpha = 1.0/npairs;
    float beta  = 0.f;

    thrust::device_vector<float> ones(npairs, 1.0f) ;

    CublasSafeCall(cublasSgemv(ctx_.cublas_handle_, CUBLAS_OP_T, npairs, nvars, &alpha, TPTR(g), npairs,
                               TPTR(ones), 1, &beta, TPTR(grad), 1));
}


void CollisionTermGPU::compute_jacobian(thrust::device_vector<float> &jac) const
{
    uint npairs = ps_.size() ;
    uint nv = ctx_.n_pose_bones_ * 4 + ctx_.n_global_params_ ;
    uint nvars = ctx_.n_vars_ + ctx_.n_global_params_ ;

    jac.resize(npairs * nvars) ;

    thrust::fill(jac.begin(), jac.end(), (float)0) ;

    const uint block_size = CUDA_BLOCK_SIZE ;
    uint grid_size = (npairs + block_size - 1) / block_size;

    collisionEnergyGradientKernel<<<dim3(grid_size, nv), block_size>>>(npairs, TPTR(ctx_.trans_), TPTR(centers_), TPTR(radius_),
                                 TPTR(ps_), TPTR(pt_), TPTR(bones_), TPTR(ctx_.bder_), ctx_.n_pose_bones_, ctx_.n_global_params_,
                                 TPTR(ctx_.pbone_dim_), TPTR(ctx_.pbone_offset_), TPTR(jac), false ) ;

    CudaSafeCall(cudaDeviceSynchronize())  ;
}


std::pair<float, VectorXf> CollisionTermGPU::energyGradient(const Pose &pose) const {
    ctx_.computeTransforms(pose);
    ctx_.computeDerivatives(pose) ;

    thrust::device_vector<float> grad ;
    compute_gradient(grad);

    VectorXf diffE(grad.size()) ;

    thrust::copy_n(grad.begin(), diffE.size(), diffE.data()) ;

    return {0.0f, diffE} ;
}

void CollisionTermGPU::jacobian(MatrixXf &jac) const
{
    thrust::device_vector<float> j ;
    compute_jacobian(j);

    thrust::copy(j.begin(), j.end(), jac.data());

}

void CollisionTermGPU::norm(Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda) {

    thrust::device_vector<float> j, e ;
    compute_errors(e) ;
    compute_jacobian(j);
    ctx_.normEq(j, e, lambda, jtj, jte) ;
}

