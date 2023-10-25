#include "image_to_model_term_gpu.hpp"

#include <iostream>


using namespace Eigen ;
using namespace std ;


struct non_outlier {
    size_t label_max_ ;
    non_outlier(size_t n): label_max_(n) {}
    __host__ __device__
      bool operator()(size_t x) {
        return x < label_max_ ;
      }
};

template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const {
            return x * x;
        }
};


double ImageToModelTermGPU::energy(const Pose &pose) const {
    ctx_.computeTransforms(pose);

    size_t n = ctx_.n_pts_ ;

    thrust::device_vector<Vec3> tpts ;
    thrust::device_vector<float> distances, mdist(n) ;
    thrust::device_vector<size_t> labels(n) ;

    sdf_.transform_points_to_bone_space(ctx_.icoords_, tpts, ctx_.itrans_) ;
    sdf_.sample(n, tpts, distances) ;
    findBestPart(distances, mdist, labels);
    size_t count = thrust::count_if(thrust::device, labels.begin(), labels.end(), non_outlier(sdf_.primitives_.size()));

    return ( count > 0 ) ?
             thrust::transform_reduce(mdist.begin(), mdist.end(), thrust::square<float>(), (float)0, thrust::plus<float>()) : 1.0e9 ;

 /*   vector<size_t> hlabels(labels.size()) ;
    vector<float> hdist(mdist.size()) ;
    vector<float> hdistances(distances.size()) ;
    vector<Vector3> hpts(tpts.size()) ;

    thrust::copy(labels.begin(), labels.end(), hlabels.begin());
    thrust::copy(mdist.begin(), mdist.end(), hdist.begin());
    thrust::copy(distances.begin(), distances.end(), hdistances.begin());
    thrust::copy(tpts.begin(), tpts.end(), hpts.begin());

    for( int i=0 ;i<n ; i++) {
        for(int p=0 ; p<sdf_.primitives_.size() ; p++ ) {
            cout << hdistances[i * sdf_.primitives_.size() + p] << ' ';
        }
        cout << endl ;
    }
*/
}

#define IDX3(i, j, k) (n_pose_bones * 4 * (i) + 4 * (j) + (k))
#define IDX2(i, j) ( n_global_vars * (i) + j)

static __device__ void print_mat(const Matrix4x4 &m) {
        printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n",
               m(0, 0), m(0, 1), m(0, 2), m(0, 3),
               m(1, 0), m(1, 1), m(1, 2), m(1, 3),
               m(2, 0), m(2, 1), m(2, 2), m(2, 3),
               m(3, 0), m(3, 1), m(3, 2), m(3, 3)) ;
}

__global__ void energyGradientKernel(uint nparts, uint n_pose_bones, uint n_global_vars, uint npts, const size_t *labels,
                                     const Vec3 *icoords, const PrimitiveGPU *pr, const float *distances,
                                     const Vec3 *gradients, const Matrix4x4 *itrans, const Matrix4x4 *der,
                                     const Matrix4x4 *gder,
                                     const size_t *pbone_dim, const size_t *pbone_offset,
                                     float *energy_grad_partial, bool square)
{
    uint tc = threadIdx.x ;
    uint bs = blockDim.x ;
    uint idx = blockIdx.x * blockDim.x  + threadIdx.x ;
    uint var = blockIdx.y ;
    uint nvars = n_pose_bones + n_global_vars ;

    extern __shared__ char smem[] ;

    size_t c = 0 ;
    size_t *slabels = (size_t *)smem ; c += sizeof(size_t) * bs ;
    float *sdistances = (float *)&smem[c] ; c += sizeof(float) * bs ;
    Vec3 *sgradients = (Vec3 *)&smem[c] ; c += sizeof(Vec3) * bs ;
    Vec3 *scoords = (Vec3 *)&smem[c] ;

   if ( idx < npts ) {
        sdistances[tc] = distances[idx] ;
        slabels[tc] = labels[idx] ;
        sgradients[tc] = gradients[idx] ;
        scoords[tc] = icoords[idx] ;
    }
    __syncthreads() ;

    if ( idx < npts ) {

        float ve = sdistances[tc] ;

        size_t part = slabels[tc] ;

        if ( part >= nparts ) return ;

        const Vec3 &grad = sgradients[tc] ;

        const PrimitiveGPU &primitive = pr[part] ;

        uint bone = primitive.bone_ ;

        const Matrix4x4 &t = itrans[bone] ;
        const Vec3 &pt = scoords[tc] ;

        if ( var < n_global_vars ) {

           const Matrix4x4 &dQ = gder[IDX2(bone, var)] ;

           Matrix4x4 dG =  - t * dQ * t ;
           Vec3 dp =  dG * pt ;

           float de = grad.dot(dp) ;

           if ( square )
                energy_grad_partial[var * npts + idx] = 2*ve*de ;
           else
               energy_grad_partial[var * npts + idx] = ( slabels[tc] >= nparts ) ? 0 : de ;
              // energy_grad_partial[idx * nvars + var] = de ;
        } else {
            var -= n_global_vars ;
            uint k = var / 4 ;// pose bone index
            uint r = var % 4 ;

            if ( r >= pbone_dim[k] ) return ;

            const Matrix4x4 &dQ = der[IDX3(bone, k, r)] ;
            Matrix4x4 dG =  - t * dQ * t ;
            Vec3 dp =  dG * pt ;

            float de = grad.dot(dp) ;

            uint ov = n_global_vars + pbone_offset[k] + r ;

            if ( square )
                 energy_grad_partial[ov * npts + idx] = 2*ve*de ;
            else
                energy_grad_partial[ov * npts + idx] = ( slabels[tc] >= nparts ) ? 0 : de ;
                //energy_grad_partial[idx * nvars + ov] = de ;

        }

    }
}


void ImageToModelTermGPU::compute_gradient_sdf(const thrust::device_vector<float> &distances,
                                               const thrust::device_vector<size_t> &labels,
                                               const thrust::device_vector<Vec3> &gradients,
                                               uint count,
                                               thrust::device_vector<float> &grad) const
{
    uint npts = ctx_.n_pts_ ;
    uint nv = ctx_.n_pose_bones_ * 4 + ctx_.n_global_params_ ;
    uint nvars = ctx_.n_vars_ + ctx_.n_global_params_ ;

    grad.resize(nvars) ;

    thrust::device_vector<float> g(npts * nvars) ;

    thrust::fill(g.begin(), g.end(), (float)0) ;

    const uint block_size = CUDA_BLOCK_SIZE ;
    uint grid_size = (npts + block_size - 1) / block_size;

    uint smem_size = block_size * ( 8 * sizeof(float) + sizeof(uint)) ;

    // compute the contribution of a slice of the points to a single variable

   // vector<Matrix4x4> der(gder_.size()) ;
  //  thrust::copy(gder_.begin(), gder_.end(), der.begin());

    energyGradientKernel<<<dim3(grid_size, nv), block_size, smem_size>>>(sdf_.primitives_.size(), ctx_.n_pose_bones_,
                                                              ctx_.n_global_params_, npts, TPTR(labels),
                                                              TPTR(ctx_.icoords_), TPTR(sdf_.primitives_),  TPTR(distances), TPTR(gradients),
                                                              TPTR(ctx_.itrans_),TPTR(ctx_.bder_), TPTR(ctx_.gder_),
                                                              TPTR(ctx_.pbone_dim_), TPTR(ctx_.pbone_offset_), TPTR(g), true ) ;

    CudaSafeCall(cudaDeviceSynchronize())  ;

    float alpha = 1.0/count;
    float beta  = 0.f;

    thrust::device_vector<float> ones(npts, 1.0f) ;

    CublasSafeCall(cublasSgemv(ctx_.cublas_handle_, CUBLAS_OP_T, npts, nvars, &alpha, TPTR(g), npts,
                               TPTR(ones), 1, &beta, TPTR(grad), 1));

}

void ImageToModelTermGPU::compute_jacobian_sdf(const thrust::device_vector<float> &distances,
                                               const thrust::device_vector<size_t> &labels,
                                               const thrust::device_vector<Vec3> &gradients,
                                               thrust::device_vector<float> &g) const
{
    uint npts = ctx_.n_pts_ ;
    uint nv = ctx_.n_pose_bones_ * 4 + ctx_.n_global_params_ ;
    uint nvars = ctx_.n_vars_ + ctx_.n_global_params_ ;

    g.resize(npts * nvars) ;


    thrust::fill(g.begin(), g.end(), (float)0) ;

    const uint block_size = CUDA_BLOCK_SIZE ;
    uint grid_size = (npts + block_size - 1) / block_size;

    uint smem_size = block_size * ( 8 * sizeof(float) + sizeof(uint)) ;

    // compute the contribution of a slice of the points to a single variable

   // vector<Matrix4x4> der(gder_.size()) ;
  //  thrust::copy(gder_.begin(), gder_.end(), der.begin());

    energyGradientKernel<<<dim3(grid_size, nv), block_size, smem_size>>>(sdf_.primitives_.size(), ctx_.n_pose_bones_,
                                                              ctx_.n_global_params_, npts, TPTR(labels),
                                                              TPTR(ctx_.icoords_), TPTR(sdf_.primitives_),  TPTR(distances), TPTR(gradients),
                                                              TPTR(ctx_.itrans_),TPTR(ctx_.bder_), TPTR(ctx_.gder_),
                                                              TPTR(ctx_.pbone_dim_), TPTR(ctx_.pbone_offset_), TPTR(g), false ) ;
}


std::pair<float, VectorXf> ImageToModelTermGPU::energyGradient(const Pose &pose) const {
    ctx_.computeTransforms(pose);
    ctx_.computeDerivatives(pose) ;

    size_t n = ctx_.n_pts_ ;

    thrust::device_vector<Vec3> tpts, gradients(n) ;
    thrust::device_vector<float> distances, mdist(n), grad ;
    thrust::device_vector<size_t> labels(n) ;

    sdf_.transform_points_to_bone_space(ctx_.icoords_, tpts, ctx_.itrans_) ;
    sdf_.sample(n, tpts, distances) ;
    findBestPart(distances, mdist, labels);
    sdf_.grad(n, tpts, labels, gradients) ;
    size_t count = thrust::count_if(thrust::device, labels.begin(), labels.end(), non_outlier(sdf_.primitives_.size()));

    compute_gradient_sdf(mdist, labels, gradients, count, grad) ;

    VectorXf diffE(grad.size()) ;

    thrust::copy_n(grad.begin(), diffE.size(), diffE.data()) ;

    float e = ( count > 0 ) ? thrust::transform_reduce(mdist.begin(), mdist.end(), thrust::square<float>(), (float)0, thrust::plus<float>())/count : 1.0e9  ;

    return make_pair(e, diffE) ;
}

void ImageToModelTermGPU::energy(VectorXf &e) const
{
    size_t n = ctx_.n_pts_ ;

    thrust::device_vector<Vec3> tpts ;
    thrust::device_vector<float> distances, mdist(n) ;
    thrust::device_vector<size_t> labels(n) ;

    sdf_.transform_points_to_bone_space(ctx_.icoords_, tpts, ctx_.itrans_) ;
    sdf_.sample(n, tpts, distances) ;
    findBestPart(distances, mdist, labels);

    thrust::copy(mdist.begin(), mdist.end(), e.data());
}

void ImageToModelTermGPU::jacobian(MatrixXf &jac) const
{
    size_t n = ctx_.n_pts_ ;

    thrust::device_vector<Vec3> tpts, gradients(n) ;
    thrust::device_vector<float> distances, mdist(n), grad ;
    thrust::device_vector<size_t> labels(n) ;

    sdf_.transform_points_to_bone_space(ctx_.icoords_, tpts, ctx_.itrans_) ;
    sdf_.sample(n, tpts, distances) ;
    findBestPart(distances, mdist, labels);
    sdf_.grad(n, tpts, labels, gradients) ;

    compute_jacobian_sdf(mdist, labels, gradients, grad) ;

    thrust::copy(grad.begin(), grad.end(), jac.data());


}

void ImageToModelTermGPU::norm(Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda)
{
    size_t n = ctx_.n_pts_ ;

    thrust::device_vector<Vec3> tpts, gradients(n) ;
    thrust::device_vector<float> distances, mdist(n), grad ;
    thrust::device_vector<size_t> labels(n) ;

    sdf_.transform_points_to_bone_space(ctx_.icoords_, tpts, ctx_.itrans_) ;
    sdf_.sample(n, tpts, distances) ;
    findBestPart(distances, mdist, labels);
    sdf_.grad(n, tpts, labels, gradients) ;

    compute_jacobian_sdf(mdist, labels, gradients, grad) ;

    ctx_.normEq(grad, mdist, lambda, jtj, jte) ;
}

__global__ void minDistKernel(uint npts, uint nparts, const float *samples, float *distances, size_t *labels, float outlier_threshold)
{
    uint idx = blockIdx.x *  blockDim.x + threadIdx.x ;

    if ( idx < npts ) {

        int bestidx = -1 ;
        float minv = std::numeric_limits<float>::max() ;

        // find minimum element of data stored in shared memory
        for (uint part=0 ; part<nparts ; part++ )
        {
            float v = samples[idx * nparts + part] ;

            if ( fabs(v) < fabs(minv)) {
                minv = v ;
                bestidx = part ;
            }
        }

        if ( fabs(minv) > outlier_threshold ) {
            labels[idx] = 1000 ;
            distances[idx] = 0 ;
        } else {
            labels[idx] = bestidx ;
            distances[idx] = minv ;
        }
    }
}

void ImageToModelTermGPU::findBestPart(const thrust::device_vector<float> &samples,
                                         thrust::device_vector<float> &distances,
                                         thrust::device_vector<size_t> &labels) const {
    uint n = ctx_.n_pts_;
    uint np = sdf_.primitives_.size() ;
    const uint blockSize = CUDA_BLOCK_SIZE ;

    uint gridSize = (n + blockSize - 1) / blockSize;

    minDistKernel<<<gridSize, blockSize>>>(n, np, TPTR(samples), TPTR(distances), TPTR(labels), outlier_threshold_) ;

    CudaSafeCall(cudaDeviceSynchronize()) ;
}
