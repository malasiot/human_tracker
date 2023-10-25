#include "model_to_image_term_gpu.hpp"
#include "distance_transform.hpp"
#include "energy_term.hpp"


using namespace std ;
using namespace Eigen ;


float ModelToImageTermGPU::energy(const Pose &pose) const {
    thrust::device_vector<Vec3> mpos ;
    thrust::device_vector<float> errors ;
    thrust::device_vector<Vec2> gradients ;

    ctx_.computeTransforms(pose);

    sampleVertices(mpos, errors, gradients) ;

    float res = thrust::transform_reduce(errors.begin(), errors.end(), thrust::square<float>(), 0.0f, thrust::plus<float>());
    return res/mesh_pos_.size() ;
}

void ModelToImageTermGPU::energy(VectorXf &e) const {
    thrust::device_vector<Vec3> mpos ;
    thrust::device_vector<float> errors ;
    thrust::device_vector<Vec2> gradients ;

    sampleVertices(mpos, errors, gradients) ;

    thrust::copy(errors.begin(), errors.end(), e.data());
}


static __device__ Vec2 proj_deriv(float f, const Vec3 &p, const Vec3 &dp) {
    float X = p.x(), Y = p.y(), Z = p.z() ;
    float ZZ = Z * Z / f ;
    return { -( dp.x() * Z -  dp.z() * X )/ZZ,  ( dp.y() * Z -  dp.z() * Y )/ZZ } ;
}

#define IDX3(i, j, k) (n_pose_bones * 4 * (i) + 4 * (j) + (k))
#define IDX2(i, j) ( n_global_vars * (i) + j)

__global__ void energyGradientKernel(uint npts, uint n_pose_bones, uint n_global_vars,
                                     const Vec3 *pos, const Vec3 *coords, const float *distances,
                                     const Vec2 *gradients, const VertexBoneDataGPU *vbd, const Matrix4x4 *der,
                                     const Matrix4x4 *gder, const Matrix4x4 *ioffset,
                                     const size_t *pbone_dim, const size_t *pbone_offset,
                                     float f,
                                     float *energy_grad_partial, bool square)
{
    uint tc = threadIdx.x ;
    uint bs = blockDim.x ;
    uint idx = blockIdx.x * blockDim.x  + threadIdx.x ;
    uint var = blockIdx.y ;
    uint nvars = n_pose_bones + n_global_vars ;

    extern __shared__ char smem[] ;

    size_t c = 0 ;
    Vec3 *spos = (Vec3 *)smem ; c += sizeof(Vec3) * bs ;
    float *sdistances = (float *)&smem[c] ; c += sizeof(float) * bs ;
    Vec2 *sgradients = (Vec2 *)&smem[c] ; c += sizeof(Vec2) * bs ;
    VertexBoneDataGPU *svbd = (VertexBoneDataGPU *)&smem[c] ; c += sizeof(VertexBoneDataGPU) * bs ;
    Vec3 *scoords = (Vec3 *)&smem[c] ;

    if ( idx < npts ) {
        sdistances[tc] = distances[idx] ;
        spos[tc] = pos[idx] ;
        sgradients[tc] = gradients[idx] ;
        scoords[tc] = coords[idx] ;
        svbd[tc] = vbd[idx] ;
    }
    __syncthreads() ;

    if ( idx < npts ) {

        Vec2 og = sgradients[tc] ;
        float ve = sdistances[tc] ;
        const Vec3 &pt = scoords[tc] ;
        const Vec3 &orig = spos[tc] ;


        const auto &bdata = svbd[tc] ;

        if ( var < n_global_vars ) {

            Matrix4x4 dG ;
            dG.setZero() ;

            for( int j=0 ; j<MAX_BONES_PER_VERTEX ; j++)  {
                int bidx = bdata.id_[j] ;
                if ( bidx < 0 ) break ;

                Matrix4x4 dQ = gder[IDX2(bidx, var)] * ioffset[bidx];

                dG += dQ * bdata.weight_[j] ;
            }


            Vec3 dp =  dG * orig ;

            Vec2 dpg = proj_deriv(f, pt, dp) ;
            float gd = og.dot(dpg) ;

            //   printf("%f %f %f\n", gd, og.x(), og.y()) ;

            if ( square )
                energy_grad_partial[var * npts + idx] = 2*ve*gd ;
            else
                energy_grad_partial[var * npts + idx] =  gd ;

        } else {
            var -= n_global_vars ;
            uint k = var / 4 ;// pose bone index
            uint r = var % 4 ;

            if ( r >= pbone_dim[k] ) return ;

            Matrix4x4 dG ;
            dG.setZero();

            for( int j=0 ; j<MAX_BONES_PER_VERTEX ; j++)  {
                int bidx = bdata.id_[j] ;
                if ( bidx < 0 ) break ;

                Matrix4x4 dQ = der[IDX3(bidx, k, r)] * ioffset[bidx];

                dG += dQ * bdata.weight_[j] ;
            }

            Vec3 dp =  dG * orig ;

            Vec2 dpg = proj_deriv(f, pt, dp) ;
            float gd = og.dot(dpg) ;

            uint ov = n_global_vars + pbone_offset[k] + r ;
            if ( square )
                energy_grad_partial[ov * npts + idx] = 2*ve*gd ;
            else
                energy_grad_partial[ov * npts + idx] =  -gd ;

        }
    }
}

void ModelToImageTermGPU::compute_gradient_sdf(const thrust::device_vector<Vec3> &mpos,
                                               const thrust::device_vector<float> &distances,
                                               const thrust::device_vector<Vec2> &gradients,
                                               thrust::device_vector<float> &grad) const
{
    uint nv = ctx_.n_pose_bones_ * 4 + ctx_.n_global_params_ ;
    uint nvars = ctx_.n_vars_ + ctx_.n_global_params_ ;

    grad.resize(nvars) ;

    thrust::device_vector<float> g(n_pts_ * nvars) ;

    thrust::fill(g.begin(), g.end(), (float)0) ;

    const uint block_size = CUDA_BLOCK_SIZE ;
    uint grid_size = (n_pts_ + block_size - 1) / block_size;

    uint smem_size = block_size * ( 9 * sizeof(float) + sizeof(VertexBoneDataGPU)) ;

    // compute the contribution of a slice of the points to a single variable


    energyGradientKernel<<<dim3(grid_size, nv), block_size, smem_size>>>(n_pts_, ctx_.n_pose_bones_, ctx_.n_global_params_,
                                                                         TPTR(mesh_pos_), TPTR(mpos), TPTR(distances), TPTR(gradients),
                                                                         TPTR(mesh_vbd_), TPTR(ctx_.bder_), TPTR(ctx_.gder_), TPTR(ctx_.ioffset_),
                                                                         TPTR(ctx_.pbone_dim_), TPTR(ctx_.pbone_offset_), cam_.fx(), TPTR(g), true ) ;

    CudaSafeCall(cudaDeviceSynchronize())  ;

    float alpha = 1.0/n_pts_;
    float beta  = 0.f;

    thrust::device_vector<float> ones(n_pts_, 1.0f) ;

    CublasSafeCall(cublasSgemv(ctx_.cublas_handle_, CUBLAS_OP_T, n_pts_, nvars, &alpha, TPTR(g), n_pts_,
                               TPTR(ones), 1, &beta, TPTR(grad), 1));

}

std::pair<float, VectorXf> ModelToImageTermGPU::energyGradient(const Pose &pose) const {

    ctx_.computeTransforms(pose);
    ctx_.computeDerivatives(pose) ;

    thrust::device_vector<Vec3> mpos ;
    thrust::device_vector<float> errors, grad ;
    thrust::device_vector<Vec2> gradients ;

    sampleVertices(mpos, errors, gradients) ;

    compute_gradient_sdf(mpos, errors, gradients, grad) ;

    VectorXf diffE(grad.size()) ;

    thrust::copy_n(grad.begin(), diffE.size(), diffE.data()) ;

    float e = thrust::transform_reduce(errors.begin(), errors.end(), thrust::square<float>(), (float)0, thrust::plus<float>())/n_pts_  ;

    return make_pair(e, diffE) ;
}

void ModelToImageTermGPU::compute_jacobian_sdf(const thrust::device_vector<Vec3> &mpos,
                                               const thrust::device_vector<float> &distances,
                                               const thrust::device_vector<Vec2> &gradients,
                                               thrust::device_vector<float> &g) const
{
    uint nv = ctx_.n_pose_bones_ * 4 + ctx_.n_global_params_ ;
    uint nvars = ctx_.n_vars_ + ctx_.n_global_params_ ;

    g.resize(n_pts_ * nvars) ;

    thrust::fill(g.begin(), g.end(), (float)0) ;

    const uint block_size = CUDA_BLOCK_SIZE ;
    uint grid_size = (n_pts_ + block_size - 1) / block_size;

    uint smem_size = block_size * ( 9 * sizeof(float) + sizeof(VertexBoneDataGPU)) ;

    // compute the contribution of a slice of the points to a single variable

    energyGradientKernel<<<dim3(grid_size, nv), block_size, smem_size>>>(n_pts_, ctx_.n_pose_bones_, ctx_.n_global_params_,
                                                                         TPTR(mesh_pos_), TPTR(mpos), TPTR(distances), TPTR(gradients),
                                                                         TPTR(mesh_vbd_), TPTR(ctx_.bder_), TPTR(ctx_.gder_), TPTR(ctx_.ioffset_), TPTR(ctx_.pbone_dim_), TPTR(ctx_.pbone_offset_),
                                                                         cam_.fx(), TPTR(g), false ) ;

    CudaSafeCall(cudaDeviceSynchronize())  ;
}


void ModelToImageTermGPU::jacobian(MatrixXf &jac) const
{
    thrust::device_vector<Vec3> mpos ;
    thrust::device_vector<float> errors, grad ;
    thrust::device_vector<Vec2> gradients ;

    sampleVertices(mpos, errors, gradients) ;

    compute_jacobian_sdf(mpos, errors, gradients, grad) ;

    thrust::copy(grad.begin(), grad.end(), jac.data());
}

void ModelToImageTermGPU::norm(Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda)
{
    thrust::device_vector<Vec3> mpos ;
    thrust::device_vector<float> errors, grad ;
    thrust::device_vector<Vec2> gradients ;

    sampleVertices(mpos, errors, gradients) ;

    compute_jacobian_sdf(mpos, errors, gradients, grad) ;

    ctx_.normEq(grad, errors, lambda, jtj, jte) ;
}

void ModelToImageTermGPU::upload(const SkinnedMesh &mesh) {
    n_pts_ = mesh.positions_.size() ;
    mesh_pos_ = mesh.positions_ ;
    mesh_vbd_ = mesh.bones_ ;
}


__device__ static uint reflect(int x, int sz) {
    if ( x < 0 ) return -x ;
    if ( x > sz-1 ) return 2 * (sz -1) - x ;
    return x ;
}

#define IIDX(i, j) ((i)*w + (j))

template<typename T>
static __device__ T bilinear(const T *dt, uint w, uint h, float ix, float iy)
{
    int x = (int)ix;
    int y = (int)iy;

    int x0 = reflect(x,   w);
    int x1 = reflect(x+1, w);
    int y0 = reflect(y,   h);
    int y1 = reflect(y+1, h);

    float a = ix - (float)x;
    float c = iy - (float)y;

    T v = (dt[IIDX(y0, x0)] * (1.f - a) + dt[IIDX(y0, x1)] * a) * (1.f - c) +
            (dt[IIDX(y1, x0)] * (1.f - a) + dt[IIDX(y1, x1)] * a) * c;

    return v ;
}


__global__ void sampleVerticesKernel(uint npts, const float *dtv, const Vec2 *dtg, uint w, uint h, float fx, float fy, float cx, float cy,
                                     const Vec3 *positions, const Matrix4x4 *trs, const Matrix4x4 *ioffset,
                                     const VertexBoneDataGPU *vbd, Vec3 *mpos, float *errors, Vec2 *grads)
{
    uint i = blockIdx.x *  blockDim.x + threadIdx.x ;

    if ( i < npts ) {
        const Vec3 &pos = positions[i] ;
        const VertexBoneDataGPU &bdata = vbd[i] ;

        Matrix4x4 boneTransform ;

        for( int j=0 ; j<MAX_BONES_PER_VERTEX ; j++) {
            int idx = bdata.id_[j] ;
            if ( idx < 0 ) break ;

            Matrix4x4 skt = trs[idx] * ioffset[idx] ;

            if ( j == 0 ) boneTransform = skt * bdata.weight_[j] ;
            else boneTransform += skt * (double)bdata.weight_[j] ;
        }

        mpos[i] = boneTransform * pos ;

        float ix = -fx * mpos[i].x() / mpos[i].z() + cx + 0.5 ;
        float iy = fy * mpos[i].y() / mpos[i].z() + cy + 0.5 ;

        errors[i] = bilinear(dtv, w, h, ix, iy) ;
        grads[i] = bilinear(dtg, w, h, ix, iy) ;
    }
}


void ModelToImageTermGPU::sampleVertices(thrust::device_vector<Vec3> &mpos,
                                         thrust::device_vector<float> &errors,
                                         thrust::device_vector<Vec2> &gradients) const {
    //  auto trs = mesh_.getSkinningTransforms(p) ;

    //  thrust::device_vector<Matrix4x4> trs_gpu(trs) ;

    mpos.resize(n_pts_) ;
    errors.resize(n_pts_) ; gradients.resize(n_pts_) ;

    const uint blockSize = CUDA_BLOCK_SIZE ;

    uint gridSize = (n_pts_ + blockSize - 1) / blockSize;

    sampleVerticesKernel<<<gridSize, blockSize>>>(n_pts_, TPTR(ctx_.dtv_), TPTR(ctx_.dtg_), ctx_.width_, ctx_.height_, cam_.fx(), cam_.fy(), cam_.cx(), cam_.cy(),
                                                  TPTR(mesh_pos_), TPTR(ctx_.trans_), TPTR(ctx_.ioffset_), TPTR(mesh_vbd_), TPTR(mpos),
                                                  TPTR(errors), TPTR(gradients)) ;

    CudaSafeCall(cudaDeviceSynchronize()) ;
}

