#include "context_gpu.hpp"
#include "energy_term.hpp"
#include "cuda_util.hpp"

#include <vector>


#include <nppcore.h>
#include <nppi_filtering_functions.h>

using namespace std ;
using namespace Eigen ;

void ContextGPU::computeTransforms(const Pose &p) {
    Context::computeTransforms(p) ;
    trans_ = Context::trans_ ;
    itrans_ = Context::itrans_ ;
}

void ContextGPU::computeDerivatives(const Pose &p) {
    Context::computeDerivatives(p) ;
    bder_ = Context::bder_ ;
    gder_ = Context::gder_ ;
}

ContextGPU::ContextGPU(const Skeleton &sk): Context(sk) {
    init() ;
    initNPP() ;
    ioffset_ = Context::ioffset_ ;
}

ContextGPU::~ContextGPU() {
    cublasDestroy(cublas_handle_) ;
}

void ContextGPU::init() {
    n_bones_ = skeleton_.bones().size() ;
    n_vars_ = skeleton_.getNumPoseBoneParams() ;
    n_global_params_ = Pose::global_rot_params + 3 ;

    const auto &pbv = skeleton_.getPoseBones() ;

    n_pose_bones_ = pbv.size() ;

    std::vector<size_t> pbone_idx, pbone_dim, pbone_offset ;

    for( const auto &pb: pbv ) {
        pbone_dim.push_back(pb.dofs()) ;
        pbone_offset.push_back(pb.offset()) ;
        pbone_idx.push_back(skeleton_.getBoneIndex(pb.name())) ;
    }

    pbone_dim_ = pbone_dim ;
    pbone_idx_ = pbone_idx ;
    pbone_offset_ = pbone_offset ;

    cublasCreate(&cublas_handle_);
}

bool ContextGPU::initNPP()
{
    npp_stream_ctx_.hStream = 0; // The NULL stream by default, set this to whatever your created stream ID is if not the NULL stream.

     CudaSafeCall(cudaGetDevice(&npp_stream_ctx_.nCudaDeviceId));

     cudaError_t cudaError = cudaDeviceGetAttribute(&npp_stream_ctx_.nCudaDevAttrComputeCapabilityMajor,
                                        cudaDevAttrComputeCapabilityMajor,
                                        npp_stream_ctx_.nCudaDeviceId);
    if (cudaError != cudaSuccess)
         return false;

     cudaError = cudaDeviceGetAttribute(&npp_stream_ctx_.nCudaDevAttrComputeCapabilityMinor,
                                        cudaDevAttrComputeCapabilityMinor,
                                        npp_stream_ctx_.nCudaDeviceId);
     if (cudaError != cudaSuccess)
         return false;

     cudaError = cudaStreamGetFlags(npp_stream_ctx_.hStream, &npp_stream_ctx_.nStreamFlags);

     cudaDeviceProp oDeviceProperties;

     cudaError = cudaGetDeviceProperties(&oDeviceProperties, npp_stream_ctx_.nCudaDeviceId);

     npp_stream_ctx_.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
     npp_stream_ctx_.nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
     npp_stream_ctx_.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
     npp_stream_ctx_.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;
}


__global__ void imgToPointCloudKernel(uint w, uint h, float fx, float fy, float cx, float cy, const ushort *dim, const uchar *mask, uchar *omask, bool *labels, Vec3 *cloud) {
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    int ix = x ;
    int iy = y ;

    if (ix >= w || iy >= h) return ;

    uint idx = ix + iy*w;

    omask[idx] = 0 ;
    labels[idx] = false ;

    if ( dim[idx] != 0 && mask[idx] != 0 ) {
        labels[idx] = true ;
        float Z = dim[idx]/1000.0f ;
        Vec3 p{(ix - cx) * Z/fx,   -(iy - cy) * Z/fy, -Z };
        cloud[idx] = p ;
        omask[idx] = 255 ;
    }

}


struct PlaneGPU {

public:
    __device__ float signedDistance(const Vec3 &p) {
        return normal_.dot(p) + d_ ;
    }

    Vec3 normal_ ;
    float d_ ;
};


__global__ void imgToPointCloudKernelClipped(uint w, uint h, float fx, float fy, float cx, float cy, const ushort *dim,
                                             const uchar *mask, uchar *omask, bool *labels,
                                             Vec3 *cloud, PlaneGPU *planes, int np) {
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return ;

    uint idx = x + y*w;

    if ( dim[idx] == 0 || mask[idx] == 0 ) {
        labels[idx] = false ;
        omask[idx] = 0 ;
    } else {
        float Z = dim[idx]/1000.0f ;
        Vec3 p{(x - cx) * Z/fx,   -(y - cy) * Z/fy, -Z };

        labels[idx] = true ;
        cloud[idx] = p ;
        omask[idx] = 255 ;

        for( int i=0 ; i<np ; i++ ) {
             if ( planes[i].signedDistance(p) < 0 ) {
                labels[idx] = false ;
                omask[idx] = 0;
            }

        }
    }

}
struct non_masked
{

  __host__ __device__
  bool operator()(bool l)
  {
    return l;
  }
};


void ContextGPU::setPointCloud(const cv::Mat &im, const cv::Mat &mask, const cvx::PinholeCamera &cam)
{
    thrust::device_vector<ushort> im_gpu ;
    thrust::device_vector<uchar> src_mask_gpu ;

    uint w = im.cols, h = im.rows ;
    im_gpu.resize(w * h) ; mask_.resize(w * h) ;
    im_gpu.assign((const ushort *)im.data, (const ushort *)im.data + w * h) ;

    src_mask_gpu.resize(w * h) ;

    if ( mask.empty() )
        thrust::fill(src_mask_gpu.begin(), src_mask_gpu.end(), 255) ;
    else
        src_mask_gpu.assign((const uchar *)mask.data, (const uchar *)mask.data + w * h) ;


    thrust::device_vector<Vec3> cloud(w * h) ;
    thrust::device_vector<bool> labels(w * h) ;

    dim3 block(16,8,1);
    dim3 grid( ceil( w / (float)block.x), ceil( h / (float)block.y ));
    imgToPointCloudKernel<<<grid,block>>>(w, h,  cam.fx(), cam.fy(), cam.cx(), cam.cy(), TPTR(im_gpu), TPTR(src_mask_gpu), TPTR(mask_),
                                          TPTR(labels), TPTR(cloud)) ;


    CudaSafeCall(cudaDeviceSynchronize()) ;

//    vector<bool> hlabels(labels.size()) ;
 //   thrust::copy(labels.begin(), labels.end(), hlabels.begin());

    icoords_.resize(w * h) ;

    auto it = thrust::copy_if(cloud.begin(), cloud.end(), labels.begin(), icoords_.begin(), non_masked());

//    vector<uchar> hmask(mask_.size()) ;
//    thrust::copy(mask_.begin(), mask_.end(), hmask.begin());

//    cv::Mat mm(h, w, CV_8UC1, hmask.data()) ;
//    cv::imwrite("/tmp/mm.png", mm) ;


    n_pts_ = it - icoords_.begin() +1 ;

}

void ContextGPU::setPointCloudClipped(const cv::Mat &im, const std::vector<Plane> &planes, const cv::Mat &mask,
                                      const cvx::PinholeCamera &cam)
{
    thrust::device_vector<ushort> im_gpu ;
    thrust::device_vector<uchar> src_mask_gpu ;

    uint w = im.cols, h = im.rows ;
    im_gpu.resize(w * h) ; mask_.resize(w * h) ;
    im_gpu.assign((const ushort *)im.data, (const ushort *)im.data + w * h) ;

    src_mask_gpu.resize(w * h) ;

    if ( mask.empty() )
        thrust::fill(src_mask_gpu.begin(), src_mask_gpu.end(), 255) ;
    else
        src_mask_gpu.assign((const uchar *)mask.data, (const uchar *)mask.data + w * h) ;

    std::vector<PlaneGPU> planes_host(planes.size()) ;
    for( int i=0 ; i<planes.size() ; i++ ) {
        planes_host[i].normal_ = planes[i].normal() ;
        planes_host[i].d_ = planes[i].offset() ;
    }

    thrust::device_vector<PlaneGPU> planes_gpu(planes_host) ;

    thrust::device_vector<Vec3> cloud(w * h) ;
    thrust::device_vector<bool> labels(w * h) ;

    dim3 block(16,8,1);
    dim3 grid( ceil( w / (float)block.x), ceil( h / (float)block.y ));
    imgToPointCloudKernelClipped<<<grid,block>>>(w, h, cam.fx(), cam.fy(), cam.cx(), cam.cy(), TPTR(im_gpu), TPTR(src_mask_gpu), TPTR(mask_),
                                                  TPTR(labels), TPTR(cloud), TPTR(planes_gpu), planes_gpu.size()) ;


    CudaSafeCall(cudaDeviceSynchronize()) ;

    //    vector<bool> hlabels(labels.size()) ;
    //   thrust::copy(labels.begin(), labels.end(), hlabels.begin());

    icoords_.resize(w * h) ;

    auto it = thrust::copy_if(cloud.begin(), cloud.end(), labels.begin(), icoords_.begin(), non_masked());

    n_pts_ = it - icoords_.begin() +1 ;

        vector<uchar> hmask(mask_.size()) ;
        thrust::copy(mask_.begin(), mask_.end(), hmask.begin());

        cv::Mat mm(h, w, CV_8UC1, hmask.data()) ;
        cv::imwrite("/tmp/mm.png", mm) ;


}


__global__ void dtGradientKernel(uint w, uint h,  const float *dt, Vec2 *grad) {
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return ;

    uint idx = x + y*w;

    grad[idx].x() = ( x < w - 1 ) ? dt[x+1 + y*w] - dt[idx] : 0.0f;
    grad[idx].y() = ( y < h - 1 ) ? dt[x + (y+1)*w] - dt[idx] : 0.0f;
}

void ContextGPU::computeDistanceTransform(const cv::Size &imsz)
{
    width_ = imsz.width ; height_ = imsz.height ;
    size_t sz = width_ * height_ ;

    // negate mask
    //thrust::device_vector<uchar> xmask(sz) ;
//    xmask.assign((const uchar *)mask.data, (const uchar *)mask.data + sz) ;
    thrust::transform(mask_.begin(), mask_.end(), mask_.begin(), [] __device__ (uchar a) { return (a == 255) ? 0 : 255 ; });

    // compute distance transform using CUDA npp

    NppStatus npp_status;

    size_t scratch_buffer_size ;
    Npp8u * scratch_buffer ;

    NppiSize roi;

    roi.width = width_ ;
    roi.height = height_ ;
    npp_status = nppiDistanceTransformPBAGetBufferSize(roi, &scratch_buffer_size);

    assert(npp_status == NPP_NO_ERROR);

    cudaMalloc((void **) &scratch_buffer, scratch_buffer_size);

    dtv_.resize(sz) ; dtg_.resize(sz) ;

    Npp8u nMinSiteValue = 0;
    Npp8u nMaxSiteValue = 0;

    assert (nppiDistanceTransformPBA_8u32f_C1R_Ctx(TPTR(mask_), roi.width * sizeof(Npp8u), nMinSiteValue, nMaxSiteValue,
                                                  0, 0, 0, 0, 0, 0,
                                                  TPTR(dtv_), roi.width * sizeof(Npp32f),
                                                  roi, scratch_buffer, npp_stream_ctx_) == NPP_SUCCESS);

    cudaFree(scratch_buffer) ;

    // set distance transform to zero inside mask
    thrust::transform_if(thrust::device, dtv_.begin(), dtv_.end(), mask_.begin(), dtv_.begin(),
                         [] __host__ __device__ (float v) { return 0.f ; },
                         [] __host__ __device__ (uchar v) { return v == 0 ; });

    // compute gradient

    dim3 block(16,8,1);
    dim3 grid( ceil( width_ / (float)block.x), ceil( height_ / (float)block.y ));
    dtGradientKernel<<<grid,block>>>(width_, height_, TPTR(dtv_), TPTR(dtg_)) ;
/*
    vector<float> data(sz) ;
    thrust::copy(dtv_.begin(), dtv_.end(), data.begin());
        vector<uchar> datac(roi.height * roi.width) ;
        std::transform(data.begin(),data.end(), datac.begin(), [](float v) { return (uchar)v ; });
    cv::Mat res(roi.height, roi.width, CV_8UC1, &datac[0]) ;
    cv::imwrite("/tmp/dt.png", res) ;
    */
}

// compute J'J and J'e
void ContextGPU::normEq(const thrust::device_vector<float> &j, const thrust::device_vector<float> &e, float lambda, Eigen::MatrixXf &jtj, Eigen::VectorXf &jte)
{
    uint N = jte.size() ;
    uint n = e.size() ;
    thrust::device_vector<float> jtj_gpu(jtj.data(), jtj.data() + N * N) ;
    thrust::device_vector<float> jte_gpu(jte.data(), jte.data() + N) ;

    float alpha = lambda/n, beta = 1.0f ;
    cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, N, N, n, &alpha,
        TPTR(j), n, TPTR(j), n, &beta, TPTR(jtj_gpu), N);

    thrust::copy(jtj_gpu.begin(), jtj_gpu.end(), jtj.data());

    alpha = -alpha;
    cublasSgemv(cublas_handle_, CUBLAS_OP_T, n, N, &alpha, TPTR(j), n, TPTR(e), 1, &beta, TPTR(jte_gpu), 1);

    thrust::copy(jte_gpu.begin(), jte_gpu.end(), jte.data());
}

