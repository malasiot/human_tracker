#pragma once

#include <htrac/model/skeleton.hpp>
#include <htrac/util/matrix4x4.hpp>

#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <nppdefs.h>

#include <cvx/camera/camera.hpp>
#include <htrac/util/vector2.hpp>

#include "context.hpp"

using Plane = Eigen::Hyperplane<float, 3> ;

class ContextGPU: public Context {
public:
    ContextGPU(const Skeleton &sk);

    ~ContextGPU() ;

    void init() ;
    bool initNPP() ;
    void computeTransforms(const Pose &p) ;
    void computeDerivatives(const Pose &pose) ;

    void setPointCloud(const std::vector<Eigen::Vector3f> &icoords) {
        n_pts_ = icoords.size() ;
        icoords_ = icoords ;
    }

    void setPointCloud(const cv::Mat &im, const cv::Mat &mask, const cvx::PinholeCamera &cam) ;
    void setPointCloudClipped(const cv::Mat &im, const std::vector<Plane> &p, const cv::Mat &mask, const cvx::PinholeCamera &cam) ;

    void computeDistanceTransform(const cv::Size &sz) ;

    void normEq(const thrust::device_vector<float> &j, const thrust::device_vector<float> &e, float lambda,
                Eigen::MatrixXf &jtj, Eigen::VectorXf &jte) ;

    thrust::device_vector<Matrix4x4> trans_, itrans_, ioffset_, bder_, gder_ ;
    uint n_pose_bones_, n_global_params_, n_vars_, n_bones_ ;
    thrust::device_vector<size_t> pbone_dim_, pbone_offset_, pbone_idx_;
    thrust::device_vector<Vec3> icoords_ ;
    thrust::device_vector<unsigned char> mask_ ;

    // distance transform
    size_t width_, height_ ;
    thrust::device_vector<float> dtv_ ;
    thrust::device_vector<Vec2> dtg_ ;

    uint n_pts_ ;
  //  const Skeleton &skeleton_ ;
    cublasHandle_t cublas_handle_ ;
    NppStreamContext npp_stream_ctx_;
};
