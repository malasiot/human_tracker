#include <htrac/pose/model_fit.hpp>

#include <cvx/misc/path.hpp>
#include <cvx/math/rng.hpp>
#include <cvx/imgproc/rgbd.hpp>

#include <htrac/util/io_util.hpp>
#include <htrac/util/pcl_util.hpp>
#include <htrac/model/primitive_sdf.hpp>
#include <htrac/util/mhx2_importer.hpp>
#include "image_to_model_term_gpu.hpp"
#include "model_to_image_term_gpu.hpp"
#include "collision_term_gpu.hpp"
#include "joint_limits_term.hpp"
#include "keypoints_2d_term.hpp"

#include "solver.hpp"

using namespace cvx ;
using namespace std ;
using namespace Eigen ;

class HumanModelFitImpl {
public:
    HumanModelFitImpl(const HumanModelFit::Parameters &cfg) ;

    Pose fit(const std::vector<Vector3f> &cloud, const cv::Mat &mask, const cvx::PinholeCamera &cam, KeyPointsDistanceField *kpts, const Pose &orig) ;
    Pose fit(const cv::Mat &im, const cv::Mat &mask, const cvx::PinholeCamera &cam, KeyPointsDistanceField *kpts, const Pose &orig) ;

    void setClippingPlanes(const std::vector<Plane> &planes) {
        cp_ = planes ;
    }

    PrimitiveSDF sdf_ ;
    SkinnedMesh mesh_ ;
    CollisionData cd_ ;
    std::vector<Plane> cp_ ;

    HumanModelFit::Parameters params_ ;
};



HumanModelFitImpl::HumanModelFitImpl(const HumanModelFit::Parameters &params): params_(params) {
    mesh_.load("models/human-cmu-low-poly.mhx2") ;
    auto &sk = mesh_.skeleton_ ;

    sdf_.readJSON(sk, "sdf.json");

    PoseParameterization solve_bones ;

    parsePoseParameterization(solve_bones, "pose.json");
    sk.setPoseBones(solve_bones);

    cd_.parseJson(sk, "collision.json") ;
}

struct ObjectiveFunctionNLS {
    ObjectiveFunctionNLS(const HumanModelFit::Parameters &params, ContextGPU &ctx,
            const Skeleton &sk, const PrimitiveSDF &model,
                         const SkinnedMesh &mesh, const cvx::PinholeCamera &cam, const CollisionData &cd, KeyPointsDistanceField *kdf):
        params_(params), ctx_(ctx), sk_(sk),
        im_term_(ctx, sk, model, params.outlier_threshold_),
        mi_term_(ctx, mesh, cam),
        col_term_(ctx, sk, cd),
        jl_term_(sk), kpts_term_(ctx, cam, kdf) {
    }

    void errors(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const {
        Pose pose(&sk_, x) ;

        ctx_.computeTransforms(pose) ;

        size_t n1 = im_term_.nTerms() ;
        VectorXf fv1(n1) ;
        im_term_.energy(fv1) ;

        size_t n2 = mi_term_.nTerms() ;
        VectorXf fv2(n2) ;
        mi_term_.energy(fv2) ;

        size_t n3 = col_term_.nTerms() ;
       VectorXf fv3(n3) ;
       col_term_.energy(fv3) ;

      size_t n4 = jl_term_.nTerms() ;
      VectorXf fv4(n4) ;
      jl_term_.energy(pose, fv4) ;


      size_t n5 = kpts_term_.nTerms() ;
      VectorXf fv5(n5) ;
      kpts_term_.energy(fv5) ;

       fvec << fv1 * sqrt(params_.lambda_im_/n1), fv2 * sqrt(params_.lambda_mi_/n2),
               fv3 * sqrt(params_.lambda_col_/n3), fv4 * sqrt(params_.lambda_jl_/n4),
               fv5 * sqrt(params_.lambda_kp_/n5);

    }


    void debug(uint iter, float eL2, const VectorXf &p) {
        cout << eL2 << endl ;
    }

    void norm(const Eigen::VectorXf &x, Eigen::MatrixXf &jtj, Eigen::VectorXf &jte) {
        Pose pose(&sk_, x) ;

        ctx_.computeTransforms(pose);
        ctx_.computeDerivatives(pose);

        jtj.setZero() ; jte.setZero() ;

        im_term_.norm(jtj, jte, params_.lambda_im_) ;
        mi_term_.norm(jtj, jte, params_.lambda_mi_) ;
        col_term_.norm(jtj, jte, params_.lambda_col_) ;
        jl_term_.norm(pose, jtj, jte, params_.lambda_jl_) ;
        if ( kpts_term_.nTerms() )
            kpts_term_.norm(jtj, jte, params_.lambda_kp_) ;
    }

    int terms() const { return im_term_.nTerms() + mi_term_.nTerms() + col_term_.nTerms() + jl_term_.nTerms() + kpts_term_.nTerms(); }

    const Skeleton &sk_ ;

    ContextGPU &ctx_ ;
    ImageToModelTermGPU im_term_ ;
    ModelToImageTermGPU mi_term_ ;
    CollisionTermGPU col_term_ ;
    JointLimitsTerm jl_term_ ;
    std::unique_ptr<KeyPointsDistanceField> kp_df_ ;
    KeyPoints2DTerm kpts_term_ ;
    const HumanModelFit::Parameters &params_ ;
};


Pose HumanModelFitImpl::fit(const std::vector<Vector3f> &cloud, const cv::Mat &mask, const PinholeCamera &cam,
                            KeyPointsDistanceField *kpts, const Pose &orig)
{
    ContextGPU ctx(mesh_.skeleton_) ;

    ctx.setPointCloud(cloud) ;
    ctx.computeDistanceTransform(mask.size());

    ObjectiveFunctionNLS obj_func(params_, ctx, mesh_.skeleton_, sdf_, mesh_, cam, cd_, kpts) ;

    Solver<float, ObjectiveFunctionNLS> lm ;

    VectorXf result = orig.coeffs() ;

    lm.minimize(obj_func, result);

    Pose pose(&mesh_.skeleton_, result) ;

    return pose ;
}

Pose HumanModelFitImpl::fit(const cv::Mat &im, const cv::Mat &mask, const PinholeCamera &cam, KeyPointsDistanceField *kpts,
                            const Pose &orig)
{
    ContextGPU ctx(mesh_.skeleton_) ;

    if ( cp_.empty() )
        ctx.setPointCloud(im, mask, cam) ;
    else
        ctx.setPointCloudClipped(im, cp_, mask, cam) ;

    ctx.computeDistanceTransform(im.size());

    ObjectiveFunctionNLS obj_func(params_, ctx, mesh_.skeleton_, sdf_,  mesh_, cam, cd_, kpts) ;

    Solver<float, ObjectiveFunctionNLS> lm ;

    VectorXf result = orig.coeffs() ;

    lm.minimize(obj_func, result, true);

    Pose pose(&mesh_.skeleton_, result) ;

    return pose ;
}


HumanModelFit::HumanModelFit(const Parameters &cfg)
{
    impl_.reset(new HumanModelFitImpl(cfg)) ;
}

HumanModelFit::~HumanModelFit()
{

}

const Skeleton &HumanModelFit::skeleton() const
{
    return impl_->mesh_.skeleton_ ;
}

void HumanModelFit::setClippingPlanes(const std::vector<Plane> &planes) {
    impl_->setClippingPlanes(planes);
}

Pose HumanModelFit::fit(const std::vector<Eigen::Vector3f> &cloud, const cv::Mat &mask, const cvx::PinholeCamera &cam,
                        KeyPointsDistanceField *kpts, const Pose &orig)
{
    return impl_->fit(cloud, mask, cam, kpts, orig) ;
}

Pose HumanModelFit::fit(const cv::Mat &im, const cv::Mat &mask, const cvx::PinholeCamera &cam,
                        KeyPointsDistanceField *kpts, const Pose &orig)
{
    return impl_->fit(im, mask, cam, kpts, orig) ;
}
