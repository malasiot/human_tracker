#include <htrac/pose/human_detector.hpp>
#include <htrac/pose/dataset.hpp>

#include <cvx/misc/path.hpp>
#include <cvx/math/rng.hpp>
#include <cvx/imgproc/rgbd.hpp>

#include <htrac/util/io_util.hpp>
#include <htrac/util/pcl_util.hpp>
#include <htrac/model/primitive_sdf.hpp>
#include <htrac/util/mhx2_importer.hpp>
#include "../image_to_model_term_gpu.hpp"
#include "../model_to_image_term_gpu.hpp"
#include "../collision_term_gpu.hpp"

#include <cvx/math/solvers/lbfgs.hpp>
#include <cvx/math/solvers/gradient_descent.hpp>

#include <cvx/misc/format.hpp>

#include "../solver.hpp"

using namespace cvx ;
using namespace std ;
using namespace Eigen ;

PrimitiveSDF *createModel(const Skeleton &sk) {
    PrimitiveSDF *pmesh = new PrimitiveSDF() ;

    pmesh->addPrimitive(sk.getBoneIndex("LeftForeArm"), new RoundCone(0.2, 0.051, 0.045)) ;
    pmesh->addPrimitive(sk.getBoneIndex("RightForeArm"), new RoundCone(0.2, 0.051, 0.045)) ;
    pmesh->addPrimitive(sk.getBoneIndex("LeftArm"), new RoundCone(0.2, 0.051, 0.045)) ;
    pmesh->addPrimitive(sk.getBoneIndex("RightArm"), new RoundCone(0.2, 0.051, 0.045)) ;
    pmesh->addPrimitive(sk.getBoneIndex("Head"), new ScaledPrimitive(new Sphere(0.12), {0.7, 1.0, 1.0}));

    pmesh->addPrimitive(sk.getBoneIndex("LeftUpLeg"), new RoundCone(0.3, 0.08, 0.07)) ;
    pmesh->addPrimitive(sk.getBoneIndex("RightUpLeg"), new RoundCone(0.3, 0.08, 0.07)) ;

    pmesh->addPrimitive(sk.getBoneIndex("LeftLeg"), new RoundCone(0.3, 0.07, 0.06)) ;
    pmesh->addPrimitive(sk.getBoneIndex("RightLeg"), new RoundCone(0.3, 0.07, 0.06)) ;

    Isometry3f btr ;
    btr.setIdentity() ;
    btr.translate(Vector3f{0, 0, 0.05}) ;
    btr.rotate(AngleAxisf(M_PI/15, Vector3f::UnitX())) ;
    pmesh->addPrimitive(sk.getBoneIndex("Spine"), new TransformedPrimitive(new Box({0.12, 0.25, 0.05}, 0.05), btr)) ;

    return pmesh ;
}

void test_im_term(const Skeleton &sk, const PrimitiveSDF &model, const vector<Vector3f> &cloud) {
    ContextGPU ctx(sk) ;

    ctx.setPointCloud(cloud) ;

    ImageToModelTermGPU im_term(ctx, sk, model, 100.0f) ;

    Pose pose(&sk), posep(&sk) ;
    float delta = 0.0001 ;

    VectorXf base(4) ;
    //base << 1.0, 0.3, 0 ;
    base << 0, 0, 0, 1 ;

    VectorXf basep(base) ;
    basep.x() += delta ;

    //  pose.setGlobalRotationParams(base) ;
    pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
    //   pose.setBoneParams("LeftLeg", {M_PI/6});



    float e = im_term.energy(pose) ;
    VectorXf g ;
    tie(e, g) = im_term.energyGradient(pose);
    cout << e << ' ' << g.adjoint() << endl ;

}


void test_mi_term(const SkinnedMesh &mesh, const cv::Mat &im, const cvx::PinholeCamera &cam) {
    ContextGPU ctx(mesh.skeleton_) ;
    ctx.computeDistanceTransform(im.size());

    ModelToImageTermGPU mi_term(ctx, mesh, cam) ;


    Pose pose(&mesh.skeleton_), posep(&mesh.skeleton_) ;

    float delta = 0.00001 ;

    VectorXf base(4) ;
    //base << 1.0, 0.3, 0 ;
    base << 0, 0, 0, 1 ;

    VectorXf basep(base) ;
 //   basep.x() += delta ;

    pose.setGlobalRotationParams(base) ;
    pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
 //   pose.setBoneParams("LeftLeg", {M_PI/6});

    posep.setGlobalTranslation(Vector3f{0 , -1.3, -3.0+delta});
    posep.setGlobalRotationParams(base) ;
 //  posep.setBoneParams("LeftLeg", {M_PI/6+delta});


  //  pose.setBoneParams("RightArm", { M_PI/6 } ) ;
 //  posep.setBoneParams("RightArm", { M_PI/6 + delta}) ;

  //  mesh.toMesh(pose).write("/tmp/mesh.obj");

    float e = mi_term.energy(pose) ;
    float ep = mi_term.energy(posep) ;

    cout << e << ' ' << ep << ' ' << (ep - e)/delta << endl ;

    VectorXf g ;
    tie(e, g) = mi_term.energyGradient(pose) ;

    cout << e << ' ' << g.adjoint() << endl ;

}


void test_collision_term(const Skeleton &sk, const CollisionData &cd) {
    ContextGPU ctx(sk) ;

    CollisionTermGPU col_term(ctx, sk, cd) ;

    Pose pose(&sk), posep(&sk) ;

    float delta = 0.00001 ;

    VectorXf base(4) ;
    //base << 1.0, 0.3, 0 ;
    base << 0, 0, 0, 1 ;

    VectorXf basep(base) ;
    basep.x() += delta ;

    pose.setGlobalRotationParams(base) ;
    pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
    pose.setBoneParams("RightArm", {-M_PI/6});

    posep.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
    posep.setGlobalRotationParams(base) ;
    posep.setBoneParams("RightArm", {static_cast<float>(-M_PI/6)+delta});


    pose.setBoneParams("RightForeArm", { M_PI/1.5, -M_PI/5, 0, 1 } ) ;
   posep.setBoneParams("RightForeArm", { M_PI/1.5, -M_PI/5, 0, 1}) ;

   sk.toMesh(pose).write("/tmp/skeleton.obj") ;

    ctx.computeTransforms(pose) ;
    float e = col_term.energy(pose) ;
    ctx.computeTransforms(posep) ;
    float ep = col_term.energy(posep) ;

    cout << e << ' ' << ep << ' ' << (ep - e)/delta << endl ;

    VectorXf g ;
    ctx.computeTransforms(pose) ;
    ctx.computeDerivatives(pose);
    tie(e, g) = col_term.energyGradient(pose) ;

    cout << e << ' ' << g.adjoint() << endl ;

}

struct ObjectiveFunction {

    ObjectiveFunction(const Skeleton &sk, const PrimitiveSDF &model, const std::vector<Vector3f> &cloud,
                      const SkinnedMesh &mesh, const cvx::PinholeCamera &cam, const cv::Mat &mask,
                      const Pose &orig):
        ctx_(sk), sk_(sk), im_term_(ctx_, sk, model, 0.5), mi_term_(ctx_, mesh, cam) {


    }

    float value(const VectorXf &x) {
        Pose pose(&sk_, x) ;
        float e1 = im_term_.energy(pose) ;
        float e2 = mi_term_.energy(pose) ;
        return ( lambda_im_ * e1 + lambda_mi_ * e2 );
    }

    void  gradient(const VectorXf &x, VectorXf &g) {
        Pose pose(&sk_, x) ;
        float e1, e2 ;
        VectorXf grad1, grad2 ;
        tie(e1, grad1) = im_term_.energyGradient(pose) ;
        tie(e2, grad2) = mi_term_.energyGradient(pose) ;
        g = ( lambda_im_ * grad1 + lambda_mi_ * grad2 ) ;
    }

    const Skeleton &sk_ ;
    ContextGPU ctx_ ;
    ImageToModelTermGPU im_term_ ;
    ModelToImageTermGPU mi_term_ ;

    const float lambda_im_ = 1.0 ;
    const float lambda_mi_ = 0.1 ;
    const float g_scale_ = 1.0e4 ;
};

void optimize(const Skeleton &sk, const PrimitiveSDF &model, const std::vector<Vector3f> &cloud,
               const SkinnedMesh &mesh, const cvx::PinholeCamera &cam, const cv::Mat &mask,
              const Pose &init) {

    using Solver = LBFGSSolver<float, ObjectiveFunction> ;
    Solver::Parameters params ;

    params.max_iter_ = 100 ;
    params.ls_.max_fev_ = 15;
    // params.rate_ = 0.0001 ;
    //   params.ls_.max_fev_ = 5 ;
    //  params.x_tol_ = 1.0e-15 ;

    //   params.x_tol_ = 0.01 ;
    Solver solver(params)  ;

    ObjectiveFunction of(sk, model, cloud, mesh, cam, mask, init) ;

    VectorXf result = init.coeffs() ;
    solver.minimize(of, result, [&](const VectorXf &x, const VectorXf &g, float f, uint it) {
        cout << it << ' ' << f << ' ' << g.norm() << endl ;
        Pose pose(&sk, x) ;
        //    sk.toMesh(pose).write(cvx::format("/tmp/result{:03d}.obj", it)) ;
    }) ;


    Pose pose(&sk, result) ;

    sk.toMesh(pose).write("/tmp/result.obj") ;
}

struct ObjectiveFunctionNLS {
    ObjectiveFunctionNLS(ContextGPU &ctx,
            const Skeleton &sk, const PrimitiveSDF &model,
                         const SkinnedMesh &mesh, const cvx::PinholeCamera &cam, const cv::Mat &mask,
                         const Pose &orig):
        ctx_(ctx), sk_(sk), mesh_(mesh), im_term_(ctx, sk, model, 0.5f), mi_term_(ctx, mesh, cam) {


    }

    void errors(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const {
        Pose pose(&sk_, x) ;

        ctx_.computeTransforms(pose) ;
        size_t n1 = ctx_.n_pts_ ;
        VectorXf fv1(n1) ;
        im_term_.energy(fv1) ;

        size_t n2 = mesh_.positions_.size() ;
        VectorXf fv2(n2) ;
        mi_term_.energy(fv2) ;
        fvec << fv1 * sqrt(lambda_im_), fv2 * sqrt(lambda_mi_);

    }

    // Compute the jacobian of the errors
    void jacobian(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const {
        Pose pose(&sk_, x) ;

        ctx_.computeTransforms(pose);
        ctx_.computeDerivatives(pose);

        size_t n1 = ctx_.n_pts_ ;
        Eigen::MatrixXf fj1(n1, fjac.cols()) ;
        im_term_.jacobian(fj1) ;
        fjac.block(0, 0, n1, fjac.cols()) = sqrt(lambda_im_) * fj1;

        size_t n2 = mesh_.positions_.size() ;
        Eigen::MatrixXf fj2(n2, fjac.cols()) ;
        mi_term_.jacobian(fj2) ;
        fjac.block(n1, 0, n2, fjac.cols()) = sqrt(lambda_mi_) * fj2;
    }

    void debug(uint iter, float eL2, const VectorXf &p) {
        cout << eL2 << endl ;
    }

    void norm(const Eigen::VectorXf &x, Eigen::MatrixXf &jtj, Eigen::VectorXf &jte) {
        Pose pose(&sk_, x) ;

        ctx_.computeTransforms(pose);
        ctx_.computeDerivatives(pose);

        jtj.setZero() ; jte.setZero() ;

        im_term_.norm(jtj, jte, lambda_im_) ;
        mi_term_.norm(jtj, jte, lambda_mi_) ;
    }

    int terms() const { return ctx_.n_pts_ + mesh_.positions_.size(); }

    const Skeleton &sk_ ;
    const SkinnedMesh &mesh_ ;
    ContextGPU &ctx_ ;
    ImageToModelTermGPU im_term_ ;
    ModelToImageTermGPU mi_term_ ;

    const float lambda_im_ = 1.0 ;
    const float lambda_mi_ = 0.00001 ;


    const float g_scale_ = 1.0e4 ;
};

void optimizeNLS(const Skeleton &sk, const PrimitiveSDF &model, const std::vector<Vector3f> &cloud,
                 const SkinnedMesh &mesh, const cvx::PinholeCamera &cam, const cv::Mat &mask,
                 const Pose &init) {

    ContextGPU ctx(sk) ;

    ctx.setPointCloud(cloud);
    ctx.computeDistanceTransform(mask.size());
    ObjectiveFunctionNLS obj_func(ctx, sk, model,  mesh, cam, mask, init) ;

    using solver_t = Solver<float, ObjectiveFunctionNLS> ;
    solver_t::Parameters params ;
    params.f_tol_ = 1.0e-3 ;
    solver_t lms(params) ;

     VectorXf result = init.coeffs() ;

   lms.minimize(obj_func, result, true);



    Pose pose(&sk, result) ;

    sk.toMesh(pose).write("/tmp/result.obj") ;
}

int main(int argc, const char* argv[]) {

    ITOP dataset("/home/malasiot/source/human_tracking/data/datasets/ITOP/train") ;

    cv::Mat img = dataset.getDepthImage(130) ;
    cv::imwrite("/tmp/dim.png", cvx::depthViz(img)) ;
    auto cam = dataset.getCamera() ;

    HumanDetector detector ;

    Plane plane(Vector3f{0, 1, 0}, 1.2) ;
    detector.setGroundPlane(plane);

    Polytope *ocs = new Polytope{{
            Plane{{0, 0, 1}, 0.05}, // bottom
            Plane{{0, 0, -1}, 2.0},  // top
            Plane{{0, -1, 0}, 3.5},  // back
            Plane{{1, 0, 0}, 1.0}, // left
            Plane{{-1, 0, 0}, 0.8} // rigtht
}};
    detector.setOcclusionSpace(ocs) ;

    cv::Rect box ;
    cv::Mat mask ;

    detector.detect(img, cam, box, mask);

    vector<Vector3f> cloud = depthToPointCloud(img, cam, mask, 2) ;

    save_point_cloud(cloud, "/tmp/cloud.obj");

    SkinnedMesh mesh ;
    mesh.load("/home/malasiot/source/human_tracking/data/models/human-cmu-low-poly.mhx2") ;

    auto &sk = mesh.skeleton_ ;

    PrimitiveSDF *sdf = createModel(sk) ;
    std::shared_ptr<SDFModel> model(sdf) ;

    PoseParameterization solve_bones ;
    solve_bones.emplace("Neck", new QuaternionParameterization) ;
    solve_bones.emplace("LeftArm", new QuaternionParameterization) ;
    solve_bones.emplace("LeftForeArm", new Flexion(RotationAxis::X, -0.785, 2)) ;
    solve_bones.emplace("RightArm", new QuaternionParameterization ) ;
    solve_bones.emplace("RightForeArm", new Flexion(RotationAxis::X, -0.785, 2)) ;
    solve_bones.emplace("LeftLeg", new Flexion(RotationAxis::X, 0, 2.4)) ;
    solve_bones.emplace("RightLeg", new Flexion(RotationAxis::X, 0, 2.4)) ;
    solve_bones.emplace("LeftUpLeg", new FlexionAbduction(RotationAxis::X, RotationAxis::Z, -2.6, 1.6, -1.5, 0.4)) ;
    solve_bones.emplace("RightUpLeg", new FlexionAbduction(RotationAxis::X, RotationAxis::Z, -2.6, 1.6, -0.4, 1.5)) ;


    sk.setPoseBones(solve_bones);

    CollisionData cd ;
    cd.parseJson(sk, "/home/malasiot/source/human_tracking/data/collision.json") ;

  //  test_im_term(sk, *sdf, cloud);
    test_mi_term(mesh, mask, cam);

   // test_collision_term(sk, cd);

    Pose pose(&sk) ;
   pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
    pose.setGlobalRotation(Quaternionf{1, 0, 0.3, 0}) ;
  //  optimize(sk, model, cloud, mesh, cam, mask, pose) ;

    optimizeNLS(sk, *sdf, cloud, mesh, cam, mask, pose) ;

}
