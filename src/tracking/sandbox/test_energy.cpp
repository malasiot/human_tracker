#include <htrac/pose/human_detector.hpp>
#include <htrac/pose/dataset.hpp>

#include <cvx/misc/path.hpp>
#include <cvx/math/rng.hpp>
#include <cvx/imgproc/rgbd.hpp>

#include <htrac/util/io_util.hpp>
#include <htrac/util/pcl_util.hpp>
#include <htrac/model/primitive_sdf.hpp>
#include <htrac/util/mhx2_importer.hpp>
#include "../image_to_model_term.hpp"
#include "../model_to_image_term.hpp"
#include "../collision_term.hpp"
#include "../joint_limits_term.hpp"
#include "../keypoints_term.hpp"
#include "../keypoints_2d_term.hpp"

#include <cvx/math/solvers/lbfgs.hpp>
#include <cvx/math/solvers/gradient_descent.hpp>
#include <cvx/misc/format.hpp>

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

struct ObjectiveFunction {

    ObjectiveFunction(const Skeleton &sk, const std::shared_ptr<SDFModel> &model, const std::vector<Vector3f> &cloud,
                      const SkinnedMesh &mesh, const cv::Mat &mask, const cvx::PinholeCamera &cam, const CollisionData &cd):
        sk_(sk), im_term_(sk, model, 0.3), mi_term_(mesh, cam), col_term_(sk, cd, 0) {
        im_term_.setImageCoords(std::move(cloud));
        mi_term_.computeDistanceTransform(mask);
    }

    float value(const VectorXf &x) {
        Pose pose(&sk_, x) ;
        float e1 = im_term_.energy(pose) ;
        float e2 = mi_term_.energy(pose) ;
        float e3 = col_term_.energy(pose) ;

        cout << e3 << endl ;
 //       cout << e1 << '*' << e2 << endl ;
        // return g_scale_ * (e1 + lambda_ * e2) ;

        return g_scale_ * e3 ;
    }

    void  gradient(const VectorXf &x, VectorXf &g) {
        Pose pose(&sk_, x) ;
   //     double e1, e2 ;
   //     VectorXd grad1, grad2 ;
   //     tie(e1, grad1) = im_term_.energyGradient(pose) ;
   //     tie(e2, grad2) = mi_term_.energyGradient(pose) ;
   //     g = (g_scale_*(grad1 + lambda_ * grad2)).cast<float>();
        float e3 ;
        VectorXf grad3 ;
         tie(e3, grad3) = col_term_.energyGradient(pose) ;
         g = g_scale_ * grad3 ;
    }

    const Skeleton &sk_ ;
    ImageToModelTerm im_term_ ;
    ModelToImageTerm mi_term_ ;
    CollisionTerm col_term_ ;
    const float lambda_ = 0.0 ;
    const float g_scale_ = 1.0e8 ;
};

void optimize(const Skeleton &sk, const std::shared_ptr<SDFModel> &model, const std::vector<Vector3f> &cloud,
              const SkinnedMesh &mesh, const cv::Mat &mask, const cvx::PinholeCamera &cam, const CollisionData &cd, const Pose &init) {

    using Solver = LBFGSSolver<float, ObjectiveFunction> ;
    Solver::Parameters params ;

    params.max_iter_ = 100 ;
 //   params.rate_ = 0.000005 ;
    params.ls_.max_fev_ = 15 ;
  //  params.x_tol_ = 1.0e-15 ;

 //   params.x_tol_ = 0.01 ;
    Solver solver(params)  ;

    ObjectiveFunction of(sk, model, cloud, mesh, mask, cam, cd) ;

    VectorXf result = init.coeffs() ;
    solver.minimize(of, result, [&](const VectorXf &x, const VectorXf &g, float f, uint it) {
        cout << it << ' ' << f << ' ' << g.norm() << endl ;
         Pose pose(&sk, x) ;
        sk.toMesh(pose).write(cvx::format("/tmp/result{:03d}.obj", it)) ;
    }) ;


    Pose pose(&sk, result) ;

    sk.toMesh(pose).write("/tmp/result.obj") ;
}

void test_im_term(const Skeleton &sk, const shared_ptr<SDFModel> &model, const vector<Vector3f> &cloud) {
    ImageToModelTerm im_term(sk, model, 100.0) ;

    std::vector<Vector3f> ipts{ { 0.1, 1.0, 0 } } ;

    im_term.setImageCoords(std::move(cloud)) ;

    Pose pose(&sk), posep(&sk) ;
  float delta = 0.0001 ;

  VectorXf base(4) ;
  //base << 1.0, 0.3, 0 ;
  base << 0, 0, 0, 1 ;

  VectorXf basep(base) ;
  basep.x() += delta ;

   pose.setGlobalRotationParams(base) ;
    pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
 //   pose.setBoneParams("LeftLeg", {M_PI/6});

    posep.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
 posep.setGlobalRotationParams(basep) ;
 //  posep.setBoneParams("LeftLeg", {M_PI/6+delta});


  //  pose.setBoneParams("LeftArm", { M_PI/6 } ) ;
 //   posep.setBoneParams("LeftArm", { M_PI/6 + delta}) ;

    sk.toMesh(pose).write("/tmp/skeleton.obj") ;


    float e = im_term.energy(pose) ;
    float ep = im_term.energy(posep) ;

    cout << e << ' ' << ep << ' ' << (ep - e)/delta << endl ;

    VectorXd g ;
    tie(e, g) = im_term.energyGradient(pose);

    cout << e << ' ' << g.adjoint() << endl ;

}


void test_kp2_term(const Skeleton &sk, const cvx::PinholeCamera &cam) {

    Context ctx(sk) ;

    KeyPoints kpts ;
    kpts.emplace("RightHand", make_pair(Vector2f(100, 100), 1.0f)) ;

    KeyPointListDistanceField *df = new KeyPointListDistanceField(kpts) ;

    KeyPoints2DTerm kp_term(ctx, cam, df) ;

    Pose pose(&sk), posep(&sk) ;
  float delta = 0.0001 ;

  VectorXf base(4) ;
  //base << 1.0, 0.3, 0 ;
  base << 0, 0, 0, 1 ;

  VectorXf basep(base) ;
  basep.y() += delta ;

   pose.setGlobalRotationParams(base) ;
    pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
    pose.setBoneParams("RightArm", {M_PI/8, 0, 0, 1});
//
    posep.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
 posep.setGlobalRotationParams(base) ;
   posep.setBoneParams("RightArm", {M_PI/8, 0+delta, 0, 1});


  //  pose.setBoneParams("LeftArm", { M_PI/6 } ) ;
 //   posep.setBoneParams("LeftArm", { M_PI/6 + delta}) ;


    ctx.computeTransforms(pose);
    float e = kp_term.energy(pose) ;
    ctx.computeTransforms(posep);
    float ep = kp_term.energy(posep) ;

    cout << e << ' ' << ep << ' ' << (ep - e)/delta << endl ;

    VectorXf g ;
    ctx.computeTransforms(pose);
    ctx.computeDerivatives(pose);
    tie(e, g) = kp_term.energyGradient(pose);

    cout << e << ' ' << g.adjoint() << endl ;

}

void test_kp_term(const Skeleton &sk) {

    Context ctx(sk) ;

    KeyPoints3 kpts ;
    kpts.emplace("LeftHand", make_pair(Vector3f(0, 1.0, -2.0), 1.0f)) ;

    KeyPointsTerm kp_term(ctx, kpts) ;

    Pose pose(&sk), posep(&sk) ;
  float delta = 0.0001 ;

  VectorXf base(4) ;
  //base << 1.0, 0.3, 0 ;
  base << 0, 0, 0, 1 ;

  VectorXf basep(base) ;
  basep.x() += delta ;

   pose.setGlobalRotationParams(base) ;
    pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
    pose.setBoneParams("LeftArm", {M_PI/8, 0, 0, 1});

    posep.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
 posep.setGlobalRotationParams(base) ;
   posep.setBoneParams("LeftArm", {M_PI/8+delta, 0, 0, 1});


  //  pose.setBoneParams("LeftArm", { M_PI/6 } ) ;
 //   posep.setBoneParams("LeftArm", { M_PI/6 + delta}) ;


    ctx.computeTransforms(pose);
    float e = kp_term.energy(pose) ;
    ctx.computeTransforms(posep);
    float ep = kp_term.energy(posep) ;

    cout << e << ' ' << ep << ' ' << (ep - e)/delta << endl ;

    VectorXf g ;
    ctx.computeTransforms(pose);
    ctx.computeDerivatives(pose);
    tie(e, g) = kp_term.energyGradient(pose);

    cout << e << ' ' << g.adjoint() << endl ;

}

void test_mi_term(const SkinnedMesh &mesh, const cv::Mat &im, const cvx::PinholeCamera &cam) {
    ModelToImageTerm mi_term(mesh, cam) ;
    mi_term.computeDistanceTransform(im);

    Pose pose(&mesh.skeleton_), posep(&mesh.skeleton_) ;

    float delta = 0.00001 ;

    VectorXf base(4) ;
    //base << 1.0, 0.3, 0 ;
    base << 0, 0, 0, 1 ;

    VectorXf basep(base) ;
    basep.x() += delta ;

    pose.setGlobalRotationParams(base) ;
    pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
 //   pose.setBoneParams("LeftLeg", {M_PI/6});

    posep.setGlobalTranslation(Vector3f{0, -1.3, -3.0+delta});
    posep.setGlobalRotationParams(base) ;
 //  posep.setBoneParams("LeftLeg", {M_PI/6+delta});


  //  pose.setBoneParams("RightArm", { M_PI/6 } ) ;
 //  posep.setBoneParams("RightArm", { M_PI/6 + delta}) ;

    mesh.toMesh(pose).write("/tmp/mesh.obj");

    float e = mi_term.energy(pose) ;
    float ep = mi_term.energy(posep) ;

    cout << e << ' ' << ep << ' ' << (ep - e)/delta << endl ;

    VectorXd g ;
    tie(e, g) = mi_term.energyGradient(pose) ;

    cout << e << ' ' << g.adjoint() << endl ;

}

void test_collision_term(const Skeleton &sk, const CollisionData &cd) {
    CollisionTerm col_term(sk, cd, 1.0/0.01) ;

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
    posep.setBoneParams("RightArm", {-M_PI/6+delta});


    pose.setBoneParams("RightForeArm", { M_PI/1.5, -M_PI/5, 0, 1 } ) ;
   posep.setBoneParams("RightForeArm", { M_PI/1.5, -M_PI/5, 0, 1}) ;

   sk.toMesh(pose).write("/tmp/skeleton.obj") ;

    float e = col_term.energy(pose) ;
    float ep = col_term.energy(posep) ;

    cout << e << ' ' << ep << ' ' << (ep - e)/delta << endl ;

    VectorXf g ;
    tie(e, g) = col_term.energyGradient(pose) ;

    cout << e << ' ' << g.adjoint() << endl ;

}

void test_limits_term(const Skeleton &sk) {
    JointLimitsTerm limits_term(sk) ;

    Pose pose(&sk), posep(&sk) ;

    float delta = 0.00001 ;

    VectorXf base(4) ;
    //base << 1.0, 0.3, 0 ;
    base << 0, 0, 0, 1 ;

    VectorXf basep(base) ;
    basep.x() += delta ;

    pose.setGlobalRotationParams(base) ;
    pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
    pose.setBoneParams("LeftLeg", {-M_PI});

    posep.setGlobalTranslation(Vector3f{0, -1.3, -3.0+delta});
    posep.setGlobalRotationParams(base) ;
    posep.setBoneParams("LeftLeg", {-M_PI+delta});


  //  pose.setBoneParams("RightArm", { M_PI/6 } ) ;
 //  posep.setBoneParams("RightArm", { M_PI/6 + delta}) ;


    float e = limits_term.energy(pose) ;
    float ep = limits_term.energy(posep) ;

    cout << e << ' ' << ep << ' ' << (ep - e)/delta << endl ;

    VectorXf g ;
    tie(e, g) = limits_term.energyGradient(pose) ;

    cout << e << ' ' << g.adjoint() << endl ;

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
    cv::Mat mask;

    detector.detect(img, cam, box, mask);

    vector<Vector3f> cloud = depthToPointCloud(img, cam, mask, 2) ;

    save_point_cloud(cloud, "/tmp/cloud.obj");

    SkinnedMesh mesh ;
    mesh.load("/home/malasiot/source/human_tracking/data/models/human-cmu-low-poly.mhx2") ;

    auto &sk = mesh.skeleton_ ;
    //sk.fromMH(importer.getModel()) ;

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
    solve_bones.emplace("LeftUpLeg", new FlexionAbduction(RotationAxis::X, RotationAxis::Z, -2.6f, 1.6f, -1.5f, 0.4f)) ;
    solve_bones.emplace("RightUpLeg", new FlexionAbduction(RotationAxis::X, RotationAxis::Z, -2.6f, 1.6f, -0.4f, 1.5f)) ;

    sk.setPoseBones(solve_bones);



    CollisionData cd ;
    cd.parseJson(sk, "/home/malasiot/source/human_tracking/data/collision.json") ;

  //  test_im_term(sk, model, cloud);
  //  test_mi_term(mesh, mask, cam);
  //  test_collision_term(sk, cd) ;
    // test_limits_term(sk) ;
    test_kp2_term(sk, cam) ;

/*    Pose pose(&sk) ;
    pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
    pose.setGlobalRotation(Quaternionf::Identity()) ;
     pose.setBoneParams("RightArm", {-M_PI/6});
     pose.setBoneParams("RightForeArm", { M_PI/1.5, -M_PI/5, 0, 1 } ) ;

     sk.toMesh(pose).write("/tmp/skeleton.obj") ;
     */
 //   optimize(sk, model, cloud, mesh, mask, cam, cd, pose) ;

}
