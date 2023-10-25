#include <human_tracker/pose_estimator_sdf.hpp>

#include <cvx/util/misc/path.hpp>
#include <cvx/util/math/solvers/bfgs.hpp>
#include <cvx/util/math/solvers/lm.hpp>
#include <cvx/util/math/solvers/gradient_descent.hpp>
#include <cvx/util/geometry/kdtree.hpp>
#include <cvx/util/geometry/octree.hpp>
#include <cvx/util/imgproc/rgbd.hpp>

#include "image_to_model_term.hpp"
#include "model_to_image_term.hpp"

#include "image_to_model_term_gpu.hpp"
#include "model_to_image_term_gpu.hpp"

#include <fstream>

using namespace std ;
using namespace cvx::util ;
using namespace Eigen ;

using namespace std ;
using namespace Eigen ;

class SDFObjectiveFunctionCPU {
public:

    SDFObjectiveFunctionCPU(const SkinnedMesh &mesh, const HumanPoseEstimatorSDF::Parameters &params): bones_(params.bones_) {
        im_.reset(new ImageToModelTerm(mesh, params.bones_, params.outlier_threshold)) ;
        mi_.reset(new ModelToImageTerm(mesh, params.bones_, 0.01, 0.5)) ;
    }

    double value(const VectorXd &x) {
        Pose pose = Pose::unpack(x, bones_) ;

        double e ;
        double v_im = im_->energy(pose) ;
        e = v_im ;

        if ( lambda_mi_ > 0 ) {
            double v_mi = mi_->energy(pose) ;
            e += lambda_mi_ * v_mi ;
        }

        cout << e  << endl ;
        return e;
    }

    void gradient(const VectorXd &x, VectorXd &g) {
        Pose pose = Pose::unpack(x, bones_) ;

        VectorXd grad_im(g.size()) ;
        im_->energy_gradient(pose, x, grad_im) ;
        g = grad_im ;

        if ( lambda_mi_ > 0 ) {
            VectorXd grad_mi(g.size()) ;
            mi_->energy_gradient(pose, x, grad_mi) ;
            g += lambda_mi_ * grad_mi ;
        }

    }

    float lambda_mi_ = 0.0, lambda_c_ ;

public:
    unique_ptr<ImageToModelTerm> im_ ;
    unique_ptr<ModelToImageTerm> mi_ ;

    const vector<string> &bones_ ;
};



class SDFObjectiveFunctionGPU {
public:

    SDFObjectiveFunctionGPU(const SkinnedMesh &mesh, const HumanPoseEstimatorSDF::Parameters &params): mesh_(mesh), bones_(params.bones_) {
        im_.reset(new ImageToModelTermGPU(mesh, params.bones_, params.outlier_threshold)) ;
   //     mi_.reset(new ModelToImageTermGPU(mesh, params.bones_, 0.01, 0.5)) ;
    }

    double value(const VectorXd &x) {
        Pose pose = Pose::unpack(x, bones_) ;

        // compute derivatives and objective function

        EVMatrix4f transforms, itransforms, imat ;

        mesh_.skeleton_.computeBoneTransforms(pose, transforms) ;

        // compute inverse transforms

        itransforms.resize(transforms.size()) ;
        imat.resize(transforms.size()) ;

        //#pragma omp parallel for
        for(uint i=0 ; i<transforms.size() ; i++ ) {
            Matrix4f offset = mesh_.skeleton_.bones_[i].offset_ ;
            itransforms[i] = transforms[i].inverse().eval() ;
            imat[i] = offset * itransforms[i];
        }

        im_->reset_optimizer_transforms(transforms, itransforms, imat) ;



        double e ;
        double v_im = im_->energy(pose) ;
        e = v_im ;
#if 0
        if ( lambda_mi_ > 0 ) {
            double v_mi = mi_->energy(pose) ;
            e += lambda_mi_ * v_mi ;
        }
#endif
        cout << e  << endl ;
        return e;
    }

    void gradient(const VectorXd &x, VectorXd &g) {
        Pose pose = Pose::unpack(x, bones_) ;

        // compute derivatives and objective function

        EVMatrix4f transforms, itransforms, imat ;

        mesh_.skeleton_.computeBoneTransforms(pose, transforms) ;

        // compute inverse transforms

        itransforms.resize(transforms.size()) ;
        imat.resize(transforms.size()) ;

        //#pragma omp parallel for
        for(uint i=0 ; i<transforms.size() ; i++ ) {
            Matrix4f offset = mesh_.skeleton_.bones_[i].offset_ ;
            itransforms[i] = transforms[i].inverse().eval() ;
            imat[i] = offset * itransforms[i];
        }

        im_->reset_optimizer_transforms(transforms, itransforms, imat) ;

        et_.compute_transform_derivatives(mesh_.skeleton_, bones_, pose, x) ;

        im_->reset_optimizer_derivatives(et_.der_, et_.gder_) ;

        VectorXd grad_im(g.size()) ;
        im_->energy_gradient(pose, grad_im) ;


        g = grad_im ;
#if 0
        if ( lambda_mi_ > 0 ) {
            VectorXd grad_mi(g.size()) ;
            mi_->energy_gradient(pose, x, grad_mi) ;
            g += lambda_mi_ * grad_mi ;
        }
#endif
    }

    float lambda_mi_ = 0.0, lambda_c_ ;

public:
    const SkinnedMesh &mesh_ ;
    unique_ptr<ImageToModelTermGPU> im_ ;
 //   unique_ptr<ModelToImageTermGPU> mi_ ;

    const vector<string> &bones_ ;
    EnergyTerm et_ ;
};

class ICPObjFunc {
public:

    ICPObjFunc(const Skeleton &sk, const map<string, Vector4f> &joints, const std::vector<std::string> &bones): skeleton_(sk),
        joints_(joints), bones_(bones) { }

    size_t terms() const { return 3 * joints_.size() ; } // should return the number of terms in least squares summation (for example if it is 2D curve fitting it is 2 times the number of points)

    void errors(const VectorXd &x, VectorXd &f)  // the error computed for all terms given the parameter vector
    {
        Pose pose = Pose::unpack(x, bones_) ;

        float e = 0 ;
        vector<Vector3f> joints ;
        skeleton_.getGeometry(pose, joints);

        int k=0 ;
        for ( const auto &lp: joints_ ) {
            const string &joint_name = lp.first ;
            const Vector4f &p = lp.second ;
            float weight = p[3] ;

            uint jidx = skeleton_.getBoneIndex(joint_name) ;

            const Vector3f &v = joints[jidx] ;

            f[k++] = weight * (v.x() - p.x()) ;
            f[k++] = weight * (v.y() - p.y()) ;
            f[k++] = weight * (v.z() - p.z()) ;

            e += weight * ( ( v.x() - p.x() )*(v.x() - p.x() ) + ( v.y() - p.y() )*(v.y() - p.y()) + ( v.z() - p.z() )*(v.z() - p.z())) ;
        }

        cout << e << endl;
    }

    // the jacobian of the errors (terms x num_params) if analytic derivates are used
    void jacobian(const VectorXd &x, MatrixXd &jac) {
        Pose pose = Pose::unpack(x, bones_) ;



    }


private:
    const std::vector<std::string> &bones_ ;
    const Skeleton &skeleton_ ;
    const map<string, Vector4f> &joints_ ;

};

HumanPoseEstimatorSDF::HumanPoseEstimatorSDF(const SkinnedMesh &mesh, const HumanPoseEstimatorSDF::Parameters &params): mesh_(mesh), params_(params) {
    if ( params.use_gpu )
        of_gpu_.reset(new SDFObjectiveFunctionGPU(mesh, params_)) ;
    else
        of_.reset(new SDFObjectiveFunctionCPU(mesh, params_)) ;
}

HumanPoseEstimatorSDF::~HumanPoseEstimatorSDF()
{

}


bool HumanPoseEstimatorSDF::init(const std::string &data_path) {
    Path model_file(data_path, params_.model_filename_) ;

    if ( !model_file.exists() ) return false ;

    shared_ptr<SDFModel> model(new SDFModel()) ;

    ifstream strm( model_file.toString(), ios::binary) ;
    IBinaryStream ar(strm) ;

    ar >> *model ;

    if ( params_.use_gpu )
        of_gpu_->im_->init(*model) ;
    else
        of_->im_->init(model) ;
}

void HumanPoseEstimatorSDF::initFromSkeleton(const std::map<string, Vector4f> &data, Pose &p)
{
    ICPObjFunc of(mesh_.skeleton_, data, params_.bones_) ;

    LMSolver<double, ICPObjFunc>::Parameters solver_params ;
    solver_params.delta_ = 1.0e-5 ;
    solver_params.max_iter_ = 1000 ;

    LMSolver<double, ICPObjFunc> solver(solver_params) ;

    VectorXd solution = p.pack(params_.bones_) ;
    solver.minimizeDiff(of, solution) ;
    p = p.unpack(solution, params_.bones_) ;
}

void HumanPoseEstimatorSDF::estimate(const std::vector<Eigen::Vector3f> &coords, Pose &p) {



    if ( params_.use_gpu ) {
        //   of_->mi_->initSDF(p, im, cam) ;
        of_gpu_->im_->setImageCoords(coords) ;

        BFGSSolver<double, SDFObjectiveFunctionGPU>::Parameters solver_params ;
        solver_params.max_iter_ = params_.opt_max_iterations ;
        solver_params.ls_.max_fev_ = params_.opt_linesearch_iterations;

        BFGSSolver<double, SDFObjectiveFunctionGPU> solver(solver_params) ;

        VectorXd solution = p.pack(params_.bones_) ;
        solver.minimize(*of_gpu_, solution) ;

        p = Pose::unpack(solution, params_.bones_) ;
    }
    else {
        of_->im_->setImageCoords(std::move(coords)) ;

        BFGSSolver<double, SDFObjectiveFunctionCPU>::Parameters solver_params ;
        solver_params.max_iter_ = params_.opt_max_iterations ;
        solver_params.ls_.max_fev_ = params_.opt_linesearch_iterations;

        BFGSSolver<double, SDFObjectiveFunctionCPU> solver(solver_params) ;

        VectorXd solution = p.pack(params_.bones_) ;
        solver.minimize(*of_, solution) ;

        p = Pose::unpack(solution, params_.bones_) ;

    }



}
