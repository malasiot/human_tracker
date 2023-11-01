#include <htrac/util/mhx2_importer.hpp>
#include <htrac/model/skeleton.hpp>
#include <unsupported/Eigen/NonLinearOptimization>
#include <iostream>
#include <fstream>

#include <ceres/solver.h>
#include <ceres/problem.h>
#include <ceres/cost_function.h>


#include <cvx/math/rng.hpp>

using namespace std ;
using namespace Eigen ;

using Joints = map<string, Vector3f>;

class Problem
{
public:
    Problem(const Skeleton &sk): skeleton_(sk) {}

    void solve(const Pose &orig, const std::map<std::string, Vector3f> &coords, Pose &result, ceres::Solver::Options &opt);

protected:

    const Skeleton &skeleton_ ;
    ceres::Problem problem_;
};


class CostFunction: public ceres::CostFunction {
public:
    CostFunction(const Skeleton &sk, const std::string &jname, const Vector3f &obs):
        skeleton_(sk), observation_(obs), jname_(jname) {
        set_num_residuals(3);

        mutable_parameter_block_sizes()->push_back(3) ;
        mutable_parameter_block_sizes()->push_back(Pose::global_rot_params) ;

        for( const auto &pb: sk.getPoseBones() ) {
            mutable_parameter_block_sizes()->push_back(pb.dofs());
        }
    }

    Eigen::VectorXf pack(const double *const *params) const {
        uint n_global = 3 + Pose::global_rot_params;
        size_t n_params = skeleton_.getNumPoseBoneParams() + n_global ;
        Eigen::VectorXf X(n_params) ;

        const auto &pbv = skeleton_.getPoseBones() ;
        for( uint i=0 ; i<pbv.size() ; i++ ) {
            const auto &pb = pbv[i] ;

            for( int k=0 ; k<pb.dofs() ; k++ )
                X[n_global + pb.offset() + k] = params[i+2][k] ;
        }

        for( int k=0 ; k<3 ; k++ )
            X[k] = params[0][k] ;

        for( int k=0 ; k<Pose::global_rot_params ; k++ )
            X[k+3] = params[1][k] ;

        return X ;

    }

    bool Evaluate(const double * const *parameters, double *residuals, double **jacobians) const {

        VectorXf X = pack(parameters) ;
        Pose pose(&skeleton_, X) ;

        auto tr = skeleton_.computeBoneTransform(pose, jname_) ;
        Vector3f coords = tr.block<3, 1>(0, 3) ;
        residuals[0] = coords.x() - observation_.x() ;
        residuals[1] = coords.y() - observation_.y() ;
        residuals[2] = coords.z() - observation_.z() ;

      //  jacobians[i][r*parameter_block_size_[i] + c] =
          //                              d residual[r] / d parameters[i][c]


        if ( jacobians == nullptr ) return true ;

        const auto &pbv = skeleton_.getPoseBones() ;
        for( uint i=0 ; i<pbv.size() ; i++ ) {
            const auto &pb = pbv[i] ;

            Matrix4f dr[4] ;
            size_t dim  = pb.dofs() ;
            skeleton_.computeBoneRotationDerivatives(pose, jname_, pb.name(), dr);
            for( int k=0 ; k<dim ; k++ ) {
               auto dj = dr[k].block<3, 1>(0, 3) ;
               for(int j=0 ;j<3 ; j++ )
                    jacobians[i+2][dim*j + k] =  dj[j];
            }
        }

        Matrix4f dt[3], dr[4] ;
        skeleton_.computeGlobalDerivatives(pose, jname_, dt, dr, false);

        for( int k=0 ;k<3 ; k++ ) {
            auto dj = dt[k].block<3, 1>(0, 3) ;
            for(int j=0 ;j<3 ; j++ )
                 jacobians[0][3*j + k] =  dj[j];
        }

        for( int k=0 ;k<Pose::global_rot_params ; k++ ) {
            auto dj = dr[k].block<3, 1>(0, 3) ;
            for(int j=0 ;j<3 ; j++ )
                 jacobians[1][Pose::global_rot_params*j + k] =  dj[j];
        }

        return true ;
    }

private:
    const Skeleton &skeleton_ ;
    Eigen::Vector3f observation_ ;
    std::string jname_ ;
};

#define DEBUG 1
void Problem::solve(const Pose &orig, const std::map<string, Vector3f> &coords, Pose &result, ceres::Solver::Options &options)
{
    size_t n_params = orig.coeffs().size();
    std::unique_ptr<double []> params(new double[n_params]) ;
    vector<double *> pblocks ;

    Eigen::VectorXd::Map(params.get(), n_params) = orig.coeffs().cast<double>() ;

    problem_.AddParameterBlock(params.get(), 3);
    pblocks.push_back(params.get()) ;

    double *block = params.get() + 3 ;
    problem_.AddParameterBlock(block, Pose::global_rot_params);
    pblocks.push_back(block) ;

    const auto &pbv = skeleton_.getPoseBones() ;
    for( size_t i=0 ; i<pbv.size() ; i++ ) {
        const auto &pb = pbv[i] ;

        int offset = + 3 + Pose::global_rot_params + pb.offset() ;
        double *block = params.get() + offset ;
        problem_.AddParameterBlock(block, pb.dofs()) ;
        if ( pb.rp_->hasLimits() ) {
            VectorXf limits(pb.dofs() * 2) ;
            pb.rp_->getLimits(limits) ;
            for ( int i=0 ; i<pb.dofs() ; i++ ) {
                 problem_.SetParameterLowerBound(block, i, limits[2*i]);
                 problem_.SetParameterUpperBound(block, i, limits[2*i+1]);
            }
        }

        pblocks.push_back(block) ;
    }

    for ( const auto &c: coords )  {
        CostFunction* cost_function = new CostFunction(skeleton_, c.first, c.second);
        problem_.AddResidualBlock(cost_function, nullptr, pblocks);
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem_, &summary);

   // result = SkeletonParameters( Eigen::VectorXd::Map(params.get(), n_params).cast<float>() ).getPose(bones_) ;

    result.setCoeffs(Eigen::VectorXd::Map(params.get(), n_params).cast<float>()) ;

#ifdef DEBUG
    std::cout << summary.FullReport() << "\n";
#endif
}

Pose Skeleton::fit(const Skeleton &sk,
                   const std::map<std::string, Eigen::Vector3f> &joints,
                   const Pose &orig, float thresh) {

       ceres::Solver::Options options;
       options.linear_solver_type = ceres::DENSE_SCHUR;
#ifdef DEBUG
       options.minimizer_progress_to_stdout = true;
#endif
       options.max_num_iterations = 50;
       options.function_tolerance = thresh ;
       Problem p(sk) ;

       Pose result(orig.getSkeleton()) ;
       p.solve(orig, joints, result, options);

       return result ;

}



