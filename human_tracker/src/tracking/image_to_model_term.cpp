#include "image_to_model_term.hpp"

#include <iostream>
#include <fstream>


using namespace std ;
using namespace Eigen ;

static const double g_scale_factor = 1.0e3 ;

static const uint OUTLIER_LABEL = 1000 ;

double ImageToModelTerm::energy(const Pose &pose) {

    // compute derivatives and objective function

    vector<Matrix4f> transforms, imat ;

    skeleton_.computeBoneTransforms(pose, transforms) ;

    // compute inverse transforms

    imat.resize(transforms.size()) ;

#pragma omp parallel for
    for(uint i=0 ; i<transforms.size() ; i++ )     {
        imat[i] = transforms[i].inverse() ;
    }

    uint N = icoords_.size() ;

    // assign pixels to bones

    MatrixXf trf = sdf_->transform_points_to_bone_space(icoords_, imat) ;
    MatrixXf distances = sdf_->eval(trf) ;

    vector<uint>  labels(N);
    VectorXf mine(N) ;


#pragma omp parallel for
    for( uint i=0; i<N ; ++i ) {
        mine[i] = std::numeric_limits<float>::max() ;
        for( uint j=0 ; j<sdf_->getNumParts() ; j++ ) {
            float d = distances(j, i) ;
            if ( fabs(d) < fabs(mine[i]) ) {
                 mine[i] = d ;

                 labels[i] = j ;
            }
        }

        if ( fabs(mine[i]) > outlier_threshold_ ) {
            mine[i] = 0 ;
            labels[i] = OUTLIER_LABEL ;
        }
    }


    uint count = 0 ;
    for( uint i=0; i<N ; ++i ) if ( labels[i] != OUTLIER_LABEL ) count++ ;

    if ( count > 0 )
        return mine.squaredNorm()/count ;
    else
        return 1.0e9 ;
}


pair<double, VectorXd> ImageToModelTerm::energyGradient(const Pose &pose) {

    VectorXd diffE(skeleton_.getNumPoseBoneParams() + 3 + Pose::global_rot_params) ;

    diffE.setZero() ;

    vector<Matrix4f> bder, gder ;
    EnergyTerm::compute_transform_derivatives(skeleton_, pose, bder, gder) ;

    // compute derivatives and objective function
    vector<Matrix4f> transforms, imat ;

    skeleton_.computeBoneTransforms(pose, transforms) ;

    // compute inverse transforms

    imat.resize(transforms.size()) ;

#pragma omp parallel for
    for(uint i=0 ; i<transforms.size() ; i++ )     {
        imat[i] = transforms[i].inverse() ;
    }

    uint N = icoords_.size() ;

    // assign pixels to bones

    MatrixXf trf = sdf_->transform_points_to_bone_space(icoords_, imat) ;
    MatrixXf distances = sdf_->eval(trf) ;

    vector<uint>  labels(N);
    VectorXf mine(N);

#pragma omp parallel for
    for( uint i=0; i<N ; ++i ) {
        mine[i] = distances.col(i).cwiseAbs().minCoeff( &labels[i] );

        if ( mine[i] > outlier_threshold_ ) {
            mine[i] = 0 ;
            labels[i] = OUTLIER_LABEL ;
        }
    }

    uint count = 0 ;
    for( uint i=0; i<N ; ++i ) if ( labels[i] != OUTLIER_LABEL ) count++ ;

    // compute derivatives

    MatrixXf der = sdf_->grad(trf, labels) ;

   // compute energy gradient (parallel computation for each point)

    MatrixXd G(pose.coeffs().size(), N) ;
    G.setZero() ;


    const auto &pbv = skeleton_.getPoseBones() ;

    size_t n_pose_bones = pbv.size() ;
    size_t n_global_params = Pose::global_rot_params + 3 ;

#define IDX3(i, j, k) (n_pose_bones * 4 * (i) + 4 * (j) + (k))
#define IDX2(i, j) ( n_global_params * (i) + j)

#pragma omp parallel for
    for( uint i=0 ; i<N ; i++ )
    {
        uint part = labels[i] ;

        if ( part == OUTLIER_LABEL ) continue ;

        float ve = distances(part, i) ;


        uint idx = sdf_->getPartBone(part) ;


        const Matrix4f &t = imat[idx] ;
        const Vector3f &pt = icoords_[i] ;

        Vector3f grad = der.col(i) ;

        for(uint k=0 ; k<pbv.size() ; k++ ) {
            const auto &pb = pbv[k];

            for(uint r=0 ; r<pb.dofs() ; r++)
            {
                const Matrix4f &dQ = bder[IDX3(idx, k, r)] ;
                Matrix4f dG = - t * dQ * t ;
                Vector4f p =  dG * Vector4f(pt.x(), pt.y(), pt.z(), 1.0) ;
                Vector3f dp(p.x(), p.y(), p.z()) ;

                double de = grad.dot(dp) ;

                G(n_global_params + pb.offset() + r, i) = 2*ve*de ;
            }
        }

        for(uint r=0 ; r<3 + Pose::global_rot_params ; r++) //? no scale
        {
            const Matrix4f &dQ = gder[IDX2(idx, r)] ;
            Matrix4f dG = - t * dQ * t ;
            Vector4f p =  dG * Vector4f(pt.x(), pt.y(), pt.z(), 1.0) ;
            Vector3f dp(p.x(), p.y(), p.z()) ;

            double de = grad.dot(dp) ;

            G(r, i) = 2*ve*de ;
        }

    }

 //   cout << G << endl;

    if ( count > 0 ) {
        diffE = G.rowwise().sum()/count ;
//        return g_scale_factor*mine.squaredNorm()/count ;
    }
  //  else return g_scale_factor ;

    return make_pair(mine.squaredNorm()/count, diffE) ;
}

