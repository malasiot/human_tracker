#include "collision_term.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace Eigen ;

CollisionTerm::CollisionTerm(const Skeleton &sk, const CollisionData &cd, float gamma):
    skeleton_(sk), cd_(cd), gamma_(gamma) {

    for( size_t i=0 ; i<cd.spheres_.size() ; i++ ) {
        const auto &s1 = cd.spheres_[i].group_ ;
        for( size_t j=0 ; j<i ; j++) {
            const auto &s2 = cd.spheres_[j].group_ ;

            if ( s1 == s2 ) continue ;
            pairs_.emplace_back(i, j) ;
        }
    }
}

static float sigmoid(float x, float beta) {
    return 1.0/(1 + exp(-x * beta)) ;
}

float CollisionTerm::energy(const Pose &pose) {

    vector<Matrix4f> transforms ;

    skeleton_.computeBoneTransforms(pose, transforms) ;

    float total = 0.0 ;
    for (const auto &cp: pairs_ ) {
        uint idx1 = cp.first ;
        uint idx2 = cp.second ;

        uint bone_s = cd_.spheres_[idx1].bone_ ;
        uint bone_t = cd_.spheres_[idx2].bone_ ;

        const Matrix4f &bts = transforms[bone_s] ;
        const Matrix4f &btt = transforms[bone_t] ;

        const Vector3f &c0s = cd_.spheres_[idx1].c_ ;
        const Vector3f &c0t = cd_.spheres_[idx2].c_ ;

        const Vector3f cs = (bts * c0s.homogeneous()).head<3>() ;
        const Vector3f ct = (btt * c0t.homogeneous()).head<3>() ;

        float rs = cd_.spheres_[idx1].r_ ;
        float rt = cd_.spheres_[idx2].r_ ;

        Vector3f cst = cs - ct ;
        float g = cst.norm() ;
        float rst = (rs + rt)  ;
        float rstg = rst - g ;

        float e = std::max(0.f, rstg);

        total += e * e ;
    }
    return total/pairs_.size()  ;

}

std::pair<float, Eigen::VectorXf> CollisionTerm::energyGradient(const Pose &pose)
{

    uint N = pairs_.size() ;
    VectorXf diffE(skeleton_.getNumPoseBoneParams() + 3 + Pose::global_rot_params) ;

    diffE.setZero() ;

    vector<Matrix4f> bder, gder ;
    EnergyTerm::compute_transform_derivatives(skeleton_, pose, bder, gder) ;

    // compute derivatives and objective function
    vector<Matrix4f> transforms ;

    skeleton_.computeBoneTransforms(pose, transforms) ;

    const auto &pbv = skeleton_.getPoseBones() ;

    size_t n_pose_bones = pbv.size() ;
    size_t n_global_params = Pose::global_rot_params + 3 ;

#define IDX3(i, j, k) (n_pose_bones * 4 * (i) + 4 * (j) + (k))
#define IDX2(i, j) ( n_global_params * (i) + j)

    MatrixXf G(pose.coeffs().size(), N) ;
    G.setZero() ;

    float total = 0.0 ;
    for (uint i=0 ; i<N ; i++) {
        const auto &cp = pairs_[i] ;
        uint idx1 = cp.first ;
        uint idx2 = cp.second ;

        uint bone_s = cd_.spheres_[idx1].bone_ ;
        uint bone_t = cd_.spheres_[idx2].bone_ ;



        const Matrix4f &bts = transforms[bone_s] ;
        const Matrix4f &btt = transforms[bone_t] ;

        const Vector3f &c0s = cd_.spheres_[idx1].c_ ;
        const Vector3f &c0t = cd_.spheres_[idx2].c_ ;

        const Vector3f cs = (bts * c0s.homogeneous()).head<3>() ;
        const Vector3f ct = (btt * c0t.homogeneous()).head<3>() ;

        float rs = cd_.spheres_[idx1].r_ ;
        float rt = cd_.spheres_[idx2].r_ ;

        Vector3f cst = cs - ct ;
        float g = cst.norm() ;
        float rst = (rs + rt)  ;
        float rstg = rst - g ;
        Vector3f dg0 = cst / g ;

        float e = std::max(0.f, rstg) ;
        total += e * e ;

        if ( rstg > 0 ) {

            for(uint k=0 ; k<pbv.size() ; k++ ) {
                const auto &pb = pbv[k];

                for(uint r=0 ; r<pb.dofs() ; r++)  {

                    Matrix4f dQs = bder[IDX3(bone_s, k, r)]  ;
                    Vector3f dps =  (dQs * c0s.homogeneous()).head<3>() ;
                    Matrix4f dQt = bder[IDX3(bone_t, k, r)] ;
                    Vector3f dpt =  (dQt * c0t.homogeneous()).head<3>() ;

                    float dG = (dps - dpt).dot(dg0) ;

                    G(n_global_params + pb.offset() + r, i) = - 2 * rstg * dG;
                }
            }
        }


    }

    diffE = G.rowwise().sum()/N;

    return make_pair(total/N, diffE) ;
 }
