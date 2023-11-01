#include "energy_term.hpp"

using namespace std ;
using namespace Eigen ;


#define IDX2(d, i, j) (d * (i) + (j))

void EnergyTerm::compute_transform_derivatives(const Skeleton &skeleton, const Pose &pose,
                                               std::vector<Matrix4f> &der, std::vector<Matrix4f> &gder) {
    const auto &pb = skeleton.getPoseBones() ;
    const auto &bones = skeleton.bones() ;

    size_t n_bones = bones.size() ;
    size_t n_pose_bones = pb.size() ;
    size_t n_global_params = Pose::global_rot_params + 3 ;

    der.resize(n_bones * n_pose_bones * 4) ;
    gder.resize(n_bones * n_global_params) ;

#define IDX3(i, j) (n_pose_bones * 4 * (i) + 4 * (j))
#pragma omp parallel for
    for( uint i=0 ; i<bones.size() ; i++ ) {
        for(uint j=0 ; j<pb.size() ; j++)  {
            //skeleton.computeBoneRotationDerivatives(pose, j, i, &(der_[j][i][0])) ;
            skeleton.computeBoneRotationDerivatives(pose, i, j, &der[IDX3(i, j)]) ;
        }
    }

    // derivatives with respect to global transform parameters
#pragma omp parallel for
    for(uint i=0 ; i<bones.size() ; i++)  {
        skeleton.computeGlobalDerivatives(pose, i, &(gder[i * n_global_params]), &(gder[i * n_global_params + 3]));
    }
}
