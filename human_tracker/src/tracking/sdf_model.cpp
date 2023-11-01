#include <htrac/model/sdf_model.hpp>

using namespace Eigen ;

MatrixXf SDFModel::transform_points_to_bone_space(const std::vector<Vector3f> &pts, const std::vector<Matrix4f> &itr) const {
    uint M = getNumParts() ;
    uint N = pts.size() ;

    MatrixXf tp(M, 3*N) ;

#pragma omp parallel for
    for(uint i=0 ; i<M ; i++) {
        uint bone = getPartBone(i) ;
        const Matrix4f &pose_bone_tr = itr[bone] ;

        for(uint j=0 ; j<N ; j++) {
            const Vector3f &v = pts[j] ;
            Vector3f pb = ( pose_bone_tr * v.homogeneous()).block<3,1>(0, 0) ;
            tp.block<1, 3>(i, 3*j) = pb.adjoint() ;
        }
    }

    return tp ;
}
