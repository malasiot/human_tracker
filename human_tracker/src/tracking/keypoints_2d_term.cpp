#include "keypoints_2d_term.hpp"


using namespace Eigen ;
using namespace std ;

float KeyPoints2DTerm::energy(const Pose &pose) {
    float e = 0.0 ;
    for( const auto &bname: kdf_->boneNames() ) {
        uint bidx = ctx_.skeleton_.getBoneIndex(bname) ;
        auto tr = ctx_.trans_[bidx] ;
        Vector3f coords = tr.block<3, 1>(0, 3) ;
        cv::Point2d pj = cam_.project(cv::Point3d(coords.x(), -coords.y(), -coords.z()));
        Vector2f ip(pj.x + 0.5, pj.y + 0.5) ;
        float dist = kdf_->getDistance(bname, ip) ;
        e += dist * dist ;
    }

    return e ;
}

static Vector2f proj_deriv(float f, const Vector3f &p, const Vector3f &dp) {
    float X = p.x(), Y = p.y(), Z = p.z() ;
    float ZZ = Z * Z / f ;
    return { -( dp.x() * Z -  dp.z() * X )/ZZ,  ( dp.y() * Z -  dp.z() * Y )/ZZ } ;
}

std::pair<float, VectorXf> KeyPoints2DTerm::energyGradient(const Pose &pose) {
    float e ;

    uint N = kdf_->nBones() ;

    MatrixXf G(pose.coeffs().size(), N) ;
    G.setZero() ;


    const auto &pbv = ctx_.skeleton_.getPoseBones() ;
    size_t n_pose_bones = pbv.size() ;
    size_t n_global_params = Pose::global_rot_params + 3 ;

#define IDX3(i, j, k) (n_pose_bones * 4 * (i) + 4 * (j) + (k))
#define IDX2(i, j) ( n_global_params * (i) + j)

    uint i=0 ;
    for( const auto &bname: kdf_->boneNames() ) {
        uint bidx = ctx_.skeleton_.getBoneIndex(bname) ;

        auto tr = ctx_.trans_[bidx] ;
        Vector3f coords = tr.block<3, 1>(0, 3) ;

        cv::Point2d pj = cam_.project(cv::Point3d(coords.x(), -coords.y(), -coords.z()));
        Vector2f ip(pj.x + 0.5, pj.y + 0.5) ;
        Vector2f og = kdf_->getDistanceGradient(bname, ip) ;
        float vd = kdf_->getDistance(bname, ip) ;

        for( uint k=0 ; k<pbv.size() ; k++ ) {
            const auto &pb = pbv[k] ;

            size_t dim  = pb.dofs() ;

            for( int r=0 ; r<dim ; r++ ) {
               auto dQ = ctx_.bder_[IDX3(bidx, k, r)] ;
               auto dj = dQ.block<3, 1>(0, 3) ;

               Vector2f dpg = proj_deriv(cam_.fx(), coords, dj) ;
               float gd = og.dot(dpg) ;

               G(n_global_params + pb.offset() + r, i) = 2*vd*gd ;
            }
        }


        for( int k=0 ;k<3 + Pose::global_rot_params ; k++ ) {
            auto dQ = ctx_.gder_[IDX2(bidx, k)] ;
            auto dj = dQ.block<3, 1>(0, 3) ;

            Vector2f dpg = proj_deriv(cam_.fx(), coords, dj) ;
            float gd = og.dot(dpg) ;

            G(k, i) = 2*vd*gd ;
        }

        ++i ;
    }

    VectorXf diffE = G.rowwise().sum() ;

    return make_pair(0, diffE) ;
}

void KeyPoints2DTerm::energy(Eigen::VectorXf &e) const {
    uint i = 0 ;

    if ( kdf_ == nullptr ) return ;

    for( const auto &bname: kdf_->boneNames() ) {
        int bidx = ctx_.skeleton_.getBoneIndex(bname) ;
        if ( bidx < 0 ) continue ;

        auto tr = ctx_.trans_[bidx] ;
        Vector3f coords = tr.block<3, 1>(0, 3) ;

        cv::Point2d pj = cam_.project(cv::Point3d(coords.x(), -coords.y(), -coords.z()));
        Vector2f ip(pj.x + 0.5, pj.y + 0.5) ;
        float dist = kdf_->getDistance(bname, ip) ;

        e[i++]  = dist/scale_ ;
    }
}

void KeyPoints2DTerm::jacobian(Eigen::MatrixXf &jac) const {

    if ( kdf_ == nullptr ) return ;

    uint N = kdf_->nBones() ;

    const auto &pbv = ctx_.skeleton_.getPoseBones() ;
    size_t n_pose_bones = pbv.size() ;
    size_t n_global_params = Pose::global_rot_params + 3 ;

#define IDX3(i, j, k) (n_pose_bones * 4 * (i) + 4 * (j) + (k))
#define IDX2(i, j) ( n_global_params * (i) + j)

    uint i=0 ;
    for( const auto &bname: kdf_->boneNames() ) {
        int bidx = ctx_.skeleton_.getBoneIndex(bname) ;
        if ( bidx < 0 ) continue ;
        auto tr = ctx_.trans_[bidx] ;
        Vector3f coords = tr.block<3, 1>(0, 3) ;

        cv::Point2d pj = cam_.project(cv::Point3d(coords.x(), -coords.y(), -coords.z()));
        Vector2f ip(pj.x + 0.5, pj.y + 0.5) ;
        Vector2f og = kdf_->getDistanceGradient(bname, ip) ;
        float vd = kdf_->getDistance(bname, ip) ;

        for( uint k=0 ; k<pbv.size() ; k++ ) {
            const auto &pb = pbv[k] ;

            size_t dim  = pb.dofs() ;

            for( int r=0 ; r<dim ; r++ ) {
               auto dQ = ctx_.bder_[IDX3(bidx, k, r)] ;
               auto dj = dQ.block<3, 1>(0, 3) ;

               Vector2f dpg = proj_deriv(cam_.fx(), coords, dj) ;
               float gd = og.dot(dpg) ;

               jac(i, n_global_params + pb.offset() + r) = gd/scale_ ;
            }
        }

        for( int k=0 ;k<3 + Pose::global_rot_params ; k++ ) {
            auto dQ = ctx_.gder_[IDX2(bidx, k)] ;
            auto dj = dQ.block<3, 1>(0, 3) ;

            Vector2f dpg = proj_deriv(cam_.fx(), coords, dj) ;
            float gd = og.dot(dpg) ;

            jac(i, k) = gd/scale_ ;
        }

        ++i ;
    }
}

void KeyPoints2DTerm::norm(Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda) const {

    if ( kdf_ == nullptr ) return ;

    MatrixXf jac(kdf_->nBones(), jte.size()) ;
    VectorXf e(kdf_->nBones()) ;

    jacobian(jac) ;
    energy(e) ;

    jtj += lambda * jac.adjoint() * jac ;
    jte += lambda * jac.adjoint() * e ;
}
