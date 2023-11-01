#include "keypoints_term.hpp"
#include "energy_term.hpp"

using namespace std ;
using namespace Eigen ;

float KeyPointsTerm::energy(const Pose &pose) {
    float e = 0.0 ;
    for( const auto &obp: kpts_ ) {
        const string &bname = obp.first ;
        uint bidx = ctx_.skeleton_.getBoneIndex(bname) ;
        const auto &obs = obp.second ;
        auto tr = ctx_.trans_[bidx] ;
        Vector3f coords = tr.block<3, 1>(0, 3) ;
        e += obs.second * (obs.first - coords).squaredNorm() ;
    }

    return e ;
}

std::pair<float, VectorXf> KeyPointsTerm::energyGradient(const Pose &pose) {
    float e ;

    uint N = kpts_.size() ;

    MatrixXf G(pose.coeffs().size(), N) ;
    G.setZero() ;


    const auto &pbv = ctx_.skeleton_.getPoseBones() ;
    size_t n_pose_bones = pbv.size() ;
    size_t n_global_params = Pose::global_rot_params + 3 ;

#define IDX3(i, j, k) (n_pose_bones * 4 * (i) + 4 * (j) + (k))
#define IDX2(i, j) ( n_global_params * (i) + j)

    uint i=0 ;
    for( const auto &obp: kpts_ ) {
        const string &bname = obp.first ;
        uint bidx = ctx_.skeleton_.getBoneIndex(bname) ;
        const auto &obs = obp.second ;
        float w = obs.second ;
        const Vector3f &o = obs.first ;
        auto tr = ctx_.trans_[bidx] ;
        Vector3f coords = tr.block<3, 1>(0, 3) ;
        Vector3f dobj = coords - o ;

        for( uint k=0 ; k<pbv.size() ; k++ ) {
            const auto &pb = pbv[k] ;

            size_t dim  = pb.dofs() ;

            for( int r=0 ; r<dim ; r++ ) {
               auto dQ = ctx_.bder_[IDX3(bidx, k, r)] ;
               auto dj = dQ.block<3, 1>(0, 3) ;

               float de = 2 * w * dobj.dot(dj) ;

               G(n_global_params + pb.offset() + r, i) = de ;
            }
        }

        for( int k=0 ;k<3 + Pose::global_rot_params ; k++ ) {
            auto dQ = ctx_.gder_[IDX2(bidx, k)] ;
            auto dj = dQ.block<3, 1>(0, 3) ;

            float de = 2 * w * dobj.dot(dj) ;

            G(k, i) = de ;
        }

        ++i ;
    }

    VectorXf diffE = G.rowwise().sum() ;

    return make_pair(0, diffE) ;
}

void KeyPointsTerm::energy(Eigen::VectorXf &e) const {
    uint i = 0 ;
    for( const auto &obp: kpts_ ) {
        const string &bname = obp.first ;
        uint bidx = ctx_.skeleton_.getBoneIndex(bname) ;
        const auto &obs = obp.second ;
        auto tr = ctx_.trans_[bidx] ;
        Vector3f coords = tr.block<3, 1>(0, 3) ;
        e[i++]  = sqrt(obs.second) * (obs.first - coords ).norm() ;
    }
}

void KeyPointsTerm::jacobian(Eigen::MatrixXf &jac) const {

    uint N = kpts_.size() ;

    const auto &pbv = ctx_.skeleton_.getPoseBones() ;
    size_t n_pose_bones = pbv.size() ;
    size_t n_global_params = Pose::global_rot_params + 3 ;

#define IDX3(i, j, k) (n_pose_bones * 4 * (i) + 4 * (j) + (k))
#define IDX2(i, j) ( n_global_params * (i) + j)

    uint i=0 ;
    for( const auto &obp: kpts_ ) {
        const string &bname = obp.first ;
        uint bidx = ctx_.skeleton_.getBoneIndex(bname) ;
        const auto &obs = obp.second ;
        float w = obs.second ;
        const Vector3f &o = obs.first ;
        auto tr = ctx_.trans_[bidx] ;
        Vector3f coords = tr.block<3, 1>(0, 3) ;
        Vector3f dobj = (coords - o).normalized() ;

        for( uint k=0 ; k<pbv.size() ; k++ ) {
            const auto &pb = pbv[k] ;

            size_t dim  = pb.dofs() ;

            for( int r=0 ; r<dim ; r++ ) {
               auto dQ = ctx_.bder_[IDX3(bidx, k, r)] ;
               auto dj = dQ.block<3, 1>(0, 3) ;

               float de =  sqrt(w) * dobj.dot(dj) ;

               jac(i, n_global_params + pb.offset() + r) = -de ;
            }
        }

        for( int k=0 ;k<3 + Pose::global_rot_params ; k++ ) {
            auto dQ = ctx_.gder_[IDX2(bidx, k)] ;
            auto dj = dQ.block<3, 1>(0, 3) ;

            float de = sqrt(w) * dobj.dot(dj) ;

            jac(i, k) = -de ;
        }

        ++i ;
    }
}

void KeyPointsTerm::norm(Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda) const {

    MatrixXf jac(kpts_.size(), jte.size()) ;
    VectorXf e(kpts_.size()) ;

    jacobian(jac) ;
    energy(e) ;

    jtj += lambda * jac.adjoint() * jac ;
    jte += lambda * jac.adjoint() * e ;
}
