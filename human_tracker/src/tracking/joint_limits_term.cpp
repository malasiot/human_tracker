#include "joint_limits_term.hpp"

#include <iostream>

using namespace Eigen ;
using namespace std ;

JointLimitsTerm::JointLimitsTerm(const Skeleton &sk): skeleton_(sk) {
    n_terms_ = 0 ;
    for( const auto &pb: skeleton_.getPoseBones()) {
        const RotationParameterization *rp = pb.getParameterization() ;
        uint dim = rp->dim() ;
        if ( rp->hasLimits() )
            n_terms_ += dim ;
    }
}

void JointLimitsTerm::compute_energy(const Pose &pose, Eigen::VectorXf &e) const {
    const VectorXf &coeffs = pose.coeffs() ;

    uint count = 0 ;

    for( const auto &pb: skeleton_.getPoseBones()) {
        const RotationParameterization *rp = pb.getParameterization() ;
        uint offset = pb.offset() + Pose::global_rot_params + 3;
        uint dim = rp->dim() ;

        if ( rp->hasLimits() ) {
            VectorXf limits(2 * dim) ;
            rp->getLimits(limits) ;
            for( uint i=0, j=0 ; i<dim ; i++, j+=2) {
                float v = coeffs[offset + i];
                float l = limits[j] ;
                float u = limits[j + 1] ;
                if ( v < l ) {
                    e[count] = l - v ;
                } else if ( v > u ) {
                    e[count] = v - u ;
                } else
                    e[count] = 0 ;
                ++count ;
            }
        }

    }

}

void JointLimitsTerm::compute_jacobian(const Pose &pose, std::vector<std::tuple<uint, uint, float>> &jac, Eigen::VectorXf &e) const {
    const VectorXf &coeffs = pose.coeffs() ;

    uint count = 0;

    for( const auto &pb: skeleton_.getPoseBones()) {
        const RotationParameterization *rp = pb.getParameterization() ;
        uint offset = pb.offset() + Pose::global_rot_params + 3;
        uint dim = rp->dim() ;

        if ( rp->hasLimits() ) {
            VectorXf limits(2*dim) ;
            rp->getLimits(limits) ;
            for( uint i=0, j=0 ; i<dim ; i++, j+=2 ) {
                float v = coeffs[offset+i];
                float l = limits[j] ;
                float u = limits[j + 1] ;
                if ( v < l ) {
                    e[count] = l - v ;
                    jac.emplace_back(count, offset+i, -1) ;
                } else if ( v > u ) {
                    e[count] = v - u ;
                    jac.emplace_back(count, offset+i, 1) ;
                }
                ++count ;
            }
        }
    }
}

float JointLimitsTerm::energy(const Pose &pose) const {
    VectorXf e(n_terms_) ;
    compute_energy(pose, e) ;

    return e.squaredNorm() ;
}

std::pair<float, VectorXf> JointLimitsTerm::energyGradient(const Pose &pose) const {
    VectorXf diffE(skeleton_.getNumPoseBoneParams() + 3 + Pose::global_rot_params) ;
    diffE.setZero() ;

    float total = 0 ;

    const VectorXf &coeffs = pose.coeffs() ;

    for( const auto &pb: skeleton_.getPoseBones()) {
        const RotationParameterization *rp = pb.getParameterization() ;
        uint offset = pb.offset() + Pose::global_rot_params + 3;
        uint dim = rp->dim() ;

        if ( rp->hasLimits() ) {
            VectorXf limits(2*dim) ;
            rp->getLimits(limits) ;
            for( uint i=0, j=0 ; i<dim ; i++, j+=2 ) {
                float v = coeffs[offset+i];
                float l = limits[j] ;
                float u = limits[j + 1] ;
                if ( v < l ) {
                    total += (l - v) * (l - v) ;
                    diffE[offset+i] = -2 * ( l -v ) ;
                } else if ( v > u ) {
                    total += (v - u) * (v - u) ;
                    diffE[offset+i] = 2 * (v - u) ;
                }
            }
        }

    }

    return std::make_pair(total, diffE) ;
}

void JointLimitsTerm::energy(const Pose &pose, Eigen::VectorXf &e) const {
    compute_energy(pose, e) ;
}

void JointLimitsTerm::norm(const Pose &p, Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda) const {
    vector<tuple<uint, uint, float>> jac ;
    VectorXf e(n_terms_) ;

    e.setZero();

    compute_jacobian(p, jac, e) ;

    for( const auto [k, j, v]: jac ) {
        jtj(j, j) += lambda * v * v ;
        jte[j] += -lambda * v * e[k] ;
    }
}


