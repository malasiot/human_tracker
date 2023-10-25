#include "context.hpp"
#include "energy_term.hpp"

using namespace std ;
using namespace Eigen ;

Context::Context(const Skeleton &sk): skeleton_(sk) {
    const auto &bones = skeleton_.bones() ;
    ioffset_.resize(bones.size()) ;

#pragma omp parallel for
    for( uint i=0 ; i<bones.size() ; i++ ) {
        const auto &b = bones[i] ;
        ioffset_[i] = b.offset_.inverse() ;
    }
}

void Context::computeTransforms(const Pose &pose) {
    skeleton_.computeBoneTransforms(pose, trans_) ;
    itrans_.resize(trans_.size()) ;

#pragma omp parallel for
    for(uint i=0 ; i<trans_.size() ; i++ )     {
        itrans_[i] = trans_[i].inverse() ;
    }
}

void Context::computeDerivatives(const Pose &pose) {
    EnergyTerm::compute_transform_derivatives(skeleton_, pose, bder_, gder_) ;
}
