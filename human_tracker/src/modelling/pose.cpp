#include <htrac/model/pose.hpp>
#include <htrac/model/skeleton.hpp>

#include "so3.hpp"
using namespace Eigen ;
using namespace std ;

extern Matrix4f quat2mat(const Quaternionf &q) ;
extern Matrix4f tran2mat(const Vector3f &t) ;

void Pose::setCoeffs(const VectorXf &data) {
    assert(data.size() == data_.size()) ;

    Vector3f gtr(data[0], data[1], data[2]) ;
    setGlobalTranslation(gtr);

#ifdef POSE_USE_QUAT_PARAM
    setGlobalRotationParams(data.block<4, 1>(3, 0));
#else
    setGlobalRotationParams(data.block<3, 1>(3, 0));
#endif

    for( const auto &pb: skeleton_->getPoseBones() ) {
        uint offset = pb.offset() ;

        for( uint k=0 ; k<pb.dofs() ; k++ ) {
            setBoneParams(pb.name(), data.data() + 3 + Pose::global_rot_params + offset)    ;
        }
    }
}

Pose::Pose(const Skeleton *skeleton): skeleton_(skeleton) {
    uint n_params = Pose::global_rot_params  + 3;
    for( const auto &pb: skeleton->getPoseBones() ) {
        n_params += pb.getParameterization()->dim() ;
    }
    data_.resize(n_params) ;
    setZero() ;
}

Pose::Pose(const Skeleton *skeleton, const Eigen::VectorXf &params): skeleton_(skeleton), data_(params.size()) {
    setCoeffs(params);
}

void Pose::setZero() {
   setGlobalTranslation(Vector3f::Zero()) ;
   setGlobalRotation(Quaternionf::Identity());

   float *data = data_.data() ;
   data += Pose::global_rot_params + 3 ;
   for ( const auto &pb: skeleton_->getPoseBones() ) {
       const RotationParameterization *rp = pb.getParameterization() ;
       rp->setZero(data) ;
       data += rp->dim() ;
   }
}

void Pose::setBoneParams(const string &name, const float *params) {
    const PoseBone *pb = skeleton_->findPoseBone(name) ;
    assert(pb) ;

    float *data = data_.data() ; data += Pose::global_rot_params + 3;
    data += pb->offset() ;

    const RotationParameterization *rp = pb->getParameterization() ;
    rp->normalize(data, params) ;
}

VectorXf Pose::getBoneParams(const std::string &name) const {
    const PoseBone *pb = skeleton_->findPoseBone(name) ;
    assert(pb) ;

    const float *data = data_.data() ; data += Pose::global_rot_params + 3 ;
    data += pb->offset() ;

    return Map<const VectorXf>(data, pb->getParameterization()->dim()) ;
}

#ifdef POSE_USE_QUAT_PARAM
void Pose::setGlobalRotation(const Eigen::Quaternionf &r) {
    Eigen::Quaternionf q = r.normalized() ;
    data_[3] = q.x() ; data_[4] = q.y() ; data_[5] = q.z() ; data_[6] = q.w() ;
}

Quaternionf Pose::getGlobalRotation() const { return Eigen::Quaternionf(data_[6], data_[3], data_[4], data_[5]); }

VectorXf Pose::getGlobalRotationParams() const {
    return data_.block<4, 1>(3, 0) ;
}

void Pose::setGlobalRotationParams(const Eigen::VectorXf &params) {
    data_[3] = params[0] ; data_[4] = params[1] ; data_[5] = params[2] ; data_[6] = params[3] ;
}

#else
void Pose::setGlobalRotation(const Eigen::Quaternionf &r) {
    SO3 so3(r.cast<double>()) ;
    const auto &c = so3.coeffs() ;
    data_[3] = c[0]; data_[4] = c[1] ; data_[5] = c[2] ;
}

void Pose::setGlobalRotationParams(const Eigen::VectorXf &params) {
    data_[3] = params[0] ; data_[4] = params[1] ; data_[5] = params[2] ;
}

Quaternionf Pose::getGlobalRotation() const {
    return SO3(data_[3], data_[4], data_[5]).toQuaternion().cast<float>();
}

VectorXf Pose::getGlobalRotationParams() const {
    return data_.block<3, 1>(3, 0) ;
}

#endif

Matrix4f Pose::getGlobalTransform() const {
    return tran2mat(getGlobalTranslation()) * quat2mat(getGlobalRotation())  ;
}

Eigen::Matrix4f Pose::getBoneTransform(const string &name) const  {
    return quat2mat(getBoneRotation(name)) ;
}

Quaternionf Pose::getBoneRotation(const std::string &name) const {
    const PoseBone *pb = skeleton_->findPoseBone(name) ;
    if ( pb == nullptr ) return Eigen::Quaternionf::Identity() ;

    const float *data = data_.data() ; data += Pose::global_rot_params + 3 ;
    const RotationParameterization *rp = pb->getParameterization() ;
    data += pb->offset() ;

    return rp->map(data) ;
}
