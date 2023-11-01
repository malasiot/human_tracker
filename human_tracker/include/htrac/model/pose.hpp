#pragma once

#include <map>
#include <Eigen/Geometry>

class Skeleton ;

#define POSE_USE_QUAT_PARAM 1

struct Pose {

    Pose(const Skeleton *skeleton) ;
    Pose(const Skeleton *skeleton, const Eigen::VectorXf &params) ;

    void setZero() ;

    void setBoneParams(const std::string &name, const float *params) ;

    void setBoneParams(const std::string &name, const std::initializer_list<float> &args) {
        std::vector<float> params{ args };
        setBoneParams(name, params.data()) ;
    }

    Eigen::VectorXf getBoneParams(const std::string &name) const ;

    void setGlobalRotation(const Eigen::Quaternionf &r);
    void setGlobalRotationParams(const Eigen::VectorXf &params) ;

    void setGlobalTranslation(const Eigen::Vector3f &t) {
        data_[0] = t.x() ; data_[1] = t.y() ; data_[2] = t.z() ;
    }

    Eigen::Quaternionf getGlobalRotation() const;
    Eigen::VectorXf getGlobalRotationParams() const ;
    Eigen::Vector3f getGlobalTranslation() const { return Eigen::Vector3f(data_[0], data_[1], data_[2]); }

    Eigen::Matrix4f getGlobalTransform() const ;
    Eigen::Matrix4f getBoneTransform(const std::string &name) const;
    Eigen::Quaternionf getBoneRotation(const std::string &name) const ;

    const Skeleton *getSkeleton() const { return skeleton_ ; }
    const Eigen::VectorXf &coeffs() const { return data_ ; }

    void setCoeffs(const Eigen::VectorXf &data);

#ifdef POSE_USE_QUAT_PARAM
    static const uint global_rot_params = 4 ;
#else
    static const uint global_rot_params = 3 ;
#endif

protected:

    const Skeleton *skeleton_ = nullptr ;
    Eigen::VectorXf data_ ;

};
