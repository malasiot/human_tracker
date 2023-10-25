#pragma once

#include <map>
#include <memory>

#include <Eigen/Geometry>

struct RotationParameterization {
    virtual void jacobian(const Eigen::VectorXf &x, Eigen::Matrix4f t[]) const = 0 ;
    virtual uint dim() const = 0 ;
    virtual void setZero(float *data) const = 0 ;

    virtual void normalize(float *dst, const float *src) const {
         for( size_t i=0 ; i<dim() ; i++ ) dst[i] = src[i] ;
    }

    virtual bool hasLimits() const { return false ; }
    virtual void getLimits(Eigen::VectorXf &) const {}

    virtual Eigen::Quaternionf map(const float *params) const=0;
};

struct QuaternionParameterization: public RotationParameterization {
    QuaternionParameterization(bool nrm = true): nrm_(nrm) {}

    void jacobian(const Eigen::VectorXf &x, Eigen::Matrix4f t[]) const override ;
    uint dim() const override { return 4 ; }
    void setZero(float *data) const override ;
    void normalize(float *dst, const float *src) const override ;
    Eigen::Quaternionf map(const float *params) const override ;

    bool nrm_ ;
};

struct ExponentialParameterization: public RotationParameterization {
    void jacobian(const Eigen::VectorXf &x, Eigen::Matrix4f t[]) const override ;
    uint dim() const override { return 3 ; }
    void setZero(float *data) const override ;
    Eigen::Quaternionf map(const float *params) const override ;
};

struct FixedAxisParameterization: public RotationParameterization {
    FixedAxisParameterization(const Eigen::Vector3f &axis): axis_(axis) {}

    void jacobian(const Eigen::VectorXf &x, Eigen::Matrix4f t[]) const override ;
    uint dim() const override { return 1 ; }
    void setZero(float *data) const override ;
    Eigen::Quaternionf map(const float *params) const override ;

private:
    Eigen::Vector3f axis_ ;
};

enum class RotationAxis { X, Y, Z } ;

struct Flexion: public RotationParameterization {

    Flexion(RotationAxis axis, float l, float u): axis_(axis), lower_(l), upper_(u) {}

    void jacobian(const Eigen::VectorXf &x, Eigen::Matrix4f t[]) const override ;
    uint dim() const override { return 1 ; }
    void setZero(float *data) const override ;
    Eigen::Quaternionf map(const float *params) const override ;

    static void jacobianMatrix(float angle, RotationAxis axis, Eigen::Matrix4f &t) ;
    static void rotationMatrix(float angle, RotationAxis axis, Eigen::Matrix4f &t) ;

    bool hasLimits() const override { return true ; }
    void getLimits(Eigen::VectorXf &limits) const override;

private:
    RotationAxis axis_ ;
    float lower_, upper_ ;
};


struct FlexionAbduction: public RotationParameterization {
    FlexionAbduction(RotationAxis flexion, RotationAxis abduction,
                     float flex_l, float flex_u, float abd_l, float abd_u):
        flexion_axis_(flexion), abduction_axis_(abduction) {
        flexion_limits_[0] = flex_l ; flexion_limits_[1] = flex_u ;
        abduction_limits_[0] = abd_l ; abduction_limits_[1] = abd_u ;
    }

    void jacobian(const Eigen::VectorXf &x, Eigen::Matrix4f t[]) const override ;
    uint dim() const override { return 2 ; }
    void setZero(float *data) const override ;
    Eigen::Quaternionf map(const float *params) const override ;

    bool hasLimits() const override { return true ; }
    void getLimits(Eigen::VectorXf &limits) const override;

private:
    RotationAxis flexion_axis_, abduction_axis_ ;
    float flexion_limits_[2] ;
    float abduction_limits_[2] ;
};

using PoseParameterization = std::map<std::string, std::shared_ptr<RotationParameterization>> ;

bool parsePoseParameterization(PoseParameterization &pr, const std::string &path) ;
