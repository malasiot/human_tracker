#include <htrac/model/parameterization.hpp>
#include <cvx/misc/variant.hpp>
#include <iostream>
#include <fstream>
#include "so3.hpp"

using namespace std ;
using namespace Eigen ;


static Matrix4f Rq(double x, double y, double z, double w)
{
    Matrix4f res ;

    float n22 = sqrt(x*x + y*y + z*z + w*w) ;

    x /= n22 ; y /= n22 ; z /= n22 ; w /= n22 ;

    res << 1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 0,
            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 0,
            2*(x*z-y*w), 2*(y*z+x*w),   1-2*(x*x+y*y), 0,
            0, 0, 0, 1 ;

    return res ;

}

static void diffR(Matrix4f rd[4], const double q[4])
{
    double x = q[0], y = q[1], z = q[2], w = q[3] ;

    rd[0] <<  0,      2*y,    2*z,    0,
            2*y,   -4*x,    -2*w,    0,
            2*z,    2*w,   -4*x,    0,
            0,      0,      0,      0 ;

    rd[1] << -4*y,    2*x,    2*w,    0,
            2*x,    0,      2*z,    0,
            -2*w,    2*z,   -4*y,    0,
            0,      0,      0,      0 ;

    rd[2] << -4*z,   -2*w,    2*x,    0,
            2*w,   -4*z,    2*y,    0,
            2*x,    2*y,    0,      0,
            0,      0,      0,      0 ;

    rd[3] <<  0,     -2*z,    2*y,    0,
            2*z,    0,     -2*x,    0,
            -2*y,    2*x,    0,      0,
            0,      0,      0,      0 ;


}

static void diffRnorm(Matrix4f rd[4], const double q[4])
{
    Matrix4f r[4] ;

    double x = q[0], y = q[1], z = q[2], w = q[3] ;
    float n2 = x*x + y*y + z*z + w*w ;
    float n22 = sqrt(n2) ;
    float n32 = n2 * n22 ;

    //    x/=n22 ; y/=n22 ; z/=n22 ; w/=n22 ;

    r[0] <<  0,      2*y,    2*z,    0,
            2*y,   -4*x,    -2*w,    0,
            2*z,    2*w,   -4*x,    0,
            0,      0,      0,      0 ;

    r[1] << -4*y,    2*x,    2*w,    0,
            2*x,    0,      2*z,    0,
            -2*w,    2*z,   -4*y,    0,
            0,      0,      0,      0 ;

    r[2] << -4*z,   -2*w,    2*x,    0,
            2*w,   -4*z,    2*y,    0,
            2*x,    2*y,    0,      0,
            0,      0,      0,      0 ;

    r[3] <<  0,     -2*z,    2*y,    0,
            2*z,    0,     -2*x,    0,
            -2*y,    2*x,    0,      0,
            0,      0,      0,      0 ;

    for(uint j=0 ; j<4 ; j++)
    {
        Matrix4f td = Matrix4f::Zero() ;

        for(uint m=0 ; m<4 ; m++)  {
            float nf = ( j == m ) ? 1/n22 : 0;
            nf -= q[m] * q[j] / n32 ;
            td += r[m] * nf ;
        }

        rd[j] = td ;

    }

}

void QuaternionParameterization::jacobian(const VectorXf &x, Eigen::Matrix4f t[]) const {
    double q[4] = { x[0], x[1], x[2], x[3] };
    if ( nrm_ ) diffRnorm(t, q) ;
    else diffR(t, q) ;
}

void QuaternionParameterization::setZero(float *data) const {
    data[0] = data[1] = data[2] = 0.0 ;
    data[3] = 1.0 ;
}

void QuaternionParameterization::normalize(float *dst, const float *src) const
{
    Quaternionf q(src[3], src[0], src[1], src[2]) ;
    q.normalize();
    dst[0] = q.x() ; dst[1] = q.y() ; dst[2] = q.z() ; dst[3] = q.w() ;
}


Quaternionf QuaternionParameterization::map(const float *params) const {
    return Quaternionf(params[3], params[0], params[1], params[2]) ;
}


void FixedAxisParameterization::jacobian(const VectorXf &x, Eigen::Matrix4f t[]) const {
    double angle = x[0] ;
    double s = sin(angle), c = cos(angle), ux = axis_.x(), uy = axis_.y(), uz = axis_.z() ;

    t[0] << -s + ux*ux*s, ux*uy*s-uz*c, ux*uz*s+uy*c, 0,
            ux*uy*s+uz*c, -s + uy*uy*s, uy*uz*s-ux*c, 0,
            ux*uz*s-uy*c, uy*uz*s+ux*c, -s + uz*uz*s, 0,
            0, 0, 0, 0 ;
}

void FixedAxisParameterization::setZero(float *data) const {
    data[0] = 0.0 ;
}


void Flexion::setZero(float *data) const {
    data[0] = 0.0 ;
}

Quaternionf Flexion::map(const float *params) const {
    switch ( axis_ ) {
    case RotationAxis::X:
        return Quaternionf(AngleAxisf(params[0], Vector3f::UnitX())).normalized() ;
    case RotationAxis::Y:
        return Quaternionf(AngleAxisf(params[0], Vector3f::UnitY())).normalized() ;
    case RotationAxis::Z:
        return Quaternionf(AngleAxisf(params[0], Vector3f::UnitZ())).normalized() ;
    }
}

void Flexion::jacobianMatrix(float angle, RotationAxis axis, Matrix4f &t) {

    double s = sin(angle), c = cos(angle) ;

    if ( axis == RotationAxis::X ) {
        t << 1, 0, 0, 0,
                0, -s, -c, 0,
                0, c, -s, 0,
                0, 0, 0, 0 ;
    } else if ( axis == RotationAxis::Y ) {
        t << -s, 0, c, 0,
                0, 1, 0, 0,
                -c, 0, -s, 0,
                0, 0, 0, 0 ;
    } else if ( axis == RotationAxis::Z ) {
        t << -s, -c, 0, 0,
                c, -s, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 0 ;
    }
}

void Flexion::rotationMatrix(float angle, RotationAxis axis, Eigen::Matrix4f &t) {
    t.setIdentity() ;

    switch ( axis ) {
    case RotationAxis::X:
        t.block<3, 3>(0, 0) = AngleAxisf(angle, Vector3f::UnitX()).toRotationMatrix() ;
        break ;
    case RotationAxis::Y:
        t.block<3, 3>(0, 0) = AngleAxisf(angle, Vector3f::UnitY()).toRotationMatrix() ;
        break ;
    case RotationAxis::Z:
        t.block<3, 3>(0, 0) = AngleAxisf(angle, Vector3f::UnitZ()).toRotationMatrix() ;
        break ;
    }
}

void Flexion::getLimits(Eigen::VectorXf &limits) const {
    limits[0] = lower_ ;
    limits[1] = upper_ ;
}

void Flexion::jacobian(const VectorXf &x, Matrix4f t[]) const {
    jacobianMatrix(x[0], axis_, t[0]) ;
}




void FlexionAbduction::setZero(float *data) const {
    data[0] = data[1] = 0.0 ;
}

Quaternionf FlexionAbduction::map(const float *params) const
{
    Matrix4f rf, ra ;

    Flexion::rotationMatrix(params[0], flexion_axis_, rf) ;
    Flexion::rotationMatrix(params[1], abduction_axis_, ra) ;

    return Quaternionf(rf.block<3, 3>(0, 0) * ra.block<3, 3>(0, 0)).normalized() ;

}

void FlexionAbduction::jacobian(const VectorXf &x, Matrix4f t[]) const {

    Matrix4f jf, ja, rf, ra ;

    Flexion::jacobianMatrix(x[0], flexion_axis_, jf) ;
    Flexion::jacobianMatrix(x[1], abduction_axis_, ja) ;

    Flexion::rotationMatrix(x[0], flexion_axis_, rf) ;
    Flexion::rotationMatrix(x[1], abduction_axis_, ra) ;

    t[0] = jf * ra ;
    t[1] = rf * ja ;
}

void FlexionAbduction::getLimits(Eigen::VectorXf &limits) const {
    limits[0] = flexion_limits_[0] ;
    limits[1] = flexion_limits_[1] ;
    limits[2] = abduction_limits_[0] ;
    limits[3] = abduction_limits_[1] ;
}

Quaternionf FixedAxisParameterization::map(const float *params) const {
    Quaternionf v(AngleAxisf(params[0], axis_)) ;
    return v.normalized() ;
}

void ExponentialParameterization::jacobian(const Eigen::VectorXf &x, Eigen::Matrix4f t[]) const
{
    SO3 so3(x[0], x[1], x[2]) ;

    Matrix3d J[3] ;
    so3.jacobian(J);

    for( uint i=0 ; i<3 ; i++ ) {
        t[i] = Matrix4f::Zero() ;
        t[i].block<3, 3>(0, 0) = J[i].cast<float>() ;
    }
}

void ExponentialParameterization::setZero(float *data) const {
    data[0] = data[1] = data[2] = 0.0 ;
}

Quaternionf ExponentialParameterization::map(const float *params) const {
    SO3 so3(params[0], params[1], params[2]) ;
    return so3.toQuaternion().cast<float>() ;
}

static void parseAxisAndLimits(const cvx::Variant &json, RotationAxis &axis, float &l, float &u) {

    string saxis = json.value("axis", "X").toString() ;
    if ( saxis == "X" ) axis = RotationAxis::X ;
    else if ( saxis == "Y" ) axis = RotationAxis::Y ;
    else if ( saxis == "Z" ) axis = RotationAxis::Z ;

    auto limits = json["limits"] ;

    if ( limits ) {
        l = limits[0].toFloat() ;
        u = limits[1].toFloat() ;
    }
}

RotationParameterization *parseRotationParam(const cvx::Variant &json) {

    string type = json["rot"].toString() ;

    if ( type == "quaternion" )
        return new QuaternionParameterization ;
    else if ( type == "exponential" )
        return new ExponentialParameterization() ;
    else if ( type == "flexion") {
        float lower, upper ;
        RotationAxis axis ;
        parseAxisAndLimits(json, axis, lower, upper) ;
        return new Flexion(axis, lower, upper) ;
    } else if ( type == "flexion-abduction" ) {
        float flower, fupper, alower, aupper ;

        RotationAxis aaxis, faxis ;
        parseAxisAndLimits(json["flexion"], faxis, flower, fupper) ;
        parseAxisAndLimits(json["abduction"], aaxis, alower, aupper) ;
        return new FlexionAbduction(faxis, aaxis, flower, fupper, alower, aupper) ;

    }
}

bool parsePoseParameterization(PoseParameterization &pr, const std::string &path) {
    ifstream strm(path) ;
    if ( !strm ) return false ;

    try {
        cvx::Variant json = cvx::Variant::fromJSON(strm) ;
        if ( !json ) return false ;

        auto it = json.begin() ;
        for( ; it != json.end() ; ++it ) {
            string bone = it.key() ;
            RotationParameterization *rp = parseRotationParam(it.value()) ;
            pr.emplace(bone, rp) ;

        }

        return true ;

    } catch ( std::runtime_error &e ) {
        cout << e.what() << endl ;
        return false ;
    }



}
