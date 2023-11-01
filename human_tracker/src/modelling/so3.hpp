#ifndef SO3_HPP
#define SO3_HPP

#include <Eigen/Geometry>

#include <iostream>

class SO3 {
public:
    SO3(const double *data): u_(data) {}
    SO3(const Eigen::Vector3d &data): u_(data) {}
    SO3(double u0, double u1, double u2): u_{u0, u1, u2} {}
    SO3(const Eigen::Quaterniond &q) {
        Eigen::AngleAxisd a(q) ;
        u_ = a.angle() * a.axis() ;
    }
    SO3(): u_(0, 0, 0) {}

    static constexpr double epsilon_ = 1.0e-10 ;

    Eigen::Quaterniond toQuaternion() const {
        double theta_sq = u_.squaredNorm();

        double imag_factor;
        double real_factor;
        if ( theta_sq < epsilon_ * epsilon_ ) {

            double theta_po4 = theta_sq * theta_sq;
            imag_factor = 0.5 - (1.0 / 48.0) * theta_sq +
                    (1.0 / 3840.0) * theta_po4;
            real_factor = 1 - (1.0 / 8.0) * theta_sq +
                    (1.0 / 384.0) * theta_po4;
        } else {
            double theta = sqrt(theta_sq) ;
            double half_theta = 0.5 * theta;
            double sin_half_theta = sin(half_theta);
            imag_factor = sin_half_theta / theta;
            real_factor = cos(half_theta);
        }

        return Eigen::Quaterniond(real_factor,  imag_factor * u_.x(),
                           imag_factor * u_.y(), imag_factor * u_.z()).normalized();
    }

    Eigen::Matrix3d matrix() const { // Rodrigues formula
        const double theta = u_.norm() ;

        if (theta < epsilon_ ) {
            return Eigen::Matrix3d::Identity() ;
        }

        double ux = u_.x()/theta ;
        double uy = u_.y()/theta ;
        double uz = u_.z()/theta ;

        double uxx = ux * ux, uxy = ux * uy, uxz = ux * uz, uyy = uy * uy, uyz = uy * uz, uzz = uz * uz ;

        const double s_theta = sin(theta);
        const double omc_theta = 1.0 - cos(theta);

        Eigen::Matrix3d r ;

        r <<  1 + omc_theta*(uxx -1),      -s_theta*uz + omc_theta*uxy, s_theta*uy + omc_theta*uxz,
                s_theta*uz + omc_theta*uxy,   1 + omc_theta*(uyy-1), -s_theta*ux + omc_theta*uyz,
                -s_theta*uy + omc_theta*uxz,  s_theta*ux + omc_theta*uyz,     1 + omc_theta*(uzz-1) ;

        return r ;
    }

    static SO3 fromMatrix(const Eigen::Matrix3d &src) {

        const double cosTheta = (src(0, 0) + src(1, 1) + src(2, 2) - 1)/2.0;
        const double theta = cosTheta >= 0.9999 ? 0 : acos(cosTheta);

        if ( theta < epsilon_ )
            return SO3(0, 0, 0) ;

        const double is = theta/(2*sin(theta)) ;

        return SO3((src(2, 1) - src(1, 2)) * is,
                   (src(0, 2) - src(2, 0)) * is,
                   (src(1, 0) - src(0, 1)) * is);
    }

    friend std::ostream &operator << (std::ostream &strm, const SO3 &s) {
        strm << s.u_ ;
        return strm ;
    }

    static inline Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
        Eigen::Matrix3d s ;
        s << 0, -v.z(), v.y(),
                v.z(), 0, -v.x(),
                -v.y(), v.x(), 0 ;
        return s ;
    }

    // A compact formula for the derivative of a 3-D rotation in exponential coordinates
    // Guillermo Gallego, Anthony Yezzi

    void jacobian(Eigen::Matrix3d J[3]) {
        double den = u_.squaredNorm() ;

        if ( den < epsilon_ * epsilon_ ) {
            for( int i=0 ; i<3 ; i++ ) {
                Eigen::Vector3d e(0, 0, 0) ;
                e[i] = 1 ;
                J[i] = skew(e) ;
            }
            return ;
        }

        auto R = matrix() ;
        auto S = skew(u_) ;
        auto I = Eigen::Matrix3d::Identity() ;
        auto IR = I - R ;

        for( int i=0 ; i<3 ; i++ ) {
            J[i] = (u_[i] * S + skew(u_.cross(IR.col(i))))*R/den ;
        }
    }

    const Eigen::Vector3d &coeffs() const { return u_ ; }

private:

    Eigen::Vector3d u_ ;
};









#endif
