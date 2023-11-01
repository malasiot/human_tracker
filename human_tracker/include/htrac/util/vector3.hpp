#ifndef HTRAC_VECTOR_HPP
#define HTRAC_VECTOR_HPP

#include <iostream>
#include <cassert>
#include <cmath>

#include <Eigen/Core>

class Vec3 {
public:

__host__ __device__    Vec3() = default ;
__host__ __device__    Vec3(float v) ;
__host__ __device__    Vec3(float x, float y, float z) ;
__host__ __device__    Vec3(const Vec3 &v) ;
__host__ __device__    Vec3(const Eigen::Vector3f &v) ;

__host__ __device__    float x() const ;
__host__ __device__    float &x() ;
__host__ __device__    float y() const ;
__host__ __device__    float &y() ;
__host__ __device__    float z() const ;
__host__ __device__    float &z() ;

__host__ __device__    static Vec3 AxisX() {
        return { 1.f, 0.f, 0.f } ;
    }
__host__ __device__    static Vec3 AxisY() {
        return { 0.f, 1.f, 0.f } ;
    }
__host__ __device__    static Vec3 AxisZ() {
        return { 0.f, 0.f, 1.f } ;
    }

__host__ __device__    Vec3 &operator=(const Vec3 &v) ;

__host__ __device__    Vec3 &operator=(const Eigen::Vector3f &v) ;

__host__ __device__    friend Vec3 operator+(const Vec3 &v1, const Vec3 &v2) ;
__host__ __device__    friend Vec3 operator-(const Vec3 &v1, const Vec3 &v2) ;

__host__ __device__    const Vec3 &operator +=(const Vec3 &v) ;
__host__ __device__    const Vec3 &operator -=(const Vec3 &v) ;

__host__ __device__    float& operator[] (size_t i) ;
__host__ __device__    const float& operator[] (size_t i) const ;

__host__ __device__    friend float dot(const Vec3 &v1, const Vec3 &v2) ;
__host__ __device__    float dot(const Vec3 &other) const ;
__host__ __device__    friend Vec3 cross(const Vec3 &v1, const Vec3 &v2) ;
__host__ __device__    Vec3 cross(const Vec3 &v) const ;

__host__ __device__    friend Vec3 operator *(const Vec3 &v, float f) ;
__host__ __device__    friend Vec3 operator /(const Vec3 &v, float f) ;
__host__ __device__    Vec3 &operator *=(float f) ;
__host__ __device__    Vec3 &operator /=(float f) ;
__host__ __device__    friend Vec3 operator *(float f, const Vec3 &b) { return b*f ; }
__host__ __device__    friend Vec3 operator *(const Vec3 &v, float f) ;
__host__ __device__    friend Vec3 operator -(const Vec3 &v) ;

__host__ __device__    float length() const ;
__host__ __device__    float norm() const ;
__host__ __device__    float squaredNorm() const ;

__host__ __device__    void normalize() ;
__host__ __device__    Vec3 normalized() const ;

    friend std::ostream &operator << (std::ostream &strm, const Vec3 &m) ;

    float x_, y_, z_ ;
} ;

#include <htrac/util/vector3.inl>

#endif
