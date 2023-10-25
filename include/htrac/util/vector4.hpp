#ifndef HTRAC_Vec4_HPP
#define HTRAC_VECTO4_HPP

#include <iostream>
#include <cassert>
#include <cmath>

#include <htrac/util/vector3.hpp>

class Vec4 {
public:

__host__ __device__   Vec4() = default ;
__host__ __device__    Vec4(float v) ;
__host__ __device__    Vec4(float x, float y, float z, float w) ;
__host__ __device__    Vec4(const Vec4 &v) ;
__host__ __device__    Vec4(const Vec3 &v, float w = 1.0f) ;

__host__ __device__    float x() const ;
__host__ __device__    float &x() ;
__host__ __device__    float y() const ;
__host__ __device__    float &y() ;
__host__ __device__    float z() const ;
__host__ __device__    float &z() ;
__host__ __device__    float w() const ;
__host__ __device__    float &w() ;

__host__ __device__    Vec4 &operator=(const Vec4 &v) ;

__host__ __device__    friend Vec4 operator+(const Vec4 &v1, const Vec4 &v2) ;
__host__ __device__    friend Vec4 operator-(const Vec4 &v1, const Vec4 &v2) ;

__host__ __device__    const Vec4 &operator +=(const Vec4 &v) ;
__host__ __device__    const Vec4 &operator -=(const Vec4 &v) ;

__host__ __device__    Vec3 head() const ;

__host__ __device__    float& operator[] (size_t i) ;
__host__ __device__   const float& operator[] (size_t i) const ;

__host__ __device__    friend float dot(const Vec4 &v1, const Vec4 &v2) ;
__host__ __device__    float dot(const Vec4 &other) const ;

__host__ __device__    friend Vec4 operator *(const Vec4 &v, float f) ;
__host__ __device__   friend Vec4 operator /(const Vec4 &v, float f) ;
__host__ __device__    Vec4 &operator *=(float f) ;
__host__ __device__    Vec4 &operator /=(float f) ;
__host__ __device__    friend Vec4 operator *(float f, const Vec4 &b) { return b*f ; }
__host__ __device__    friend Vec4 operator *(const Vec4 &v, float f) ;
__host__ __device__    friend Vec4 operator -(const Vec4 &v) ;

__host__ __device__    float length() const ;
__host__ __device__    float norm() const ;
__host__ __device__    float squaredNorm() const ;

__host__ __device__    void normalize() ;
__host__ __device__    Vec4 normalized() const ;

    friend std::ostream &operator << (std::ostream &strm, const Vec4 &m) ;

    float x_, y_, z_, w_ ;
} ;

#include <htrac/util/vector4.inl>


#endif
