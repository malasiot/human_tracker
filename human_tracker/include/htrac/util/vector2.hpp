#pragma once

#include <Eigen/Geometry>

class Vec2 {
public:

__host__ __device__    Vec2() = default ;
__host__ __device__    Vec2(float v) ;
__host__ __device__    Vec2(float x, float y) ;
__host__ __device__    Vec2(const Vec2 &v) ;
__host__ __device__    Vec2(const Eigen::Vector2f &v) ;

__host__ __device__    float x() const ;
__host__ __device__    float &x() ;
__host__ __device__    float y() const ;
__host__ __device__    float &y() ;

__host__ __device__    Vec2 &operator=(const Vec2 &v) ;

__host__ __device__    Vec2 &operator=(const Eigen::Vector2f &v) ;

__host__ __device__    friend Vec2 operator+(const Vec2 &v1, const Vec2 &v2) ;
__host__ __device__    friend Vec2 operator-(const Vec2 &v1, const Vec2 &v2) ;

__host__ __device__    const Vec2 &operator +=(const Vec2 &v) ;
__host__ __device__    const Vec2 &operator -=(const Vec2 &v) ;

__host__ __device__    float& operator[] (size_t i) ;
__host__ __device__    const float& operator[] (size_t i) const ;

__host__ __device__    friend float dot(const Vec2 &v1, const Vec2 &v2) ;
__host__ __device__    float dot(const Vec2 &other) const ;
__host__ __device__    friend Vec2 cross(const Vec2 &v1, const Vec2 &v2) ;
__host__ __device__    Vec2 cross(const Vec2 &v) const ;

__host__ __device__    friend Vec2 operator *(const Vec2 &v, float f) ;
__host__ __device__    friend Vec2 operator /(const Vec2 &v, float f) ;
__host__ __device__    Vec2 &operator *=(float f) ;
__host__ __device__    Vec2 &operator /=(float f) ;
__host__ __device__    friend Vec2 operator *(float f, const Vec2 &b) { return b*f ; }
__host__ __device__    friend Vec2 operator *(const Vec2 &v, float f) ;
__host__ __device__    friend Vec2 operator -(const Vec2 &v) ;

__host__ __device__    float length() const ;
__host__ __device__    float norm() const ;
__host__ __device__    float squaredNorm() const ;

__host__ __device__    void normalize() ;
__host__ __device__    Vec2 normalized() const ;

    friend std::ostream &operator << (std::ostream &strm, const Vec2 &m) ;

    float x_, y_ ;
} ;

#include <htrac/util/vector2.inl>
