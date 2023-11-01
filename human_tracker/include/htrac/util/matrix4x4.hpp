#ifndef HTRAC_MATRIX4x4_HPP
#define HTRAC_MATRIX4x4_HPP

#include <htrac/util/vector4.hpp>
#include <htrac/util/matrix3x3.hpp>

class Matrix4x4 {
public:

__host__ __device__    Matrix4x4() = default ;
__host__ __device__    Matrix4x4(float v): Matrix4x4(Vec4{v, v, v, v}) {}
__host__ __device__    Matrix4x4(const Matrix4x4 &other) ;
__host__ __device__    Matrix4x4(const Eigen::Matrix4f &other) ;
__host__ __device__    Matrix4x4(const Matrix3x3 &rs, const Vec3 &t) ;
__host__ __device__    Matrix4x4(const Matrix3x3 &rs) ;

    // initialize from column vectors
__host__ __device__    Matrix4x4(const Vec4 &v1, const Vec4 &v2, const Vec4 &v3, const Vec4 &v4) ;

__host__ __device__    Matrix4x4(float a11, float a12, float a13, float a14,
              float a21, float a22, float a23, float a24,
              float a31, float a32, float a33, float a34,
              float a41, float a42, float a43, float a44) ;

    // Diagonal matrix

__host__ __device__    Matrix4x4(const Vec4 &diag) ;

__host__ __device__    void setUpperLeft(const Matrix3x3 &ul) ;

__host__ __device__    Matrix3x3 upperLeft() const ;

__host__ __device__    void setZero();
__host__ __device__    void setIdentity();

__host__ __device__    static Matrix4x4 identity() ;

    // Indexing operators

__host__ __device__    float& operator() (int i, int j) ;
__host__ __device__    const float& operator() (int i, int j) const ;

__host__ __device__    float *operator[] (int i) ;
__host__ __device__    const float *operator[] (int i) const ;

__host__ __device__    Vec4 row(int r) const ;
__host__ __device__    Vec4 column(int c) const ;

__host__ __device__    void setRow(int r, const Vec4 &) ;
__host__ __device__    void setColumn(int c, const Vec4 &) ;

__host__ __device__    Matrix4x4 &operator = (const Matrix4x4 &) ;
__host__ __device__    Matrix4x4 &operator = (const Eigen::Matrix4f &m) ;

    friend std::ostream &operator << (std::ostream &strm, const Matrix4x4 &m) ;

__host__ __device__    friend Matrix4x4 operator + (const Matrix4x4 &m1, const Matrix4x4 &m2) ;
__host__ __device__    friend Matrix4x4 operator - (const Matrix4x4 &m1, const Matrix4x4 &m2) ;
__host__ __device__    friend Matrix4x4 operator * (const Matrix4x4 &m1, const Matrix4x4 &m2) ;
__host__ __device__    friend Matrix4x4 operator * (const Matrix4x4 &m1, float s) ;
__host__ __device__    friend Matrix4x4 operator / (const Matrix4x4 &m1, float s) { return m1 * (1/s) ; }
__host__ __device__    friend Matrix4x4 operator * (float s, const Matrix4x4 &m1) { return m1*s ; }
__host__ __device__    friend Vec4 operator * (const Matrix4x4 &m1, const Vec4 &v) ;
__host__ __device__    friend Vec3 operator * (const Matrix4x4 &m1, const Vec3 &v) ;
__host__ __device__    friend Vec4 operator * (const Vec4 &v, const Matrix4x4 &m1) ;

__host__ __device__    Matrix4x4 &operator += (const Matrix4x4 &m) ;
__host__ __device__    Matrix4x4 &operator -= (const Matrix4x4 &m) ;

__host__ __device__    Matrix4x4 &operator *= (const Matrix4x4 &m) ;
__host__ __device__    Matrix4x4 &operator *= (float s) ;
__host__ __device__    Matrix4x4 &operator /= (float s) ;

__host__ __device__    friend Matrix4x4 operator - (const Matrix4x4 &);

    // Compute the inverse (inv) and determinant (det) of a matrix
__host__ __device__    Matrix4x4 inverse(bool *invertible = nullptr) const ;
__host__ __device__    void invert(bool *invertible) ;

    // Compute the transpose (float) of a matrix
__host__ __device__    Matrix4x4 transpose() const ;
__host__ __device__    void tranpose() ;

    // Return the determinant
__host__ __device__    float det() const ;

__host__ __device__    static Matrix4x4 lookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up) ;
__host__ __device__    static Matrix4x4 ortho( float left, float right, float bottom, float top, float zNear, float zFar) ;
__host__ __device__    static Matrix4x4 perspective(float yfov, float aspect, float znear, float zfar);

    // Matrix elements

protected:

    float m_[4][4] ;
} ;

inline void Matrix4x4::setUpperLeft(const Matrix3x3 &ul) {
    m_[0][0] = ul(0, 0) ; m_[0][1] = ul(0, 1) ; m_[0][2] = ul(0, 2) ;
    m_[1][0] = ul(1, 0) ; m_[1][1] = ul(1, 1) ; m_[1][2] = ul(1, 2) ;
    m_[2][0] = ul(2, 0) ; m_[2][1] = ul(2, 1) ; m_[2][2] = ul(2, 2) ;
}

inline Matrix3x3 Matrix4x4::upperLeft() const {
    return {
        m_[0][0], m_[0][1], m_[0][2],
        m_[1][0], m_[1][1], m_[1][2],
        m_[2][0], m_[2][1], m_[2][2]
    };
}

inline Matrix4x4 Matrix4x4::lookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up) {
    Vec3 f = (center - eye).normalized();
    Vec3 u = up.normalized();
    Vec3 s = f.cross(u).normalized();
    u = s.cross(f);
    return  {
        s.x(), s.y(), s.z(), -s.dot(eye),
        u.x(), u.y(), u.z(), -u.dot(eye),
       -f.x(),-f.y(),-f.z(),  f.dot(eye),
          0.f, 0.f, 0.f, 1.f
    };
}

inline Matrix4x4 Matrix4x4::ortho(float left, float right, float bottom, float top, float zNear, float zFar) {

    Matrix4x4 mat(1.f) ;

    mat(0,0) = 2.f / (right - left);
    mat(1,1) = 2.f / (top - bottom);
    mat(2,2) = - 2.f / (zFar - zNear);
    mat(3,0) = - (right + left) / (right - left);
    mat(3,1) = - (top + bottom) / (top - bottom);
    mat(3,2) = - (zFar + zNear) / (zFar - zNear);
    return mat;
}

inline Matrix4x4 Matrix4x4::perspective(float yfov, float aspect, float znear, float zfar) {
    assert(abs(aspect - std::numeric_limits<float>::epsilon()) > static_cast<float>(0));

//    float xfov = aspect_ * yfov_ ;
    float const d = 1/tan(yfov / static_cast<float>(2));

    Matrix4x4 result ;

    result(0, 0) = d / aspect ;
    result(1, 1) = d ;
    result(2, 2) =  (zfar + znear) / (znear - zfar);
    result(2, 3) =  2 * zfar * znear /(znear - zfar) ;
    result(3, 2) = -1 ;

    return result;
}

#include <htrac/util/matrix4x4.inl>


#endif
