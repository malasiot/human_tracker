#ifndef HTRAC_MATRIX3x3_HPP
#define HTRAC_MATRIX3x3_HPP

#include <htrac/util/vector3.hpp>


class Matrix3x3 {
public:

__host__ __device__    Matrix3x3() = default ;
__host__ __device__    Matrix3x3(const Matrix3x3 &other) ;

    // initialize from column vectors
__host__ __device__    Matrix3x3(const Vec3 &v1, const Vec3 &v2, const Vec3 &v3) ;

__host__ __device__    Matrix3x3(float a11, float a12, float a13,
              float a21, float a22, float a23,
              float a31, float a32, float a33) ;

__host__ __device__    Matrix3x3(float a[9]) ;

    // Diagonal matrix

__host__ __device__    Matrix3x3(const Vec3 &diag) ;

__host__ __device__    void setZero();
__host__ __device__    void setIdentity();

__host__ __device__    static Matrix3x3 identity() ;
    // equals: v1 * v2'
__host__ __device__    static Matrix3x3 outer(const Vec3 &v1, const Vec3 &v2) ;

    // Indexing operators

__host__ __device__    float& operator() (int i, int j) ;
__host__ __device__    float operator() (int i, int j) const ;
__host__ __device__    float *operator[] (int i) ;
__host__ __device__    const float *operator[] (int i) const ;

__host__ __device__    Vec3 row(int r) const ;
__host__ __device__    Vec3 column(int c) const ;

__host__ __device__    void setRow(int r, const Vec3 &) ;
__host__ __device__    void setColumn(int c, const Vec3 &) ;

__host__ __device__    Matrix3x3 &operator = (const Matrix3x3 &) ;

__host__ __device__    friend std::ostream &operator << (std::ostream &strm, const Matrix3x3 &m) ;

__host__ __device__    friend Matrix3x3 operator + (const Matrix3x3 &m1, const Matrix3x3 &m2) ;
__host__ __device__    friend Matrix3x3 operator - (const Matrix3x3 &m1, const Matrix3x3 &m2) ;
__host__ __device__    friend Matrix3x3 operator * (const Matrix3x3 &m1, const Matrix3x3 &m2) ;
__host__ __device__    friend Matrix3x3 operator * (const Matrix3x3 &m1, float s) ;
__host__ __device__    friend Matrix3x3 operator / (const Matrix3x3 &m1, float s) { return m1 * (1/s) ; }
__host__ __device__    friend Matrix3x3 operator * (float s, const Matrix3x3 &m1) { return m1*s ; }
__host__ __device__    friend Vec3 operator * (const Matrix3x3 &m1, const Vec3 &v) ;
__host__ __device__    friend Vec3 operator * (const Vec3 &v, const Matrix3x3 &m1) ;

__host__ __device__    Matrix3x3 &operator += (const Matrix3x3 &m) ;
__host__ __device__    Matrix3x3 &operator -= (const Matrix3x3 &m) ;

__host__ __device__    Matrix3x3 &operator *= (const Matrix3x3 &m) ;
__host__ __device__    Matrix3x3 &operator *= (float s) ;
__host__ __device__    Matrix3x3 &operator /= (float s) ;

__host__ __device__    friend Matrix3x3 operator - (const Matrix3x3 &);

    // Compute the inverse (inv) and determinant (det) of a matrix
__host__ __device__    Matrix3x3 inverse(bool *invertible = nullptr) const ;
__host__ __device__    void invert(bool *invertible) ;

    // Compute the transpose (float) of a matrix
__host__ __device__    Matrix3x3 transpose() const ;
__host__ __device__    void tranpose() ;

    // Return the determinant
__host__ __device__    float det() const ;

__host__ __device__    static Matrix3x3 rotationX(float a) ;
__host__ __device__    static Matrix3x3 rotationY(float a) ;
__host__ __device__    static Matrix3x3 rotationZ(float a) ;
__host__ __device__    static Matrix3x3 rotationAxisAngle(const Vec3 &axis, float a) ;

protected:

    // Matrix elements

    float m_[3][3] ;
} ;

inline Matrix3x3 Matrix3x3::rotationX(float a) {
   const float c = cos(a), s = sin(a);
   return { 1.f, 0.f, 0.f,
            0.f,   c,  -s,
            0.f,   s,   c } ;
}

inline Matrix3x3 Matrix3x3::rotationY(float a) {
   const float c = cos(a), s = sin(a);
   return {   c,   0.f,    s,
            0.f,   1.f,  0.f,
             -s,   0.f,   c } ;
}

inline Matrix3x3 Matrix3x3::rotationZ(float a) {
   const float c = cos(a), s = sin(a);
   return {   c,   s,  0.f,
              s,   c,  0.f,
            0.f, 0.f,  1.f } ;
}

inline Matrix3x3 Matrix3x3::rotationAxisAngle(const Vec3 &v, float a) {
    float s = sin(a), c = cos(a), c_1 = 1-c,
         xx = v.x()*v.x()*c_1, xy = v.x()*v.y()*c_1, xz = v.x()*v.z()*c_1,
         yy = v.y()*v.y()*c_1, yz = v.y()*v.z()*c_1, zz = v.z()*v.z()*c_1,
         xs = v.x()*s, ys = v.y()*s, zs = v.z()*s;
    return { xx+c, xy-zs, xz+ys,
            xy+zs, yy+c,  yz-xs,
            xz-ys, yz+xs,  zz+c };
}


#include <htrac/util/matrix3x3.inl>


#endif
