
inline Matrix4x4::Matrix4x4(float a11, float a12, float a13, float a14,
                            float a21, float a22, float a23, float a24,
                            float a31, float a32, float a33, float a34,
                            float a41, float a42, float a43, float a44) {
    m_[0][0] = a11 ; m_[0][1] = a12 ; m_[0][2] = a13 ; m_[0][3] = a14 ;
    m_[1][0] = a21 ; m_[1][1] = a22 ; m_[1][2] = a23 ; m_[1][3] = a24 ;
    m_[2][0] = a31 ; m_[2][1] = a32 ; m_[2][2] = a33 ; m_[2][3] = a34 ;
    m_[3][0] = a41 ; m_[3][1] = a42 ; m_[3][2] = a43 ; m_[3][3] = a44 ;
}


inline float& Matrix4x4::operator() (int i, int j) {
    assert((0<=i) && (i<=3) && (0<=j) && (j<=3));
    return m_[i][j] ;
}


inline const float& Matrix4x4::operator() (int i, int j) const {
    assert((0<=i) && (i<=3) && (0<=j) && (j<=3));
    return m_[i][j] ;
}

inline float *Matrix4x4::operator[] (int i) {
    assert((0<=i) && (i<=3));
    return m_[i] ;
}


inline const float *Matrix4x4::operator[] (int i) const {
    assert((0<=i) && (i<=3));
    return m_[i];
}

inline void Matrix4x4::setZero() {
    for(int i=0 ; i<4 ; i++ )
        for( int j=0 ; j<4 ; j++ )
            m_[i][j] = 0.0f ;
}

inline void Matrix4x4::setIdentity()
{
    m_[0][0] = m_[1][1] = m_[2][2] = m_[3][3] = 1.f ;

    m_[0][1] = m_[0][2] = m_[0][3] =
            m_[1][0] = m_[1][2] = m_[1][3] =
            m_[2][0] = m_[2][1] = m_[2][3] =
            m_[3][0] = m_[3][1] = m_[3][4] = 0.f ;
}

inline Matrix4x4 Matrix4x4::identity() {
    return { 1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f } ;
}


inline Matrix4x4::Matrix4x4(const Matrix4x4 &m) {
    for( size_t i = 0 ; i < 4 ; i++ )
        for( size_t j = 0 ; j < 4 ; j++ )
            m_[i][j] = m.m_[i][j] ;
}

inline Matrix4x4::Matrix4x4(const Eigen::Matrix4f &m) {
    for( size_t i = 0 ; i < 4 ; i++ )
        for( size_t j = 0 ; j < 4 ; j++ )
            m_[i][j] = m(i, j) ;
}

inline Matrix4x4 &Matrix4x4::operator = (const Matrix4x4 &m) {
    for( size_t i = 0 ; i < 4 ; i++ )
        for( size_t j = 0 ; j < 4 ; j++ )
            m_[i][j] = m.m_[i][j] ;

    return *this ;
}

inline Matrix4x4 &Matrix4x4::operator = (const Eigen::Matrix4f &m) {
    for( size_t i = 0 ; i < 4 ; i++ )
        for( size_t j = 0 ; j < 4 ; j++ )
            m_[i][j] = m(i, j) ;

    return *this ;
}


inline Matrix4x4::Matrix4x4(const Vec4 &v1, const Vec4 &v2, const Vec4 &v3, const Vec4 &v4) {
    m_[0][0] = v1.x() ; m_[0][1] = v2.x() ; m_[0][2] = v3.x() ; m_[0][3] = v4.x() ;
    m_[1][0] = v1.y() ; m_[1][1] = v2.y() ; m_[1][2] = v3.y() ; m_[1][3] = v4.y() ;
    m_[2][0] = v1.z() ; m_[2][1] = v2.z() ; m_[2][2] = v3.z() ; m_[2][3] = v4.z() ;
    m_[3][0] = v1.w() ; m_[3][1] = v2.w() ; m_[3][2] = v4.w() ; m_[3][3] = v4.w() ;
}


inline std::ostream &operator << (std::ostream &strm, const Matrix4x4 &m) {
    strm << m.m_[0][0] << ' ' << m.m_[0][1] << ' ' << m.m_[0][2] << ' ' << m.m_[0][3] << std::endl ;
    strm << m.m_[1][0] << ' ' << m.m_[1][1] << ' ' << m.m_[1][2] << ' ' << m.m_[1][3] << std::endl ;
    strm << m.m_[2][0] << ' ' << m.m_[2][1] << ' ' << m.m_[2][2] << ' ' << m.m_[2][3] << std::endl ;
    strm << m.m_[3][0] << ' ' << m.m_[3][1] << ' ' << m.m_[3][2] << ' ' << m.m_[3][3] << std::endl ;

    return strm ;
}

inline Matrix4x4::Matrix4x4(const Matrix3x3 &m)
{
    m_[0][0] = m[0][0] ; m_[0][1] = m[0][1] ; m_[0][2] = m[0][2] ; m_[0][3] = 0.0 ;
    m_[1][0] = m[1][0] ; m_[1][1] = m[1][1] ; m_[1][2] = m[1][2] ; m_[1][3] = 0.0 ;
    m_[2][0] = m[2][0] ; m_[2][1] = m[2][1] ; m_[2][2] = m[2][2] ; m_[2][3] = 0.0 ;
    m_[3][0] = 0.0   ; m_[3][1] = 0.0   ; m_[3][2] = 0.0   ; m_[3][3] = 1.0 ;
}

inline Matrix4x4::Matrix4x4(const Matrix3x3 &m, const Vec3 &v) {
    m_[0][0] = m[0][0] ; m_[0][1] = m[0][1] ; m_[0][2] = m[0][2] ; m_[0][3] = v.x() ;
    m_[1][0] = m[1][0] ; m_[1][1] = m[1][1] ; m_[1][2] = m[1][2] ; m_[1][3] = v.y() ;
    m_[2][0] = m[2][0] ; m_[2][1] = m[2][1] ; m_[2][2] = m[2][2] ; m_[2][3] = v.z() ;
    m_[3][0] = 0.0   ; m_[3][1] = 0.0   ; m_[3][2] = 0.0   ; m_[3][3] = 1.0 ;
}

inline Matrix4x4 operator + (const Matrix4x4 &m1, const Matrix4x4 &m2) {
    Matrix4x4 T ;

    T.m_[0][0] = m1.m_[0][0] + m2.m_[0][0] ;  T.m_[0][1] = m1.m_[0][1] + m2.m_[0][1] ;  T.m_[0][2] = m1.m_[0][2] + m2.m_[0][2] ; T.m_[0][3] = m1.m_[0][3] + m2.m_[0][3] ;
    T.m_[1][0] = m1.m_[1][0] + m2.m_[1][0] ;  T.m_[1][1] = m1.m_[1][1] + m2.m_[1][1] ;  T.m_[1][2] = m1.m_[1][2] + m2.m_[1][2] ; T.m_[1][3] = m1.m_[1][3] + m2.m_[1][3] ;
    T.m_[2][0] = m1.m_[2][0] + m2.m_[2][0] ;  T.m_[2][1] = m1.m_[2][1] + m2.m_[2][1] ;  T.m_[2][2] = m1.m_[2][2] + m2.m_[2][2] ; T.m_[2][3] = m1.m_[2][3] + m2.m_[2][3] ;
    T.m_[3][0] = m1.m_[3][0] + m2.m_[3][0] ;  T.m_[3][1] = m1.m_[3][1] + m2.m_[3][1] ;  T.m_[3][2] = m1.m_[3][2] + m2.m_[3][2] ; T.m_[3][3] = m1.m_[3][3] + m2.m_[3][3] ;

    return T ;
}


inline Matrix4x4 operator - (const Matrix4x4 &m1, const Matrix4x4 &m2) {
    Matrix4x4 T ;

    T.m_[0][0] = m1.m_[0][0] - m2.m_[0][0] ;  T.m_[0][1] = m1.m_[0][1] - m2.m_[0][1] ;  T.m_[0][2] = m1.m_[0][2] - m2.m_[0][2] ; T.m_[0][3] = m1.m_[0][3] - m2.m_[0][3] ;
    T.m_[1][0] = m1.m_[1][0] - m2.m_[1][0] ;  T.m_[1][1] = m1.m_[1][1] - m2.m_[1][1] ;  T.m_[1][2] = m1.m_[1][2] - m2.m_[1][2] ; T.m_[1][3] = m1.m_[1][3] - m2.m_[1][3] ;
    T.m_[2][0] = m1.m_[2][0] - m2.m_[2][0] ;  T.m_[2][1] = m1.m_[2][1] - m2.m_[2][1] ;  T.m_[2][2] = m1.m_[2][2] - m2.m_[2][2] ; T.m_[2][3] = m1.m_[2][3] - m2.m_[2][3] ;
    T.m_[3][0] = m1.m_[3][0] - m2.m_[3][0] ;  T.m_[3][1] = m1.m_[3][1] - m2.m_[3][1] ;  T.m_[3][2] = m1.m_[3][2] - m2.m_[3][2] ; T.m_[3][3] = m1.m_[3][3] - m2.m_[3][3] ;

    return T ;
}



inline Matrix4x4 operator - (const Matrix4x4 &m) {
    return {-m.m_[0][0], -m.m_[0][1], -m.m_[0][2], -m.m_[0][3],
                -m.m_[1][0], -m.m_[1][1], -m.m_[1][2], -m.m_[1][3],
                -m.m_[2][0], -m.m_[2][1], -m.m_[2][2], -m.m_[2][3],
                -m.m_[3][0], -m.m_[3][1], -m.m_[3][2], -m.m_[3][3]} ;
}

inline Matrix4x4 Matrix4x4::transpose() const {
    return { m_[0][0], m_[1][0], m_[2][0], m_[3][0],
                m_[0][1], m_[1][1], m_[2][1], m_[3][1],
                m_[0][2], m_[1][2], m_[2][2], m_[3][2],
                m_[0][3], m_[1][3], m_[2][3], m_[3][3]   } ;
}


inline Vec4 Matrix4x4::row(int r) const {
    return { m_[r][0], m_[r][1], m_[r][2], m_[r][3] } ;
}


inline Vec4 Matrix4x4::column(int c) const {
    return { m_[0][c], m_[1][c], m_[2][c], m_[3][c] } ;
}


inline void Matrix4x4::setRow(int r, const Vec4 &v) {
    m_[r][0] = v.x() ;
    m_[r][1] = v.y() ;
    m_[r][2] = v.z() ;
    m_[r][3] = v.w() ;
}


inline void Matrix4x4::setColumn(int c, const Vec4 &v) {
    m_[0][c] = v.x() ;
    m_[1][c] = v.y() ;
    m_[2][c] = v.z() ;
    m_[3][c] = v.w() ;
}


inline Matrix4x4::Matrix4x4(const Vec4 &diag) {
    m_[0][0] = diag.x() ; m_[1][1] = diag.y() ; m_[2][2] = diag.z() ; m_[3][3] = diag.w() ;
}

inline Matrix4x4 operator * (const Matrix4x4 &m1, float s) {
    return {
        m1.m_[0][0] * s, m1.m_[0][1] * s, m1.m_[0][2] * s, m1.m_[0][3] * s,
                m1.m_[1][0] * s, m1.m_[1][1] * s, m1.m_[1][2] * s, m1.m_[1][3] * s,
                m1.m_[2][0] * s, m1.m_[2][1] * s, m1.m_[2][2] * s, m1.m_[2][3] * s,
                m1.m_[3][0] * s, m1.m_[3][1] * s, m1.m_[3][2] * s, m1.m_[3][3] * s
    };
}


inline Matrix4x4 &Matrix4x4::operator += (const Matrix4x4 &m2)
{
    m_[0][0] += m2.m_[0][0] ;  m_[0][1] += m2.m_[0][1] ;  m_[0][2] += m2.m_[0][2] ; m_[0][3] += m2.m_[0][3] ;
    m_[1][0] += m2.m_[1][0] ;  m_[1][1] += m2.m_[1][1] ;  m_[1][2] += m2.m_[1][2] ; m_[1][3] += m2.m_[1][3] ;
    m_[2][0] += m2.m_[2][0] ;  m_[2][1] += m2.m_[2][1] ;  m_[2][2] += m2.m_[2][2] ; m_[2][3] += m2.m_[2][3] ;
    m_[3][0] += m2.m_[3][0] ;  m_[3][1] += m2.m_[3][1] ;  m_[3][2] += m2.m_[3][2] ; m_[3][3] += m2.m_[3][3] ;

    return *this ;
}


inline Matrix4x4 &Matrix4x4::operator -= (const Matrix4x4 &m2)
{
    m_[0][0] -= m2.m_[0][0] ;  m_[0][1] -= m2.m_[0][1] ;  m_[0][2] -= m2.m_[0][2] ; m_[0][3] -= m2.m_[0][3] ;
    m_[1][0] -= m2.m_[1][0] ;  m_[1][1] -= m2.m_[1][1] ;  m_[1][2] -= m2.m_[1][2] ; m_[1][3] -= m2.m_[1][3] ;
    m_[2][0] -= m2.m_[2][0] ;  m_[2][1] -= m2.m_[2][1] ;  m_[2][2] -= m2.m_[2][2] ; m_[2][3] -= m2.m_[2][3] ;
    m_[3][0] -= m2.m_[3][0] ;  m_[3][1] -= m2.m_[3][1] ;  m_[3][2] -= m2.m_[3][2] ; m_[3][3] -= m2.m_[3][3] ;

    return *this ;
}


inline Matrix4x4 &Matrix4x4::operator *= (float s)
{
    m_[0][0] *= s ; m_[1][0] *= s ; m_[2][0] *= s ; m_[3][0] *= s ;
    m_[0][1] *= s ; m_[1][1] *= s ; m_[2][1] *= s ; m_[3][1] *= s ;
    m_[0][2] *= s ; m_[1][2] *= s ; m_[2][2] *= s ; m_[3][2] *= s ;
    m_[0][3] *= s ; m_[1][3] *= s ; m_[2][3] *= s ; m_[3][3] *= s ;

    return *this ;
}

inline Matrix4x4 &Matrix4x4::operator /= (float s)
{
    *this *= 1/s ;
    return *this ;
}


inline Matrix4x4 operator * (const Matrix4x4 &m1, const Matrix4x4 &m2)
{
    Matrix4x4 t ;
    double s1, s2, s3, s4 ;

    s1 = m2.m_[0][0] ; s2 = m2.m_[1][0] ; s3 = m2.m_[2][0] ; s4 = m2.m_[3][0] ;

    t.m_[0][0] = m1.m_[0][0] * s1 + m1.m_[0][1] * s2 + m1.m_[0][2] * s3 + m1.m_[0][3] * s4 ;
    t.m_[1][0] = m1.m_[1][0] * s1 + m1.m_[1][1] * s2 + m1.m_[1][2] * s3 + m1.m_[1][3] * s4 ;
    t.m_[2][0] = m1.m_[2][0] * s1 + m1.m_[2][1] * s2 + m1.m_[2][2] * s3 + m1.m_[2][3] * s4 ;
    t.m_[3][0] = m1.m_[3][0] * s1 + m1.m_[3][1] * s2 + m1.m_[3][2] * s3 + m1.m_[3][3] * s4 ;

    s1 = m2.m_[0][1] ; s2 = m2.m_[1][1] ; s3 = m2.m_[2][1] ; s4 = m2.m_[3][1] ;

    t.m_[0][1] = m1.m_[0][0] * s1 + m1.m_[0][1] * s2 + m1.m_[0][2] * s3 + m1.m_[0][3] * s4 ;
    t.m_[1][1] = m1.m_[1][0] * s1 + m1.m_[1][1] * s2 + m1.m_[1][2] * s3 + m1.m_[1][3] * s4 ;
    t.m_[2][1] = m1.m_[2][0] * s1 + m1.m_[2][1] * s2 + m1.m_[2][2] * s3 + m1.m_[2][3] * s4 ;
    t.m_[3][1] = m1.m_[3][0] * s1 + m1.m_[3][1] * s2 + m1.m_[3][2] * s3 + m1.m_[3][3] * s4 ;

    s1 = m2.m_[0][2] ; s2 = m2.m_[1][2] ; s3 = m2.m_[2][2] ; s4 = m2.m_[3][2] ;

    t.m_[0][2] = m1.m_[0][0] * s1 + m1.m_[0][1] * s2 + m1.m_[0][2] * s3 + m1.m_[0][3] * s4 ;
    t.m_[1][2] = m1.m_[1][0] * s1 + m1.m_[1][1] * s2 + m1.m_[1][2] * s3 + m1.m_[1][3] * s4 ;
    t.m_[2][2] = m1.m_[2][0] * s1 + m1.m_[2][1] * s2 + m1.m_[2][2] * s3 + m1.m_[2][3] * s4 ;
    t.m_[3][2] = m1.m_[3][0] * s1 + m1.m_[3][1] * s2 + m1.m_[3][2] * s3 + m1.m_[3][3] * s4 ;

    s1 = m2.m_[0][3] ; s2 = m2.m_[1][3] ; s3 = m2.m_[2][3] ; s4 = m2.m_[3][3] ;

    t.m_[0][3] = m1.m_[0][0] * s1 + m1.m_[0][1] * s2 + m1.m_[0][2] * s3 + m1.m_[0][3] * s4 ;
    t.m_[1][3] = m1.m_[1][0] * s1 + m1.m_[1][1] * s2 + m1.m_[1][2] * s3 + m1.m_[1][3] * s4 ;
    t.m_[2][3] = m1.m_[2][0] * s1 + m1.m_[2][1] * s2 + m1.m_[2][2] * s3 + m1.m_[2][3] * s4 ;
    t.m_[3][3] = m1.m_[3][0] * s1 + m1.m_[3][1] * s2 + m1.m_[3][2] * s3 + m1.m_[3][3] * s4 ;

    return t ;
}



inline Vec4 operator * (const Matrix4x4 &m1, const Vec4 &v) {
    float s1, s2, s3, s4, t1, t2, t3, t4 ;

    s1 = v.x() ; s2 = v.y() ; s3 = v.z() ; s4 = v.w() ;

    t1 = m1.m_[0][0] * s1 + m1.m_[0][1] * s2 + m1.m_[0][2] * s3 + m1.m_[0][3] * s4 ;
    t2 = m1.m_[1][0] * s1 + m1.m_[1][1] * s2 + m1.m_[1][2] * s3 + m1.m_[1][3] * s4 ;
    t3 = m1.m_[2][0] * s1 + m1.m_[2][1] * s2 + m1.m_[2][2] * s3 + m1.m_[2][3] * s4 ;
    t4 = m1.m_[3][0] * s1 + m1.m_[3][1] * s2 + m1.m_[3][2] * s3 + m1.m_[3][3] * s4 ;

    return { t1, t2, t3, t4 } ;
}

inline Vec3 operator * (const Matrix4x4 &m1, const Vec3 &v) {
    float s1, s2, s3, s4, t1, t2, t3, t4 ;

    s1 = v.x() ; s2 = v.y() ; s3 = v.z() ; s4 = 1.0f;

    t1 = m1.m_[0][0] * s1 + m1.m_[0][1] * s2 + m1.m_[0][2] * s3 + m1.m_[0][3] ;
    t2 = m1.m_[1][0] * s1 + m1.m_[1][1] * s2 + m1.m_[1][2] * s3 + m1.m_[1][3] ;
    t3 = m1.m_[2][0] * s1 + m1.m_[2][1] * s2 + m1.m_[2][2] * s3 + m1.m_[2][3] ;

    return { t1, t2, t3 } ;
}

inline Vec4 operator * (const Vec4 &v, const Matrix4x4 &m1) {
    float s1, s2, s3, s4, t1, t2, t3, t4 ;

    s1 = v.x() ; s2 = v.y() ; s3 = v.z() ; s4 = v.w() ;

    t1 = s1 * m1.m_[0][0] + s2 * m1.m_[1][0] + s3 * m1.m_[2][0] + s4 * m1.m_[3][0] ;
    t2 = s1 * m1.m_[0][1] + s2 * m1.m_[1][1] + s3 * m1.m_[2][1] + s4 * m1.m_[3][1] ;
    t3 = s1 * m1.m_[0][2] + s2 * m1.m_[1][2] + s3 * m1.m_[2][2] + s4 * m1.m_[3][2] ;
    t4 = s1 * m1.m_[0][3] + s2 * m1.m_[1][3] + s3 * m1.m_[2][3] + s4 * m1.m_[3][3] ;

    return Vec4(t1, t2, t3, t4) ;
}

inline Matrix4x4 &Matrix4x4::operator *= (const Matrix4x4 &m2)
{
    Matrix4x4 t ;
    float s1, s2, s3, s4 ;

    s1 = m2.m_[0][0] ; s2 = m2.m_[1][0] ; s3 = m2.m_[2][0] ; s4 = m2.m_[3][0] ;

    t.m_[0][0] = m_[0][0] * s1 + m_[0][1] * s2 + m_[0][2] * s3 + m_[0][3] * s4 ;
    t.m_[1][0] = m_[1][0] * s1 + m_[1][1] * s2 + m_[1][2] * s3 + m_[1][3] * s4 ;
    t.m_[2][0] = m_[2][0] * s1 + m_[2][1] * s2 + m_[2][2] * s3 + m_[2][3] * s4 ;
    t.m_[3][0] = m_[3][0] * s1 + m_[3][1] * s2 + m_[3][2] * s3 + m_[3][3] * s4 ;

    s1 = m2.m_[0][1] ; s2 = m2.m_[1][1] ; s3 = m2.m_[2][1] ; s4 = m2.m_[3][1] ;

    t.m_[0][1] = m_[0][0] * s1 + m_[0][1] * s2 + m_[0][2] * s3 + m_[0][3] * s4 ;
    t.m_[1][1] = m_[1][0] * s1 + m_[1][1] * s2 + m_[1][2] * s3 + m_[1][3] * s4 ;
    t.m_[2][1] = m_[2][0] * s1 + m_[2][1] * s2 + m_[2][2] * s3 + m_[2][3] * s4 ;
    t.m_[3][1] = m_[3][0] * s1 + m_[3][1] * s2 + m_[3][2] * s3 + m_[3][3] * s4 ;

    s1 = m2.m_[0][2] ; s2 = m2.m_[1][2] ; s3 = m2.m_[2][2] ; s4 = m2.m_[3][2] ;

    t.m_[0][2] = m_[0][0] * s1 + m_[0][1] * s2 + m_[0][2] * s3 + m_[0][3] * s4 ;
    t.m_[1][2] = m_[1][0] * s1 + m_[1][1] * s2 + m_[1][2] * s3 + m_[1][3] * s4 ;
    t.m_[2][2] = m_[2][0] * s1 + m_[2][1] * s2 + m_[2][2] * s3 + m_[2][3] * s4 ;
    t.m_[3][2] = m_[3][0] * s1 + m_[3][1] * s2 + m_[3][2] * s3 + m_[3][3] * s4 ;

    s1 = m2.m_[0][3] ; s2 = m2.m_[1][3] ; s3 = m2.m_[2][3] ; s4 = m2.m_[3][3] ;

    t.m_[0][3] = m_[0][0] * s1 + m_[0][1] * s2 + m_[0][2] * s3 + m_[0][3] * s4 ;
    t.m_[1][3] = m_[1][0] * s1 + m_[1][1] * s2 + m_[1][2] * s3 + m_[1][3] * s4 ;
    t.m_[2][3] = m_[2][0] * s1 + m_[2][1] * s2 + m_[2][2] * s3 + m_[2][3] * s4 ;
    t.m_[3][3] = m_[3][0] * s1 + m_[3][1] * s2 + m_[3][2] * s3 + m_[3][3] * s4 ;

    *this = t ;

    return *this ;
}


#define det2x2(a, b, c, d) ((a)*(d) - (b)*(c))
#define det3x3(a, b, c, d, e, f, g, h, i) (a * det2x2(e,f,h,i) - d * det2x2(b,c,h,i) + g * det2x2(b,c,e,f))

inline float Matrix4x4::det() const {
    return m_[0][0] * det3x3( m_[1][1], m_[1][2], m_[1][3], m_[2][1], m_[2][2], m_[2][3], m_[3][1], m_[3][2], m_[3][3] )
            - m_[0][1] * det3x3( m_[1][0], m_[1][2], m_[1][3], m_[2][0], m_[2][2], m_[2][3], m_[3][0], m_[3][2], m_[3][3] )
            + m_[0][2] * det3x3( m_[1][0], m_[1][1], m_[1][3], m_[2][0], m_[2][1], m_[2][3], m_[3][0], m_[3][1], m_[3][3] )
            - m_[0][3] * det3x3( m_[1][0], m_[1][1], m_[1][2], m_[2][0], m_[2][1], m_[2][2], m_[3][0], m_[3][1], m_[3][2] ) ;
}

inline Matrix4x4 Matrix4x4::inverse(bool *invertible) const
{
    Matrix4x4 inv ;
    float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 ;
    float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15 ;
    float d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15 ;

    s0 = m_[0][0] ;  s1 = m_[1][0] ;  s2 = m_[2][0]  ; s3 = m_[3][0] ;
    s4 = m_[0][1] ;  s5 = m_[1][1] ;  s6 = m_[2][1]  ; s7 = m_[3][1] ;
    s8 = m_[0][2] ;  s9 = m_[1][2] ;  s10 = m_[2][2] ; s11 = m_[3][2] ;
    s12 = m_[0][3] ; s13 = m_[1][3] ; s14 = m_[2][3] ; s15 = m_[3][3] ;

    /* calculate pairs for first 8 elements (cofactors) */
    t0 = s10 * s15 ; t1 = s11 * s14 ;  t2 = s9 * s15  ;
    t3 = s11 * s13 ; t4 = s9 * s14  ;  t5 = s10 * s13 ;
    t6 = s8 * s15  ; t7 = s11 * s12 ;  t8 = s8 * s14  ;
    t9 = s10 * s12 ; t10 = s8 * s13 ;  t11 = s9 * s12 ;

    /* calculate first 8 elements (cofactors) */
    d0 =  t0*s5 + t3*s6 + t4* s7 - t1*s5 - t2*s6 - t5 *s7 ;
    d1 =  t1*s4 + t6*s6 + t9* s7 - t0*s4 - t7*s6 - t8 *s7 ;
    d2 =  t2*s4 + t7*s5 + t10*s7 - t3*s4 - t6*s5 - t11*s7 ;
    d3 =  t5*s4 + t8*s5 + t11*s6 - t4*s4 - t9*s5 - t10*s6 ;
    d4 =  t1*s1 + t2*s2 + t5* s3 - t0*s1 - t3*s2 - t4* s3 ;
    d5 =  t0*s0 + t7*s2 + t8* s3 - t1*s0 - t6*s2 - t9* s3 ;
    d6 =  t3*s0 + t6*s1 + t11*s3 - t2*s0 - t7*s1 - t10*s3 ;
    d7 =  t4*s0 + t9*s1 + t10*s2 - t5*s0 - t8*s1 - t11*s2 ;

    /* calculate pairs for second 8 elements (cofactors) */
    t0  = s2*s7;  t1  = s3*s6;  t2  = s1*s7;  t3  = s3*s5;
    t4  = s1*s6;  t5  = s2*s5;  t6  = s0*s7;  t7  = s3*s4;
    t8  = s0*s6;  t9  = s2*s4;  t10 = s0*s5;  t11 = s1*s4;

    /* calculate second 8 elements (cofactors) */
    d8 =  t0*s13 + t3*s14 + t4*s15 - t1*s13 - t2*s14 - t5*s15 ;
    d9 =  t1*s12 + t6*s14 + t9*s15 - t0*s12 - t7*s14 - t8*s15 ;
    d10 = t2*s12 + t7*s13 + t10*s15 - t3*s12 - t6*s13 - t11*s15 ;
    d11 = t5*s12 + t8*s13 + t11*s14 - t4*s12 - t9*s13 - t10*s14 ;
    d12 = t2*s10 + t5*s11 + t1*s9 - t4*s11 - t0*s9  - t3*s10 ;
    d13 = t8*s11 + t0*s8  + t7*s10 - t6*s10 - t9*s11 - t1*s8 ;
    d14 = t6*s9 + t11*s11 + t3*s8 - t10*s11 - t2*s8 - t7*s9 ;
    d15 = t10*s10 + t4*s8 + t9*s9 - t8*s9 - t11*s10 - t5*s8 ;

    float det = s0*d0 + s1*d1 + s2*d2 + s3*d3 ;

    if ( fabs(det) < std::numeric_limits<float>::epsilon() ) {
        if ( invertible ) *invertible = false ;
        return inv ;
    } else {

        float idet = 1/det;

        inv.m_[0][0] = d0 * idet  ; inv.m_[0][1] = d1 * idet  ; inv.m_[0][2] = d2 * idet  ; inv.m_[0][3] = d3 * idet ;
        inv.m_[1][0] = d4 * idet  ; inv.m_[1][1] = d5 * idet  ; inv.m_[1][2] = d6 * idet  ; inv.m_[1][3] = d7 * idet ;
        inv.m_[2][0] = d8 * idet  ; inv.m_[2][1] = d9 * idet  ; inv.m_[2][2] = d10 * idet ; inv.m_[2][3] = d11 * idet ;
        inv.m_[3][0] = d12 * idet ; inv.m_[3][1] = d13 * idet ; inv.m_[3][2] = d14 * idet ; inv.m_[3][3] = d15 * idet ;

        if ( invertible ) {
            *invertible = true ;
            return inv ;
        }
    }
}
