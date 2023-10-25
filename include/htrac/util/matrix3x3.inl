
inline Matrix3x3::Matrix3x3(float m11, float m12, float m13,
                            float m21, float m22, float m23,
                            float m31, float m32, float m33 ) {
    m_[0][0] = m11 ; m_[0][1] = m12 ; m_[0][2] = m13 ;
    m_[1][0] = m21 ; m_[1][1] = m22 ; m_[1][2] = m23 ;
    m_[2][0] = m31 ; m_[2][1] = m32 ; m_[2][2] = m33 ;
}


inline float& Matrix3x3::operator() (int i, int j) {
    assert((0<=i) && (i<=2) && (0<=j) && (j<=2));
    return m_[i][j] ;
}
inline float Matrix3x3::operator() (int i, int j) const {
    assert((0<=i) && (i<=2) && (0<=j) && (j<=2));
    return m_[i][j] ;
}

inline float *Matrix3x3::operator[] (int i) {
    assert((0<=i) && (i<=2));
    return m_[i] ;
}


inline const float *Matrix3x3::operator[] (int i) const {
    assert((0<=i) && (i<=2));
    return m_[i];
}

inline void Matrix3x3::setZero() {
    m_[0][0] = m_[0][1] = m_[0][2] =
            m_[1][0] = m_[1][1] = m_[1][2] =
            m_[2][0] = m_[2][1] = m_[2][2] = 0.f ;
}

inline void Matrix3x3::setIdentity()
{
    m_[0][0] = m_[1][1] = m_[2][2] = 1.f ;

    m_[0][1] = m_[0][2] =
    m_[1][0] = m_[1][2] =
    m_[2][0] = m_[2][1] = 0.f ;
}

inline Matrix3x3 Matrix3x3::identity() {
    return { 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f } ;
}


inline Matrix3x3::Matrix3x3(const Matrix3x3 &m) {
    for( size_t i = 0 ; i < 3 ; i++ )
        for( size_t j = 0 ; j < 3 ; j++ )
            m_[i][j] = m.m_[i][j] ;
}


inline Matrix3x3 &Matrix3x3::operator = (const Matrix3x3 &m) {
    for( size_t i = 0 ; i < 3 ; i++ )
        for( size_t j = 0 ; j < 3 ; j++ )
            m_[i][j] = m.m_[i][j] ;

    return *this ;
}


inline Matrix3x3::Matrix3x3(const Vec3 &v1, const Vec3 &v2, const Vec3 &v3) {
    m_[0][0] = v1.x() ; m_[0][1] = v2.x() ; m_[0][2] = v3.x() ;
    m_[1][0] = v1.y() ; m_[1][1] = v2.y() ; m_[1][2] = v3.y() ;
    m_[2][0] = v1.z() ; m_[2][1] = v2.z() ; m_[2][2] = v3.z() ;
}


inline Matrix3x3 Matrix3x3::outer(const Vec3 &v1, const Vec3 &v2) {
    return {
        v1.x() * v2.x(), v1.x() * v2.y(), v1.x() * v2.z(),
                v1.y() * v2.x(), v1.y() * v2.y(), v1.y() * v2.z(),
                v1.z() * v2.x(), v1.z() * v2.y(), v1.z() * v2.z()
    };
}


inline std::ostream &operator << (std::ostream &strm, const Matrix3x3 &m) {
    strm << m.m_[0][0] << ' ' << m.m_[0][1] << ' ' << m.m_[0][2] << std::endl ;
    strm << m.m_[1][0] << ' ' << m.m_[1][1] << ' ' << m.m_[1][2] << std::endl ;
    strm << m.m_[2][0] << ' ' << m.m_[2][1] << ' ' << m.m_[2][2] << std::endl ;

    return strm ;
}


inline Matrix3x3 operator + (const Matrix3x3 &m1, const Matrix3x3 &m2) {
    return {
        m1.m_[0][0] + m2.m_[0][0], m1.m_[0][1] + m2.m_[0][1], m1.m_[0][2] + m2.m_[0][2],
                m1.m_[1][0] + m2.m_[1][0], m1.m_[1][1] + m2.m_[1][1], m1.m_[1][2] + m2.m_[1][2],
                m1.m_[2][0] + m2.m_[2][0], m1.m_[2][1] + m2.m_[2][1], m1.m_[2][2] + m2.m_[2][2]
    };
}


inline Matrix3x3 operator - (const Matrix3x3 &m1, const Matrix3x3 &m2) {
    return {
        m1.m_[0][0] - m2.m_[0][0], m1.m_[0][1] - m2.m_[0][1], m1.m_[0][2] - m2.m_[0][2],
                m1.m_[1][0] - m2.m_[1][0], m1.m_[1][1] - m2.m_[1][1], m1.m_[1][2] - m2.m_[1][2],
                m1.m_[2][0] - m2.m_[2][0], m1.m_[2][1] - m2.m_[2][1], m1.m_[2][2] - m2.m_[2][2]
    };
}


inline Matrix3x3 operator - (const Matrix3x3 &m) {
    return {-m.m_[0][0], -m.m_[0][1], -m.m_[0][2], -m.m_[1][0], -m.m_[1][1], -m.m_[1][2], -m.m_[2][0], -m.m_[2][1], -m.m_[2][2]} ;
}

inline Matrix3x3 Matrix3x3::transpose() const {
    return { m_[0][0], m_[1][0], m_[2][0], m_[0][1], m_[1][1], m_[2][1], m_[0][2], m_[1][2], m_[2][2] } ;
}


inline Vec3 Matrix3x3::row(int r) const {
    return { m_[r][0], m_[r][1], m_[r][2] } ;
}


inline Vec3 Matrix3x3::column(int c) const {
    return { m_[0][c], m_[1][c], m_[2][c] } ;
}


inline void Matrix3x3::setRow(int r, const Vec3 &v) {
    m_[r][0] = v.x() ;
    m_[r][1] = v.y() ;
    m_[r][2] = v.z() ;
}


inline void Matrix3x3::setColumn(int c, const Vec3 &v) {
    m_[0][c] = v.x() ;
    m_[1][c] = v.y() ;
    m_[2][c] = v.z() ;
}


inline Matrix3x3::Matrix3x3(const Vec3 &diag) {
    m_[0][0] = diag.x() ; m_[1][1] = diag.y() ; m_[2][2] = diag.z() ;
}

inline Matrix3x3 operator * (const Matrix3x3 &m1, float s) {
    return {
        m1.m_[0][0] * s, m1.m_[0][1] * s, m1.m_[0][2] * s,
                m1.m_[1][0] * s, m1.m_[1][1] * s, m1.m_[1][2] * s,
                m1.m_[2][0] * s, m1.m_[2][1] * s, m1.m_[2][2] * s
    };
}


inline Vec3 operator * (const Matrix3x3 &m1, const Vec3 &v) {
    float s1, s2, s3, t1, t2, t3 ;

    s1 = v.x() ; s2 = v.y() ; s3 = v.z() ;

    t1 = m1.m_[0][0] * s1 + m1.m_[0][1] * s2 + m1.m_[0][2] * s3 ;
    t2 = m1.m_[1][0] * s1 + m1.m_[1][1] * s2 + m1.m_[1][2] * s3 ;
    t3 = m1.m_[2][0] * s1 + m1.m_[2][1] * s2 + m1.m_[2][2] * s3 ;

    return { t1, t2, t3 } ;
}


inline Vec3 operator * (const Vec3 &v, const Matrix3x3 &m1) {
    float s1, s2, s3, t1, t2, t3 ;

    s1 = v.x() ; s2 = v.y() ; s3 = v.z() ;

    t1 = s1 * m1.m_[0][0] + s2 * m1.m_[1][0] + s3 * m1.m_[2][0] ;
    t2 = s1 * m1.m_[0][1] + s2 * m1.m_[1][1] + s3 * m1.m_[2][1] ;
    t3 = s1 * m1.m_[0][2] + s2 * m1.m_[1][2] + s3 * m1.m_[2][2] ;

    return { t1, t2, t3 } ;
}


inline Matrix3x3 &Matrix3x3::operator += (const Matrix3x3 &m2) {
    m_[0][0] += m2.m_[0][0] ;  m_[0][1] += m2.m_[0][1] ;  m_[0][2] += m2.m_[0][2] ;
    m_[1][0] += m2.m_[1][0] ;  m_[1][1] += m2.m_[1][1] ;  m_[1][2] += m2.m_[1][2] ;
    m_[2][0] += m2.m_[2][0] ;  m_[2][1] += m2.m_[2][1] ;  m_[2][2] += m2.m_[2][2] ;

    return *this ;
}


inline Matrix3x3 &Matrix3x3::operator -= (const Matrix3x3 &m2) {
    m_[0][0] -= m2.m_[0][0] ;  m_[0][1] -= m2.m_[0][1] ;  m_[0][2] -= m2.m_[0][2] ;
    m_[1][0] -= m2.m_[1][0] ;  m_[1][1] -= m2.m_[1][1] ;  m_[1][2] -= m2.m_[1][2] ;
    m_[2][0] -= m2.m_[2][0] ;  m_[2][1] -= m2.m_[2][1] ;  m_[2][2] -= m2.m_[2][2] ;

    return *this ;
}


inline Matrix3x3 &Matrix3x3::operator *= (float s) {
    m_[0][0] *= s ; m_[1][0] *= s ; m_[2][0] *= s ;
    m_[0][1] *= s ; m_[1][1] *= s ; m_[2][1] *= s ;
    m_[0][2] *= s ; m_[1][2] *= s ; m_[2][2] *= s ;

    return *this ;
}

inline Matrix3x3 operator * (const Matrix3x3 &m1, const Matrix3x3 &m2) {
    Matrix3x3 t ;
    double s1, s2, s3 ;

    s1 = m2.m_[0][0] ; s2 = m2.m_[1][0] ; s3 = m2.m_[2][0] ;

    t.m_[0][0] = m1.m_[0][0] * s1 + m1.m_[0][1] * s2 + m1.m_[0][2] * s3 ;
    t.m_[1][0] = m1.m_[1][0] * s1 + m1.m_[1][1] * s2 + m1.m_[1][2] * s3 ;
    t.m_[2][0] = m1.m_[2][0] * s1 + m1.m_[2][1] * s2 + m1.m_[2][2] * s3 ;

    s1 = m2.m_[0][1] ; s2 = m2.m_[1][1] ; s3 = m2.m_[2][1] ;

    t.m_[0][1] = m1.m_[0][0] * s1 + m1.m_[0][1] * s2 + m1.m_[0][2] * s3 ;
    t.m_[1][1] = m1.m_[1][0] * s1 + m1.m_[1][1] * s2 + m1.m_[1][2] * s3 ;
    t.m_[2][1] = m1.m_[2][0] * s1 + m1.m_[2][1] * s2 + m1.m_[2][2] * s3 ;

    s1 = m2.m_[0][2] ; s2 = m2.m_[1][2] ; s3 = m2.m_[2][2] ;

    t.m_[0][2] = m1.m_[0][0] * s1 + m1.m_[0][1] * s2 + m1.m_[0][2] * s3 ;
    t.m_[1][2] = m1.m_[1][0] * s1 + m1.m_[1][1] * s2 + m1.m_[1][2] * s3 ;
    t.m_[2][2] = m1.m_[2][0] * s1 + m1.m_[2][1] * s2 + m1.m_[2][2] * s3 ;

    return t ;
}

inline Matrix3x3 &Matrix3x3::operator *= (const Matrix3x3 &m2)
{
    double s1, s2, s3 ;
    Matrix3x3 t ;

    s1 = m2.m_[0][0] ; s2 = m2.m_[1][0] ; s3 = m2.m_[2][0] ;

    t.m_[0][0] = m_[0][0] * s1 + m_[0][1] * s2 + m_[0][2] * s3 ;
    t.m_[1][0] = m_[1][0] * s1 + m_[1][1] * s2 + m_[1][2] * s3 ;
    t.m_[2][0] = m_[2][0] * s1 + m_[2][1] * s2 + m_[2][2] * s3 ;

    s1 = m2.m_[0][1] ; s2 = m2.m_[1][1] ; s3 = m2.m_[2][1] ;

    t.m_[0][1] = m_[0][0] * s1 + m_[0][1] * s2 + m_[0][2] * s3 ;
    t.m_[1][1] = m_[1][0] * s1 + m_[1][1] * s2 + m_[1][2] * s3 ;
    t.m_[2][1] = m_[2][0] * s1 + m_[2][1] * s2 + m_[2][2] * s3 ;

    s1 = m2.m_[0][2] ; s2 = m2.m_[1][2] ; s3 = m2.m_[2][2] ;

    t.m_[0][2] = m_[0][0] * s1 + m_[0][1] * s2 + m_[0][2] * s3 ;
    t.m_[1][2] = m_[1][0] * s1 + m_[1][1] * s2 + m_[1][2] * s3 ;
    t.m_[2][2] = m_[2][0] * s1 + m_[2][1] * s2 + m_[2][2] * s3 ;

    *this = t ;

    return *this ;
}



inline Matrix3x3 &Matrix3x3::operator /= (float s)
{
    assert(fabs(s) > std::numeric_limits<float>::min() ) ;

    m_[0][0] /= s ; m_[1][0] /= s ; m_[2][0] /= s ;
    m_[0][1] /= s ; m_[1][1] /= s ; m_[2][1] /= s ;
    m_[0][2] /= s ; m_[1][2] /= s ; m_[2][2] /= s ;

    return *this ;
}

#define det2x2(a, b, c, d) ((a)*(d) - (b)*(c))

inline float Matrix3x3::det() const
{
    return m_[0][0] * det2x2( m_[1][1], m_[1][2], m_[2][1], m_[2][2] )
            - m_[1][0] * det2x2( m_[0][1], m_[0][2], m_[2][1], m_[2][2] )
            + m_[2][0] * det2x2( m_[0][1], m_[0][2], m_[1][1], m_[1][2] );
}

inline Matrix3x3 Matrix3x3::inverse(bool *invertible) const
{
    Matrix3x3 inv ;
    inv.m_[0][0] =   det2x2(m_[1][1], m_[1][2], m_[2][1], m_[2][2]) ;
    inv.m_[1][0] = - det2x2(m_[1][0], m_[1][2], m_[2][0], m_[2][2]) ;
    inv.m_[2][0] =   det2x2(m_[1][0], m_[1][1], m_[2][0], m_[2][1]) ;
    inv.m_[0][1] = - det2x2(m_[0][1], m_[0][2], m_[2][1], m_[2][2]) ;
    inv.m_[1][1] =   det2x2(m_[0][0], m_[0][2], m_[2][0], m_[2][2]) ;
    inv.m_[2][1] = - det2x2(m_[0][0], m_[0][1], m_[2][0], m_[2][1]) ;
    inv.m_[0][2] =   det2x2(m_[0][1], m_[0][2], m_[1][1], m_[1][2]) ;
    inv.m_[1][2] = - det2x2(m_[0][0], m_[0][2], m_[1][0], m_[1][2]) ;
    inv.m_[2][2] =   det2x2(m_[0][0], m_[0][1], m_[1][0], m_[1][1]) ;

    float det = m_[0][0] * inv.m_[0][0] + m_[1][0] * inv.m_[0][1] + m_[2][0] * inv.m_[0][2] ;

    if ( fabs(det) < std::numeric_limits<float>::epsilon() ) {
        if ( invertible ) *invertible = false ;
        return Matrix3x3::identity() ;
    } else {
        if ( invertible ) *invertible = true ;
        return inv/det ;
    }
}
