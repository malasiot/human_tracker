
inline Vec3::Vec3(float v): x_(v), y_(v), z_(v) {
}

inline Vec3::Vec3(float x, float y, float z): x_(x), y_(y), z_(z) {
}

inline Vec3::Vec3(const Vec3 &o): x_(o.x_), y_(o.y_), z_(o.z_) {
}

inline Vec3::Vec3(const Eigen::Vector3f &o): x_(o.x()), y_(o.y()), z_(o.z()) {
}

inline Vec3 &Vec3::operator = (const Vec3 &o) {
    x_ = o.x_ ; y_ = o.y_ ; z_ = o.z_ ;
    return *this ;
}

inline Vec3 &Vec3::operator = (const Eigen::Vector3f &o) {
    x_ = o.x() ; y_ = o.y() ; z_ = o.z() ;
    return *this ;
}

inline float Vec3::x() const {
    return x_ ;
}

inline float &Vec3::x() {
    return x_ ;
}

inline float Vec3::y() const {
    return y_ ;
}

inline float &Vec3::y() {
    return y_ ;
}

inline float Vec3::z() const {
    return z_ ;
}

inline float &Vec3::z() {
    return z_ ;
}

inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.x_ + v2.x_, v1.y_ + v2.y_, v1.z_ + v2.z_) ;
}

inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.x_ - v2.x_, v1.y_ - v2.y_, v1.z_ - v2.z_) ;
}

inline const Vec3 &Vec3::operator +=(const Vec3 &v) {
  x_ += v.x_ ; y_ += v.y_ ; z_ += v.z_ ;
  return *this ;
}

inline const Vec3 &Vec3::operator -=(const Vec3 &v) {
  x_ -= v.x_ ; y_ -= v.y_ ; z_ -= v.z_ ;
  return *this ;
}

inline float dot(const Vec3 &v1, const Vec3 &v2) {
   return v1.x_ * v2.x_ + v1.y_ * v2.y_ + v1.z_ * v2.z_ ;
}

inline float Vec3::dot(const Vec3 &v2) const {
   return x_ * v2.x_ + y_ * v2.y_ + z_ * v2.z_ ;
}

inline Vec3 operator *(const Vec3 &v, float f) {
  return Vec3(v.x_*f, v.y_*f, v.z_*f) ;
}

inline Vec3 operator /(const Vec3 &v, float f) {
  return Vec3(v.x_/f, v.y_/f, v.z_/f) ;
}

inline Vec3 &Vec3::operator *=(float f) {
  x_ *= f ; y_ *= f ; z_ *= f ;
  return *this ;
}

inline Vec3 &Vec3::operator /=(float f) {
  x_ /= f ; y_ /= f ; z_ /= f ;
  return *this ;
}

inline std::ostream &operator << (std::ostream &strm, const Vec3 &m) {
  strm << m.x_ << ' ' << m.y_ << ' ' << m.z_ ; return strm ;
}

inline float& Vec3::operator[] (size_t i) {
    assert((0<=i) && (i<=3));
    switch ( i ) {
    case 0: return x_ ;
    case 1: return y_ ;
    case 2: return z_ ;
    }
}
inline const float& Vec3::operator[] (size_t i) const {
    assert((0<=i) && (i<=3));
    switch ( i ) {
    case 0: return x_ ;
    case 1: return y_ ;
    case 2: return z_ ;
    }

}

inline Vec3 operator -(const Vec3 &v) { return Vec3(-v.x_, -v.y_, -v.z_) ; }

inline float Vec3::length() const { return sqrt(dot(*this)) ; }

inline float Vec3::norm() const { return length() ; }

inline float Vec3::squaredNorm() const { return dot(*this) ; }

inline void Vec3::normalize() { *this /= length() ; }

inline Vec3 Vec3::normalized() const { return *this / length() ; }

inline Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
    return {v1.y_ * v2.z_ - v1.z_ * v2.y_, v1.z_ * v2.x_ - v1.x_ * v2.z_, v1.x_ * v2.y_ - v1.y_ * v2.x_} ;
}

inline Vec3 Vec3::cross(const Vec3 &v) const {
    return {y_ * v.z_ - z_ * v.y_, z_ * v.x_ - x_ * v.z_, x_ * v.y_ - y_ * v.x_} ;
}
