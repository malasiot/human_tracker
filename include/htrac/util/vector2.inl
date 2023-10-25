
inline Vec2::Vec2(float v): x_(v), y_(v) {
}

inline Vec2::Vec2(float x, float y): x_(x), y_(y) {
}

inline Vec2::Vec2(const Vec2 &o): x_(o.x_), y_(o.y_) {
}

inline Vec2::Vec2(const Eigen::Vector2f &o): x_(o.x()), y_(o.y()) {
}

inline Vec2 &Vec2::operator = (const Vec2 &o) {
    x_ = o.x_ ; y_ = o.y_ ;
    return *this ;
}

inline Vec2 &Vec2::operator = (const Eigen::Vector2f &o) {
    x_ = o.x() ; y_ = o.y() ;
    return *this ;
}

inline float Vec2::x() const {
    return x_ ;
}

inline float &Vec2::x() {
    return x_ ;
}

inline float Vec2::y() const {
    return y_ ;
}

inline float &Vec2::y() {
    return y_ ;
}

inline Vec2 operator+(const Vec2 &v1, const Vec2 &v2) {
  return Vec2(v1.x_ + v2.x_, v1.y_ + v2.y_) ;
}

inline Vec2 operator-(const Vec2 &v1, const Vec2 &v2) {
  return Vec2(v1.x_ - v2.x_, v1.y_ - v2.y_) ;
}

inline const Vec2 &Vec2::operator +=(const Vec2 &v) {
  x_ += v.x_ ; y_ += v.y_ ;
  return *this ;
}

inline const Vec2 &Vec2::operator -=(const Vec2 &v) {
  x_ -= v.x_ ; y_ -= v.y_ ;
  return *this ;
}

inline float dot(const Vec2 &v1, const Vec2 &v2) {
   return v1.x_ * v2.x_ + v1.y_ * v2.y_  ;
}

inline float Vec2::dot(const Vec2 &v2) const {
   return x_ * v2.x_ + y_ * v2.y_  ;
}

inline Vec2 operator *(const Vec2 &v, float f) {
  return Vec2(v.x_*f, v.y_*f) ;
}

inline Vec2 operator /(const Vec2 &v, float f) {
  return Vec2(v.x_/f, v.y_/f) ;
}

inline Vec2 &Vec2::operator *=(float f) {
  x_ *= f ; y_ *= f ;
  return *this ;
}

inline Vec2 &Vec2::operator /=(float f) {
  x_ /= f ; y_ /= f ;
  return *this ;
}

inline std::ostream &operator << (std::ostream &strm, const Vec2 &m) {
  strm << m.x_ << ' ' << m.y_  ; return strm ;
}

inline float& Vec2::operator[] (size_t i) {
    assert((0<=i) && (i<=2));
    switch ( i ) {
    case 0: return x_ ;
    case 1: return y_ ;
    }
}
inline const float& Vec2::operator[] (size_t i) const {
    assert((0<=i) && (i<=2));
    switch ( i ) {
    case 0: return x_ ;
    case 1: return y_ ;
    }

}

inline Vec2 operator -(const Vec2 &v) { return Vec2(-v.x_, -v.y_) ; }

inline float Vec2::length() const { return sqrt(dot(*this)) ; }

inline float Vec2::norm() const { return length() ; }

inline float Vec2::squaredNorm() const { return dot(*this) ; }

inline void Vec2::normalize() { *this /= length() ; }

inline Vec2 Vec2::normalized() const { return *this / length() ; }

