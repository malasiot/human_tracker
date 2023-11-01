
inline Vec4::Vec4(float v): x_(v), y_(v), z_(v), w_(v) {
}

inline Vec4::Vec4(float x, float y, float z, float w): x_(x), y_(y), z_(z), w_(w) {
}

inline Vec4::Vec4(const Vec4 &o): x_(o.x_), y_(o.y_), z_(o.z_), w_(o.w_) {
}

inline Vec4::Vec4(const Vec3 &o, float w): x_(o.x_), y_(o.y_), z_(o.z_), w_(w) {
}


inline Vec4 &Vec4::operator = (const Vec4 &o) {
    x_ = o.x_ ; y_ = o.y_ ; z_ = o.z_ ; w_ = o.w_ ;
    return *this ;
}

inline float Vec4::x() const {
    return x_ ;
}

inline float &Vec4::x() {
    return x_ ;
}

inline float Vec4::y() const {
    return y_ ;
}

inline float &Vec4::y() {
    return y_ ;
}

inline float Vec4::z() const {
    return z_ ;
}

inline float &Vec4::z() {
    return z_ ;
}

inline float Vec4::w() const {
    return w_ ;
}

inline float &Vec4::w() {
    return w_ ;
}

inline Vec4 operator+(const Vec4 &v1, const Vec4 &v2) {
  return Vec4(v1.x_ + v2.x_, v1.y_ + v2.y_, v1.z_ + v2.z_, v1.w_ + v2.w_) ;
}

inline Vec4 operator-(const Vec4 &v1, const Vec4 &v2) {
  return Vec4(v1.x_ - v2.x_, v1.y_ - v2.y_, v1.z_ - v2.z_, v1.w_ - v2.w_) ;
}

inline const Vec4 &Vec4::operator +=(const Vec4 &v) {
  x_ += v.x_ ; y_ += v.y_ ; z_ += v.z_ ; w_ += v.w_ ;
  return *this ;
}

inline const Vec4 &Vec4::operator -=(const Vec4 &v) {
  x_ -= v.x_ ; y_ -= v.y_ ; z_ -= v.z_ ; w_ -= v.w_ ;
  return *this ;
}

inline float dot(const Vec4 &v1, const Vec4 &v2) {
   return v1.x_ * v2.x_ + v1.y_ * v2.y_ + v1.z_ * v2.z_ + v1.w_ * v2.w_ ;
}

inline float Vec4::dot(const Vec4 &v2) const {
   return x_ * v2.x_ + y_ * v2.y_ + z_ * v2.z_ + w_ * v2.w_ ;
}

inline Vec4 operator *(const Vec4 &v, float f) {
  return Vec4(v.x_*f, v.y_*f, v.z_*f, v.w_*f) ;
}

inline Vec4 operator /(const Vec4 &v, float f) {
  return Vec4(v.x_/f, v.y_/f, v.z_/f, v.w_/f) ;
}

inline Vec4 &Vec4::operator *=(float f) {
  x_ *= f ; y_ *= f ; z_ *= f ; w_ *= f ;
  return *this ;
}

inline Vec4 &Vec4::operator /=(float f) {
  x_ /= f ; y_ /= f ; z_ /= f ; w_ /= f ;
  return *this ;
}

inline std::ostream &operator << (std::ostream &strm, const Vec4 &m) {
  strm << m.x_ << ' ' << m.y_ << ' ' << m.z_ << ' ' << m.w_ ; return strm ;
}

inline float& Vec4::operator[] (size_t i) {
    assert((0<=i) && (i<=4));
    switch ( i ) {
    case 0: return x_ ;
    case 1: return y_ ;
    case 2: return z_ ;
    case 3: return w_ ;
    }
}
inline const float& Vec4::operator[] (size_t i) const {
    assert((0<=i) && (i<=4));
    switch ( i ) {
    case 0: return x_ ;
    case 1: return y_ ;
    case 2: return z_ ;
    case 3: return w_ ;
    }

}

inline Vec3 Vec4::head() const { return { x_, y_, z_ } ; }

inline Vec4 operator -(const Vec4 &v) { return Vec4(-v.x_, -v.y_, -v.z_, -v.w_) ; }

inline float Vec4::length() const { return sqrt(dot(*this)) ; }

inline float Vec4::norm() const { return length() ; }

inline float Vec4::squaredNorm() const { return dot(*this) ; }

inline void Vec4::normalize() { *this /= length() ; }

inline Vec4 Vec4::normalized() const { return *this / length() ; }
