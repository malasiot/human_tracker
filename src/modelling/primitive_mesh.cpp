#include <htrac/model/primitive_sdf.hpp>
#include <cvx/misc/variant.hpp>
#include <fstream>
#include <iostream>

using namespace std ;
using namespace Eigen ;
extern void makeCircleTable(vector<float> &sint, vector<float> &cost, int n, bool half_circle) ;

void RoundCone::toMesh(size_t slices, size_t stacks, size_t head_stacks,
                       std::vector<Eigen::Vector3f> &vertices,
                       std::vector<uint32_t> &indices) const{
    // make cone

    float y0,y1;

    float delta = r2_ - r1_ ;
    float a = delta/l_ ;
    float s = sqrt(l_ * l_ - delta * delta) ;
    float b = s/l_ ;
    float r2 = r2_ * b ;  // radius of cone
    float r1 = r1_ * b ;
    float h1 = r1_ * a ;  // distance from cone apex to sphere center
    float h2 = r2_ * a ;
    float l = ( delta == 0.0 ) ? s : (r2 - r1)*s/delta; // length of cone

    const float yStep = l / std::max(stacks, (size_t)1) ;

    vector<float> sint, cost;
    makeCircleTable( sint, cost, slices, false);

    y0 =  r1_ - h1 ;
    y1 = y0 + yStep;

    for( unsigned int i=0 ; i<slices ; i++ ) {
        vertices.push_back({cost[i]*r1, y0, sint[i]*r1}) ;
    }


    float r = r1 ;

    for( size_t j = 1;  j <= stacks; j++ ) {

        r =  r1 + (r2 - r1) * j/(float)stacks ;

        for( unsigned int i=0 ; i<slices ; i++ ) {
            vertices.push_back({cost[i]*r, y1, sint[i]*r}) ;
        }

        for( unsigned int i=0 ; i<slices ; i++ ) {
            size_t pn = ( i == slices - 1 ) ? 0 : i+1 ;
            indices.push_back((j-1)*slices + i) ;
            indices.push_back((j)*slices + pn) ;
            indices.push_back((j-1)*slices + pn) ;

            indices.push_back((j-1)*slices + i) ;
            indices.push_back((j)*slices + i) ;
            indices.push_back((j)*slices + pn) ;
        }

        y1 += yStep;
    }

    vector<Vector3f> s1_vertices, s1_normals, s2_vertices, s2_normals ;
    vector<uint32_t> s1_indices, s2_indices ;

    Mesh sphere1 = Mesh::makeSphere(r1_, slices, head_stacks) ;
    Mesh sphere2 = Mesh::makeSphere(r2_, slices, head_stacks) ;

//    create_sphere(r1_, slices, head_stacks, s1_vertices, s1_normals, s1_indices) ;
//    create_sphere(r2_, slices, head_stacks, s2_vertices, s2_normals, s2_indices) ;

    uint32_t v_offset = vertices.size() ;

    for( uint i=0 ; i<sphere2.vertices().size() ; i++ ) {
        vertices.push_back(sphere2.vertices()[i] + Vector3f{0, l_ + r1_, 0}) ;
    }

    for( uint i=0 ; i<sphere2.indices().size() ; i++ ) {
        indices.push_back(sphere2.indices()[i] + v_offset) ;
    }

    v_offset = vertices.size() ;

    for( uint i=0 ; i<sphere1.vertices().size() ; i++ ) {
        const auto &v = sphere1.vertices()[i] ;
        vertices.push_back(Vector3f{v.x(), v.y() + r1, v.z()}) ;
    }

    for( uint i=0 ; i<sphere1.indices().size() ; i++ ) {
        indices.push_back(sphere1.indices()[i] + v_offset) ;
    }

}

float RoundCone::sdf(const Eigen::Vector3f &p) const
{
    float delta = r1_ - r2_ ;
    float b = delta/l_ ;
    float s = sqrt(l_ * l_ - delta * delta) ;
    float a = s/l_ ;

    float px = sqrt(p.x() * p.x() + p.z() * p.z()), py = p.y() - r1_ ;
    Vector2f q{px, py} ;

    float k = - px * b + py * a ;
    if ( k < 0 ) return q.norm() - r1_ ;
    if ( k > s ) return (q - Vector2f{0, l_}).norm() - r2_ ;
    return px * a + py * b - r1_ ;
}

Vector3f RoundCone::grad(const Eigen::Vector3f &p) const {
    float x = p.x(), y = p.y() - r1_, z = p.z() ;

    float delta = r1_ - r2_ ;
    float b = delta/l_ ;
    float s = sqrt(l_ * l_ - delta * delta) ;
    float a = s/l_ ;

    float px = sqrt(x * x + z * z), py = y ;
    float k = - px * b + py * a ;
    if ( k < 0 ) {
        float d = sqrt(x*x + z*z + y*y) ;
        return { x / d, y / d, z / d } ;
    }
    if ( k > s ) {
        float d = sqrt(x*x + z*z + (l_-y)*(l_-y)) ;
        return { x/d,  (y-l_)/d, z/d } ;
    }
    return { a * x / px, b, a * z / px } ;
}

Mesh RoundCone::mesh() const {
    Mesh m ;
    toMesh(11, 10, 10, m.vertices(), m.indices()) ;
    return m ;
}



Model3d PrimitiveSDF::makeMesh(const Skeleton &skeleton, const Pose &p)
{
    Model3d model ;
    for( const auto &bp: primitives_ ) {
        const Primitive *pr = bp.second.get() ;
        auto tr = skeleton.computeBoneTransform(p, bp.first) ;

        Mesh m = pr->mesh() ;
        Model3d child ;
        child.setGeometry(m);
        child.setPose(tr) ;
        model.addChild(std::move(child)) ;
    }

    return model ;
}

float PrimitiveSDF::eval(const Skeleton &skeleton, const Eigen::Vector3f &p, const Pose &pose) const {
    float min_dist = std::numeric_limits<float>::max(), best_dist ;
    for( const auto &bp: primitives_ ) {
        const Primitive *pr = bp.second.get() ;
        auto tr = skeleton.computeBoneTransform(pose, bp.first) ;
        auto itr = tr.inverse() ;
        Vector3f pt = (itr * p.homogeneous()).head<3>() ;

        float dist = pr->sdf(pt) ;
        if ( fabs(dist) < min_dist ) {
            min_dist = fabs(dist) ;
            best_dist = dist ;
        }
    }

    return best_dist ;
}

Vector3f PrimitiveSDF::grad(const Skeleton &skeleton, const Vector3f &p, const Pose &pose) const {
    float min_dist = std::numeric_limits<float>::max() ;
    Vector3f best_g ;
    for( const auto &bp: primitives_ ) {
        const Primitive *pr = bp.second.get() ;
        auto tr = skeleton.computeBoneTransform(pose, bp.first) ;
        auto itr = tr.inverse() ;
        Vector3f pt = (itr * p.homogeneous()).head<3>() ;

        float dist = pr->sdf(pt) ;

        if ( fabs(dist) < min_dist ) {
            min_dist = fabs(dist) ;
            best_g = tr.block<3, 3>(0, 0) * pr->grad(pt) ;
        }
    }

    return best_g ;
}

float PrimitiveSDF::evalPart(uint part, const Eigen::Vector3f &p, const Eigen::Matrix4f &imat) const {
    const auto &bp = primitives_[part] ;
    const Primitive *pr = bp.second.get() ;
    Vector3f pt = (imat * p.homogeneous()).head<3>() ;
    return pr->sdf(pt) ;
}

Vector3f PrimitiveSDF::gradPart(uint part, const Eigen::Vector3f &p, const Eigen::Matrix4f &imat) const {
    const auto &bp = primitives_[part] ;
    const Primitive *pr = bp.second.get() ;
    Vector3f pt = (imat * p.homogeneous()).head<3>() ;
    return pr->grad(pt) ;
}

MatrixXf PrimitiveSDF::eval(const Eigen::MatrixXf &pts) const {
    uint M = pts.rows(), N = pts.cols();

    MatrixXf distances(M, N/3) ;

#pragma omp parallel for
    for(uint i=0 ; i<M ; i++) {
        for(uint j=0, k=0 ; j<N ; j+=3, k++ ) {
            Vector3f pb = pts.block<1, 3>(i, j) ;
            distances(i, k) = primitives_[i].second->sdf(pb) ;
        }
    }

    return distances ;
}

MatrixXf PrimitiveSDF::grad(const Eigen::MatrixXf &pts, const std::vector<uint> &idxs) const
{
    uint N = idxs.size() ;

    MatrixXf grad(3, N) ;

#pragma omp parallel for
    for(uint i=0 ; i<N ; i++) {
        uint idx = idxs[i] ;

        if ( idx > primitives_.size() )
            { grad.col(i) = Vector3f(0, 0, 0) ; continue ; }

        Vector3f pb = pts.block<1, 3>(idx, 3*i) ;

        grad.col(i) = primitives_[idx].second->grad(pb) ;
    }
    return grad ;
}

float Box::sdf(const Eigen::Vector3f &p) const
{
    Vector3f q{ fabs(p.x()) - hs_.x(), fabs(p.y()) - hs_.y(), fabs(p.z()) - hs_.z()} ;
    float r = max(q.x(), max(q.y(), q.z())) ;
    Vector3f mq{ max(q.x(), 0.0f), max(q.y(), 0.0f), max(q.z(), 0.0f)};
    return mq.norm() + min(r, 0.0f) - r_;
}

static inline float sign(float x) {
    if ( x < 0 ) return -1 ;
    else return 1 ;
}

Vector3f Box::grad(const Eigen::Vector3f &p) const {
    float x = p.x(), y = p.y(), z = p.z() ;
    Vector3f q{ fabs(x) - hs_.x(), fabs(y) - hs_.y(), fabs(z) - hs_.z()} ;
    Vector3f mq{ max(q.x(), 0.0f), max(q.y(), 0.0f), max(q.z(), 0.0f)};
    float r = max(q.x(), max(q.y(), q.z())) ;

    if ( r > 0 ) { //outside
        float dmx = ( q.x() > 0 ) ? sign(x) : 0 ;
        float dmy = ( q.y() > 0 ) ? sign(y) : 0 ;
        float dmz = ( q.z() > 0 ) ? sign(z) : 0 ;
        return { mq.x() * dmx / mq.norm(), mq.y() * dmy / mq.norm(), mq.z() * dmz / mq.norm()} ;
    }

    if ( q.x() >= q.y() && q.x() > q.z() ) {
        return { sign(x), 0, 0 } ;
    } else if ( q.y() > q.x() && q.y() >= q.z() ) {
        return { 0, sign(y), 0 } ;
    } else {
        return { 0,  0, sign(z) } ;
    }
}

Mesh Box::mesh() const
{
    return Mesh::makeCube(hs_ + Vector3f{r_, r_, r_}) ;

}

inline float clamp(float x, float low, float high) {
    if ( x < low ) return low ;
    else if ( x > high ) return high ;
    else return x ;
}

float sdf_elongate(Primitive *d, const Vector3f &p, float h) {
    Vector3f q ;
    q.x() = p.x() - clamp(p.x(), -h, h) ;
    q.y() = p.y() - clamp(p.y(), -h, h) ;
    q.z() = p.z() - clamp(p.z(), -h, h) ;

    return d->sdf(q) ;
}

float ElongatedPrimitive::sdf(const Eigen::Vector3f &p) const {
    Vector3f q ;
    q.x() = p.x() - clamp(p.x(), -eh_.x(), eh_.x()) ;
    q.y() = p.y() - clamp(p.y(), -eh_.y(), eh_.y()) ;
    q.z() = p.z() - clamp(p.z(), -eh_.z(), eh_.z()) ;

    return base_->sdf(q) ;
}

Vector3f ElongatedPrimitive::grad(const Eigen::Vector3f &p) const
{
    return {0, 0, 0};
}

Vector3f ScaledPrimitive::grad(const Eigen::Vector3f &p) const {
    Vector3f q ;
    q.x() = p.x() / scale_.x() ;
    q.y() = p.y() / scale_.y() ;
    q.z() = p.z() / scale_.z() ;
    q = base_->grad(q) ;

    float factor = min(scale_.x(), min(scale_.y(), scale_.z())) ;
    return Vector3f{ factor * q.x() / scale_.x(), factor * q.y() * scale_.y(), factor * q.z() * scale_.z()};
}

Mesh ScaledPrimitive::mesh() const {
    Mesh m = base_->mesh() ;
    for( auto &v: m.vertices() ) {
        v.x() *= scale_.x() ;
        v.y() *= scale_.y() ;
        v.z() *= scale_.z() ;
    }

    return m ;
}

float ScaledPrimitive::sdf(const Eigen::Vector3f &p) const
{
    Vector3f q ;
    q.x() = p.x() / scale_.x() ;
    q.y() = p.y() / scale_.y() ;
    q.z() = p.z() / scale_.z() ;

    float factor = min(scale_.x(), min(scale_.y(), scale_.z())) ;
    return base_->sdf(q) * factor ;
}


Vector3f Sphere::grad(const Eigen::Vector3f &p) const {
    return p/p.norm() ;
}

Mesh Sphere::mesh() const {
    return Mesh::makeSphere(r_, 11, 10) ;
}


Vector3f TransformedPrimitive::grad(const Eigen::Vector3f &p) const
{
   Vector3f bg = base_->grad(itr_ * p) ;
    return tr_.linear() * bg ;
}

Mesh TransformedPrimitive::mesh() const {
    Mesh m = base_->mesh() ;
    for( auto &v: m.vertices() ) {
        v = tr_ * v ;
    }
    return m ;
}

Primitive *parsePrimitive(const cvx::Variant &json) {
    string shape = json["shape"].toString();
    if ( shape == "round-cone") {
        float r1 = json["r1"].toFloat() ;
        float r2 = json["r2"].toFloat() ;
        float l = json["l"].toFloat() ;
        return new RoundCone(l, r1, r2) ;
    } else if ( shape == "box" ) {
        auto sz = json["hs"] ;
        float r = json["r"].toFloat() ;
        return new Box({sz[0].toFloat(), sz[1].toFloat(), sz[2].toFloat()}, r);
    } else if ( shape == "sphere" ) {
        float r = json["r"].toFloat() ;
        return new Sphere(r) ;
    } else if ( shape == "scaled" ) {
        auto sc = json["scale"] ;
        Vector3f scale(sc[0].toFloat(), sc[1].toFloat(), sc[2].toFloat()) ;
        Primitive *child = parsePrimitive(json["child"]) ;
        return new ScaledPrimitive(child, scale) ;
    } else if ( shape == "transformed" ) {
        auto r = json["rot"], t = json["trans"];
        Vector3f rpy, trans ;
        rpy << r[0].toFloat(), r[1].toFloat(), r[2].toFloat() ;
        trans << t[0].toFloat(), t[1].toFloat(), t[2].toFloat() ;
        Primitive *child = parsePrimitive(json["child"]) ;

        auto m = AngleAxisf(rpy.x(), Vector3f::UnitX())
          * AngleAxisf(rpy.y(), Vector3f::UnitY())
          * AngleAxisf(rpy.z(), Vector3f::UnitZ());
        Isometry3f btr ;
        btr.setIdentity() ;
        btr.translate(trans) ;
        btr.rotate(m) ;
        return new TransformedPrimitive(child, btr) ;
    }
}

void PrimitiveSDF::readJSON(const Skeleton &sk, const std::string &path) {
    ifstream strm(path) ;
    cvx::Variant json = cvx::Variant::fromJSON(strm) ;
    auto it = json.begin() ;
    for( ; it != json.end() ; ++it ) {
        string bone = it.key() ;
        addPrimitive(sk.getBoneIndex(bone), parsePrimitive(it.value())) ;
    }
}
