#include "primitive_sdf_gpu.hpp"
#include "cuda_util.hpp"

using namespace Eigen ;
using namespace std ;

__host__ void PrimitiveGPU::set(const Primitive *p) {
    if ( const RoundCone *rc = dynamic_cast<const RoundCone *>(p) ) {
        params_[0] = rc->l_ ;
        params_[1] = rc->r1_ ;
        params_[2] = rc->r2_ ;
        type_ = CappedCylinder ;
    } else if ( const Box *rc = dynamic_cast<const Box *>(p) ) {
        params_[0] = rc->hs_.x() ;
        params_[1] = rc->hs_.y() ;
        params_[2] = rc->hs_.z() ;
        params_[3] = rc->r_ ;
        type_ = BoxPrimitive ;
    } else if ( const Sphere *rc = dynamic_cast<const Sphere *>(p) ) {
        params_[0] = rc->r_ ;
        type_ = SpherePrimitive ;
    }else if ( const ScaledPrimitive *sc = dynamic_cast<const ScaledPrimitive *>(p) ) {
        scale_ = sc->scale_ ;
        set(sc->base_.get()) ;
        scaled_ = true ;
    } else if ( const TransformedPrimitive *tp = dynamic_cast<const TransformedPrimitive *>(p) ) {
        tr_ = tp->tr_.matrix() ;
        itr_ = tr_.inverse() ;
        set(tp->base_.get()) ;
        transformed_ = true ;
    }
}
// Eigen matrix vector product not working

#if 0
__device__ static Vector3f product(const Matrix4f &m, const Vector3f &v) {
    Vector3f r ;

    for( int i=0 ; i<3 ; i++ ) {
        float row = 0.0 ;
        for( int j=0 ; j<3 ; j++ ) {
            row += m(i, j) * v[j] ;
        }
        row += m(i, 3) ;
        r[i] = row ;
    }

    return r ;
}

__device__ static Vector3f product3(const Matrix4f &m, const Vector3f &v) {
    Vector3f r ;

    for( int i=0 ; i<3 ; i++ ) {
        float row = 0.0 ;
        for( int j=0 ; j<3 ; j++ ) {
            row += m(i, j) * v[j] ;
        }
        r[i] = row ;
    }

    return r ;
}
#endif
float PrimitiveGPU::eval(const Vec3 &p) const {

    if ( scaled_ ) {
        Vec3 q ;
        q.x() = p.x() / scale_.x() ;
        q.y() = p.y() / scale_.y() ;
        q.z() = p.z() / scale_.z() ;

        float factor = min(scale_.x(), min(scale_.y(), scale_.z())) ;
        PrimitiveGPU sc(*this) ;
        sc.scaled_ = false ;

        return sc.eval(q) * factor ;
    } else if ( transformed_ ) {
        PrimitiveGPU sc(*this) ;
        sc.transformed_ = false ;
        return sc.eval(itr_ * p) ;
    }
    else if ( type_ == SpherePrimitive ) {
        return p.norm() - params_[0] ;
    } else if ( type_ == BoxPrimitive ) {
        Vector3f hs(params_[0], params_[1], params_[2]);
        Vector3f q{ fabs(p.x()) - hs.x(), fabs(p.y()) - hs.y(), fabs(p.z()) - hs.z()} ;
        float r = max(q.x(), max(q.y(), q.z())) ;
        Vector3f mq{ max(q.x(), 0.0f), max(q.y(), 0.0f), max(q.z(), 0.0f)};
        return mq.norm() + min(r, 0.0f) - params_[3];
    } else if ( type_ == CappedCylinder ) {
        float l = params_[0] ;
        float r1 = params_[1] ;
        float r2 = params_[2] ;

        float delta = r1 - r2 ;
        float b = delta/l ;
        float s = sqrt(l * l - delta * delta) ;
        float a = s/l ;

        float px = sqrt(p.x() * p.x() + p.z() * p.z()), py = p.y() - r1 ;
        Vector2f q{px, py} ;

        float k = - px * b + py * a ;
        if ( k < 0 ) return q.norm() - r1 ;
        if ( k > s ) return (q - Vector2f{0, l}).norm() - r2 ;
        return px * a + py * b - r1 ;
    }
}

__device__ static inline float sign(float x) {
    if ( x < 0 ) return -1 ;
    else return 1 ;
}

Vec3 PrimitiveGPU::grad(const Vec3 &p) const
{
    if ( scaled_ ) {

        Vec3 q ;
        q.x() = p.x() / scale_.x() ;
        q.y() = p.y() / scale_.y() ;
        q.z() = p.z() / scale_.z() ;

        float factor = min(scale_.x(), min(scale_.y(), scale_.z())) ;

        PrimitiveGPU sc(*this) ;
        sc.scaled_ = false ;

        q = sc.grad(q) ;
        return Vec3{ factor * q.x() / scale_.x(), factor * q.y() * scale_.y(), factor * q.z() * scale_.z()};

    } else if ( transformed_ ) {
        PrimitiveGPU sc(*this) ;
        sc.transformed_ = false ;
        Vec3 bg = sc.grad(itr_ * p) ;
        return tr_ * bg ;
    }
    else if ( type_ == SpherePrimitive ) {
        return p/p.norm() ;
    } else if ( type_ == BoxPrimitive ) {
        Vec3 hs(params_[0], params_[1], params_[2]);
        float rad = params_[3] ;

        float x = p.x(), y = p.y(), z = p.z() ;
        Vec3 q{ fabs(x) - hs.x(), fabs(y) - hs.y(), fabs(z) - hs.z()} ;
        Vec3 mq{ max(q.x(), 0.0f), max(q.y(), 0.0f), max(q.z(), 0.0f)};
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
    } else if ( type_ == CappedCylinder ) {
        float l = params_[0] ;
        float r1 = params_[1] ;
        float r2 = params_[2] ;

        float x = p.x(), y = p.y() - r1, z = p.z() ;

        float delta = r1 - r2 ;
        float b = delta/l ;
        float s = sqrt(l * l- delta * delta) ;
        float a = s/l ;

        float px = sqrt(x * x + z * z), py = y ;
        float k = - px * b + py * a ;
        if ( k < 0 ) {
            float d = sqrt(x*x + z*z + y*y) ;
            return { x / d, y / d, z / d } ;
        }
        if ( k > s ) {
            float d = sqrt(x*x + z*z + (l-y)*(l-y)) ;
            return { x/d,  (y-l)/d, z/d } ;
        }
        return { a * x / px, b, a * z / px } ;

    }
}

#define LOG_MAT4(x) printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n", x(0, 0), x(0, 1), x(0, 2), x(0, 3), x(1, 0), x(1, 1), x(1, 2), x(1, 3), x(2, 0), x(2, 1), x(2, 2), x(2, 3), x(3, 0), x(3, 1), x(3, 2), x(3, 3));
#define LOG_VEC(a) printf("%f %f %f\n", a.x(), a.y(), a.z())


__global__ void transformPointsKernel(size_t npts, size_t npr, const PrimitiveGPU *bfp,
                                      const Vec3 *ipts, const Matrix4x4 *imat, Vec3 *tpos)
{
    uint part = blockIdx.y ;
    uint tc = threadIdx.x ;
    uint idx = blockIdx.x * blockDim.x + threadIdx.x ;
    uint bone = bfp[part].bone_ ;

    __shared__ Matrix4x4 itrans ;

    if ( tc == 0 )  // for each block we have a shared bone matrix (correpsonding to the part). we need to load this only once
        itrans = imat[bone] ;

    __syncthreads() ;

    if ( idx < npts ) {
        uint oidx = npr * idx + part ;

        const Vec3 &v = ipts[idx] ;

        Vec3 pb = itrans * v ;

        // store the result into the output buffer (data stored collumnwise)
        tpos[oidx] = pb ;
    }
}

void PrimitiveSDFGPU::transform_points_to_bone_space(const thrust::device_vector<Vec3> &ipts,
                                                        thrust::device_vector<Vec3> &tpts,
                                                        const thrust::device_vector<Matrix4x4> &itran) const
{
    size_t blockSize = CUDA_BLOCK_SIZE ;
    size_t n = ipts.size() ;
    size_t np = primitives_.size() ;

    uint gridSizeX = (n + blockSize - 1) / blockSize;
    uint gridSizeY = np ;

    // each block will correspond to a different part and a tile of the input points

    tpts.resize(np * n) ;

    transformPointsKernel<<<dim3(gridSizeX, gridSizeY), blockSize>>>(n, np,
            TPTR(primitives_), TPTR(ipts), TPTR(itran), TPTR(tpts)) ;

    CudaSafeCall(cudaDeviceSynchronize()) ;
}

__global__ void samplePointsKernel(uint npts, uint npr, const PrimitiveGPU *pr, const Vec3 *pts, float *samples) {
    uint part = blockIdx.y ;
    uint idx = blockIdx.x * blockDim.x + threadIdx.x ;

    if ( idx < npts ) {
        uint oidx = npr * idx + part ;
        samples[oidx] = pr[part].eval(pts[oidx]) ;
    }
}


void PrimitiveSDFGPU::sample(uint n, const thrust::device_vector<Vec3> &tpts, thrust::device_vector<float> &samples) const {
    const uint blockSize = CUDA_BLOCK_SIZE ;
    const size_t np = primitives_.size() ;

    uint gridSizeX = (n + blockSize - 1) / blockSize;
    uint gridSizeY = np ;

    samples.resize(tpts.size()) ;

    samplePointsKernel<<<dim3(gridSizeX, gridSizeY), blockSize>>>(n, np, TPTR(primitives_), TPTR(tpts), TPTR(samples)) ;

    CudaSafeCall(cudaDeviceSynchronize()) ;
}

__global__ void sampleGradientKernel(uint npts, uint npr, const PrimitiveGPU *pr, const Vec3 *pts, const size_t *labels, Vec3 *grad) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if ( idx < npts ) {
        size_t part = labels[idx] ;
        Vec3 g(0, 0, 0) ;
        if ( part < npr ) {
            Vec3 pt = pts[npr * idx + part] ;
            g = pr[part].grad(pt) ;
        }
        grad[idx] = g ;
    }
}

void PrimitiveSDFGPU::grad(uint n, const thrust::device_vector<Vec3> &tpts, const thrust::device_vector<size_t> &labels,
                           thrust::device_vector<Vec3> &g) const
{
    const uint blockSize = CUDA_BLOCK_SIZE ;
    const size_t np = primitives_.size() ;

    uint gridSize = (n + blockSize - 1) / blockSize;

    sampleGradientKernel<<<gridSize, blockSize>>>(n, np, TPTR(primitives_), TPTR(tpts), TPTR(labels), TPTR(g)) ;

    CudaSafeCall(cudaDeviceSynchronize()) ;
}


PrimitiveSDFGPU::PrimitiveSDFGPU(const PrimitiveSDF &sdf) {

    std::vector<PrimitiveGPU, Eigen::aligned_allocator<PrimitiveGPU>> primitives ;

    for( uint i=0 ; i<sdf.getNumParts() ; i++ ) {
        const Primitive *p = sdf.getPrimitive(i) ;
        uint bone = sdf.getPartBone(i) ;

        PrimitiveGPU pr ;
        pr.bone_ = bone ;

        pr.set(p) ;
        primitives.emplace_back(std::move(pr)) ;

    }

    primitives_ = primitives ;
}

