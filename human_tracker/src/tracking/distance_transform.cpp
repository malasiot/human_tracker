#include "distance_transform.hpp"

#include <memory>
#include <cmath>

using namespace std ;


static const float INF = 1.0e20f ;

static void distanceTransform1D(float* in, float* out, unsigned int width, float* z, unsigned int * v, unsigned int stride, float take_sqrt) {

    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for ( unsigned int q = 1 ; q <= width-1; q++ ) {
        unsigned int sq = q * stride ;
        float s  = ((in[sq]+q*q)-(in[v[k] * stride]+v[k]*v[k]))/(2*q-2*v[k]);
        while (s <= z[k]) {
            k--;
            s  = ((in[sq]+q*q)-(in[v[k] * stride]+v[k]*v[k]))/(2*q-2*v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INF;
    }

    k = 0;
    for (unsigned int q = 0; q <= width-1; q++) {
        while (z[k+1] < q) k++;
        float dist = (q-v[k])*(q-v[k]) + in[v[k]*stride];
        out[q * stride ] = take_sqrt ? sqrt(dist) : dist ;
    }

}

void distanceTransform1D(float *src, float *dst, unsigned int w, bool take_sqrt) {
    unique_ptr<float []> z(new float [w + 1]) ;
    unique_ptr<unsigned int []> v(new unsigned int [w]) ;
    distanceTransform1D(src, dst, w, z.get(), v.get(), 1, take_sqrt) ;
}

void distanceTransform2D(float *src, float *dst, unsigned int w, unsigned int h, bool take_sqrt) {
    unique_ptr<float []> z(new float [(w + 1)*(h + 1)]) ;
    unique_ptr<unsigned int []> v(new unsigned int [w * h]) ;
    unique_ptr<float []> s(new float [w * h]) ;

    // row wise
#pragma omp parallel for
    for( uint i=0 ; i<h ; i++ )
        distanceTransform1D(src + i*w,
                            s.get() + i*w,
                            w,
                            z.get() + i*(w+1),
                            v.get() + i*w,
                            1,  // stride
                            false) ;

    // column wise
#pragma omp parallel for
    for( uint j=0 ; j<w ; j++ )
        distanceTransform1D(s.get() + j,
                            dst + j,
                            h,
                            z.get() + j*(h+1),
                            v.get() + j*h,
                            w,
                            take_sqrt) ;

}

void distanceTransform3D(float *src, float *dst, unsigned int w, unsigned int h, unsigned int d, bool take_sqrt) {
    unique_ptr<float []> z(new float [(w +1) *(h+1) * (d + 1)]) ; // we need to allocate this much space to work in parallel
    unique_ptr<unsigned int []> v(new unsigned int [w*h*d]) ;
    unique_ptr<float []> s(new float [w * h * d]) ;

    // z slices
#pragma omp parallel for
    for( uint k=0 ; k<d ; k++ )
        distanceTransform2D(src + k*w*h,
                            s.get() + k*w*h,
                            w,
                            h,  // stride
                            false) ;
#pragma omp parallel for
    for ( uint i=0 ; i<h ; i++ )
        for( uint j=0 ; j<w ; j++ )
            distanceTransform1D(s.get() + i*w + j,
                                dst + i*w + j,
                                d,
                                z.get() + i*(w+1) + j,
                                v.get() + i*w + j,
                                w * h,
                                take_sqrt) ;

}

void signedDistanceTransform3D(float *src, float *dst, unsigned int w, unsigned int h, unsigned depth, bool take_sqrt)
{
    for (uint z=0; z<depth; z++) {
        for (uint y=0; y<h; y++) {
            for (uint x=0; x<w; x++) {

                const int index = x + w*(y + h*z);

                if ( src[index] == 0 ) { // inside
                    dst[index] = INF;
                    if (x > 0) {
                        if (src[index-1] != 0.0) {
                            dst[index] = 0.0;
                            continue;
                        }
                    }
                    if (x < w - 1 ) {
                        if (src[index+1] != 0.0) {
                            dst[index] = 0.0;
                            continue;
                        }
                    }
                    if (y > 0) {
                        if (src[index-w] != 0.0) {
                            dst[index] = 0.0;
                            continue;
                        }
                    }
                    if (y < h - 1) {
                        if (src[index+w] != 0.0) {
                            dst[index] = 0.0;
                            continue;
                        }
                    }
                    if (z > 0) {
                        if (src[index-w*h] != 0.0) {
                            dst[index] = 0.0;
                            continue;
                        }
                    }
                    if (z < depth - 1) {
                        if (src[index+w*h] != 0.0) {
                            dst[index] = 0.0;
                            continue;
                        }
                    }
                }
                else {
                    dst[index] = 0;
                }

            }
        }
    }

    std::unique_ptr<float []> tmp(new float [w * h * depth]) ;

    distanceTransform3D(dst, tmp.get(), w, h, depth, take_sqrt);
    distanceTransform3D(src, dst, w, h, depth, take_sqrt);

    float *p = src, *q = tmp.get(), *r = dst ;
    for (uint i=0; i<w*h*depth; i++, ++p, ++q, ++r ) {
        if ( *p == 0 )
            *r = -(*q) ;
    }
}
