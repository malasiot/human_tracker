#pragma once

#include <thrust/device_vector.h>

#include <htrac/model/primitive_sdf.hpp>

#include <htrac/util/matrix4x4.hpp>
#include <htrac/util/vector.hpp>

enum PrimitiveType { SpherePrimitive, BoxPrimitive, CappedCylinder } ;

struct alignas(16) PrimitiveGPU {
    PrimitiveGPU() = default ;

    Matrix4x4 tr_, itr_ ;
    Vec3 scale_ ;
    float params_[4] ;

    bool scaled_ = false, transformed_ = false ;
    PrimitiveType type_ ;
    uint bone_ ;

    __host__ void set(const Primitive *p) ;

    __device__ float eval(const Vec3 &p) const ;
    __device__ Vec3 grad(const Vec3 &p) const ;

};


struct PrimitiveSDFGPU {

    PrimitiveSDFGPU(const PrimitiveSDF &psdf) ;

    thrust::device_vector<PrimitiveGPU> primitives_ ;

    void transform_points_to_bone_space(const thrust::device_vector<Vec3> &ipts,
                                              thrust::device_vector<Vec3> &tpts,
                                              const thrust::device_vector<Matrix4x4> &itran) const ;

    void sample(uint n, const thrust::device_vector<Vec3> &tpts, thrust::device_vector<float> &samples) const ;
    void grad(uint n, const thrust::device_vector<Vec3> &tpts, const thrust::device_vector<size_t> &labels,
              thrust::device_vector<Vec3> &g) const ;
};
