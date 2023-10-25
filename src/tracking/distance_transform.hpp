#ifndef DISTANCETRANSFORM_HPP
#define DISTANCETRANSFORM_HPP

#include <iostream>

void distanceTransform3D(float *src, float *dst, unsigned int w, unsigned int h, unsigned depth, bool take_sqrt) ;

void distanceTransform2D(float *src, float *dst, unsigned int w, unsigned int h, bool take_sqrt) ;

void signedDistanceTransform3D(float *src, float *dst, unsigned int w, unsigned int h, unsigned depth, bool take_sqrt) ;

#endif

