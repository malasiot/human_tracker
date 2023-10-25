#pragma once

#include <vector>
#include <string>

#include <Eigen/Core>

#include <htrac/model/skeleton.hpp>

struct CollisionData {
    struct Sphere {
        Eigen::Vector3f c_ ;
        float r_ ;
        std::string group_ ;
        uint bone_ ;
        std::string name_  ;
    };

    std::vector<Sphere> spheres_ ;

    void parseJson(const Skeleton &sk, const std::string &fpath) ;
};

