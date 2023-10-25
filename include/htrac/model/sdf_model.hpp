#ifndef HTRAC_SDF_MODEL_HPP
#define HTRAC_SDF_MODEL_HPP

#include <htrac/model/skeleton.hpp>

class SDFModel {

public:

    SDFModel() {}

    virtual uint getNumParts() const = 0;
    virtual uint getPartBone(uint part) const = 0 ;

    // global sdf of a point
    virtual float eval(const Skeleton &sk, const Eigen::Vector3f &v, const Pose &p) const = 0 ;
    // global sdf gradient
    virtual Eigen::Vector3f grad(const Skeleton &sk, const Eigen::Vector3f &v, const Pose &p) const = 0;

    // sdf of part
    virtual float evalPart(uint part, const Eigen::Vector3f &v, const Eigen::Matrix4f &imat) const = 0;
    // sdf gradient of part
    virtual Eigen::Vector3f gradPart(uint part, const Eigen::Vector3f &v, const Eigen::Matrix4f &imat) const = 0 ;

    // return a M x N matrix with distance to each part
    virtual Eigen::MatrixXf eval(const Eigen::MatrixXf &pts) const = 0;

    // return a 3 x N matrix withe columns equals to the gradients of the SDF at given points
    virtual Eigen::MatrixXf grad(const Eigen::MatrixXf &pts, const std::vector<uint> &idxs) const = 0;

    // return a M x 3n matrix where each row corresponds to a bone
    Eigen::MatrixXf transform_points_to_bone_space(const std::vector<Eigen::Vector3f> &pts, const std::vector<Eigen::Matrix4f> &itr) const;
};


#endif
