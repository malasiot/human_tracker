#pragma once

#include <htrac/model/skeleton.hpp>
#include <htrac/pose/keypoint_detector.hpp>

class KeyPointsDistanceField {
public:
    virtual size_t nBones() const = 0 ;
    virtual std::vector<std::string> boneNames() const = 0 ;
    virtual float getDistance(const std::string &bname, const Eigen::Vector2f &p) const = 0;
    virtual Eigen::Vector2f getDistanceGradient(const std::string &bname, const Eigen::Vector2f &p) const =0;
};



class KeyPointListDistanceField: public KeyPointsDistanceField {
public:
    KeyPointListDistanceField(const KeyPoints &kp): kp_(kp) {}

    size_t nBones() const override { return kp_.size() ; }
    std::vector<std::string> boneNames() const override;

    virtual float getDistance(const std::string &bname, const Eigen::Vector2f &p) const override;

    virtual Eigen::Vector2f getDistanceGradient(const std::string &bname, const Eigen::Vector2f &p) const override;


private:
    const KeyPoints &kp_ ;
};
