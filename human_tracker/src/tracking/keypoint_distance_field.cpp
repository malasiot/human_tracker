#include <htrac/pose/keypoint_distance_field.hpp>

using namespace std ;

std::vector<std::string> KeyPointListDistanceField::boneNames() const {
    std::vector<std::string> bnames ;
    for( const auto &bp: kp_ )
        bnames.push_back(bp.first) ;
    return bnames ;
}

float KeyPointListDistanceField::getDistance(const std::string &bname, const Eigen::Vector2f &p) const {
    auto it = kp_.find(bname) ;
    assert( it != kp_.end()) ;
    const auto &pt = it->second.first ;
    float w = it->second.second ;
    return w * ( pt - p ).norm() ;
}

Eigen::Vector2f KeyPointListDistanceField::getDistanceGradient(const std::string &bname, const Eigen::Vector2f &p) const {
    auto it = kp_.find(bname) ;
    assert( it != kp_.end()) ;
    const auto &pt = it->second.first ;
    float w = it->second.second ;
    return w * (pt - p)/( pt - p ).norm() ;
}
