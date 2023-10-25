#include <htrac/pose/pose_from_keypoints.hpp>

#include <cvx/imgproc/rgbd.hpp>
#include <cvx/pcl/align.hpp>



using namespace Eigen ;
using namespace std ;

bool PoseFromKeyPoints::hasChest(const KeyPoints3 &kpts) {
    auto itl = kpts.find("LeftArm") ;
    auto itr = kpts.find("RightArm") ;

    if ( itl != kpts.end() && itr != kpts.end() ) {
        const Vector3f &kl = itl->second ;
        const Vector3f &kr = itr->second ;

        float dist = (kl - kr).norm() ;

        if ( dist < 0.5f && dist > 0.2f ) return true ;
    }

    return false ;
}

bool PoseFromKeyPoints::hasHips(const KeyPoints3 &kpts) {
    auto itl = kpts.find("LeftUpLeg") ;
    auto itr = kpts.find("RightUpLeg") ;

    if ( itl != kpts.end() && itr != kpts.end() ) {
        const Vector3f &kl = itl->second ;
        const Vector3f &kr = itr->second ;

        float dist = (kl - kr).norm() ;

        if ( dist < 0.6f && dist > 0.4f ) return true ;
    }

    return false;
}

Vector3f PoseFromKeyPoints::approxHipsFromChest(const KeyPoints3 &kpts)
{
    auto itl = kpts.find("LeftArm") ;
    auto itr = kpts.find("RightArm") ;

    assert( itl != kpts.end() && itr != kpts.end() ) ;

    const Vector3f &kl = itl->second ;
    const Vector3f &kr = itr->second ;

    Vector3f mid = ( kl + kr )/2.0f ;
    mid.y() -= 0.5f ;
    return mid ;
}

bool PoseFromKeyPoints::estimate(const KeyPoints &kps, const cvx::PinholeCamera &cam, const cv::Mat &depth, Pose &result) {

    // neutral pose
    Pose pose(&skeleton_) ;

    // lift key points using depth image
    KeyPoints3 kpts3 = getKeyPoints3d(kps, cam, depth) ;

    bool has_chest = hasChest(kpts3) ;
    bool has_hips = hasHips(kpts3) ;

    if ( !has_chest ) return false ;

    static vector<std::string> torso_joints = {"LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"} ;
    auto jcoords = skeleton_.getJointCoordinates(torso_joints, pose) ;
    auto hips_coords = skeleton_.getJointCoordinates({"Hips"}, pose) ;


    vector<Vector3f> src, dst ;
    for( const auto &jn: torso_joints ) {
        auto it = kpts3.find(jn) ;
        if ( it != kpts3.end() ) {
            src.push_back((*it).second) ;
            dst.push_back(jcoords[jn]) ;
        }

    }


    if ( !has_hips ) {
        dst.push_back(hips_coords["Hips"]) ;
        src.push_back(approxHipsFromChest(kpts3)) ;
    }


    // too few valid keypoints
    if ( src.size() < 3 ) return false ;

    // compute rigid transform between neutral pose joints and torso keypoints
    Isometry3f tr = cvx::alignRigid(dst, src) ;

    // initialize pose
 //   pose.setGlobalRotation(Quaternionf(tr.linear()));
  //  pose.setGlobalTranslation(tr.translation());

    // nlms fit of all points

    result = Skeleton::fit(skeleton_, kpts3, pose) ;

    return true ;
}


PoseFromKeyPoints::KeyPoints3 PoseFromKeyPoints::getKeyPoints3d(const KeyPoints &kpts, const cvx::PinholeCamera &cam, const cv::Mat &depth) {
    KeyPoints3 res ;

    for( const auto &kp: kpts ) {
        const auto &name = kp.first ;
        const auto &coords = kp.second.first ;
        const auto &weight = kp.second.second ;
        if ( skeleton_.findBone(name) == nullptr ) continue ;

        ushort z ;
        if ( cvx::sampleNearestNonZeroDepth(depth, round(coords.x()), round(coords.y()), z, 3) && weight > thresh_ ) {
            Vector3f p = cam.backProject(coords.x(), coords.y(), z/1000.0) ;
            p.y() =-p.y() ;
            p.z() =-p.z() ;
            res.emplace(name, p) ;
        }
    }

    return res ;
}
