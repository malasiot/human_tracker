#pragma once

#include <htrac/util/mhx2.hpp>
#include <htrac/util/mesh.hpp>
#include <htrac/model/parameterization.hpp>
#include <htrac/model/pose.hpp>

#include <map>
#include <memory>

#include <Eigen/Geometry>


struct Bone {
    Bone *parent_ = nullptr ;
    Eigen::Matrix4f offset_ ; // aligns the bone with the armature
    Eigen::Matrix4f mat_ ; // relative transform between current node and its parent
    std::string name_ ;
    float length_ ;
    std::vector<Bone *> children_ ;

    bool isChildOf(const Bone *other) const{
        const Bone *p = other ;
        while ( p != this ) {
            if ( p == nullptr ) return false ;
            p = p->parent_ ;
        }
        return true ;
    }
};

struct PoseBone {
    const std::string &name() const { return bone_->name_ ; }
    const RotationParameterization *getParameterization() const { return rp_.get() ; }
    size_t offset() const { return offset_ ; }
    size_t dofs() const { return rp_->dim() ; }

    const Bone *bone_ ;
    std::shared_ptr<RotationParameterization> rp_ ;
    size_t offset_ = 0 ;
};

using FullPose = std::map<std::string, Eigen::Matrix4f> ;

struct Skeleton {

    Skeleton() = default;

    void setPoseBones(const PoseParameterization &pp) ;
    const std::vector<PoseBone> &getPoseBones() const { return pbones_ ; }

    void load(const std::string &fileName, bool zUp = false) ;
    void load(const MHX2Model &model);

    const std::vector<Bone> &bones() const { return bones_ ; }

    const Bone &getBone(uint32_t idx) const { return bones_[idx] ; }

    std::map<std::string, Eigen::Vector3f> getJointCoordinates(const std::vector<std::string> &jnames, const Pose &p) const;

    void computeBoneTransforms(const Pose &p, std::map<std::string, Eigen::Matrix4f> &trs) const ;
    void computeBoneTransforms(const Pose &p, std::vector<Eigen::Matrix4f> &trs) const ;

    void computeBoneTransforms(const FullPose &p, std::map<std::string, Eigen::Matrix4f> &trs) const ;
    void computeBoneTransforms(const FullPose &p, std::vector<Eigen::Matrix4f> &trs) const ;

    Eigen::Matrix4f computeBoneTransform(const Pose &pose, const std::string &boneName) const ;
    Eigen::Matrix4f computeBoneTransform(const Pose &pose, uint boneIdx) const;

    Eigen::Matrix4f computeBoneTransform(const FullPose &pose, uint boneIdx) const;
    Eigen::Matrix4f computeBoneTransform(const FullPose &pose, const std::string &boneName) const;

    std::vector<std::string> getBoneNames() const ;
    const Bone *getRoot() const { return root_ ; }

    // get derivatives of bone "b" transform with respect to quaternion rotation of bone "v"
    void computeBoneRotationDerivatives(const Pose &pose, const Bone *b, const PoseBone *v, Eigen::Matrix4f dr[]) const;
    void computeBoneRotationDerivatives(const Pose &pose, const std::string &b, const std::string &v, Eigen::Matrix4f dr[]) const ;
    void computeBoneRotationDerivatives(const Pose &pose, uint bidx, uint vidx, Eigen::Matrix4f dr[]) const ;

    void computeGlobalDerivatives(const Pose &pose, const std::string &b,  Eigen::Matrix4f dt[3], Eigen::Matrix4f dr[4], bool norm = true) const ;
    void computeGlobalDerivatives(const Pose &pose, uint bidx, Eigen::Matrix4f dt[3], Eigen::Matrix4f dr[4], bool norm = true) const ;

    const Bone *findBone(const std::string &bn) const ;
    int32_t getBoneIndex(const std::string &bn) const ;

    const PoseBone *findPoseBone(const std::string &bn) const ;
    uint getNumPoseBoneParams() const ;

    // find the skeleton pose that fits a set of joint coordinates values
    // free_bones: are the set of bones that are going to be optimized
    // orig: original pose to set at start of iteration
    // thresh: the error threshold to stop iterations

    static Pose fit(const Skeleton &sk,
                    const std::map<std::string, Eigen::Vector3f> &jc,
                    const Pose &orig, float thresh = 0.01) ;


    Model3d toMesh(const Pose &pose) const;

protected:

    std::vector<Bone> bones_ ;
    std::vector<PoseBone> pbones_ ;
    std::map<std::string, size_t> bone_map_ ;
    std::map<std::string, size_t> pbone_map_ ;
    Bone *root_ ;

protected:

    void computeBoneTransformsRecursive(const Bone *bone, const Pose &p, const Eigen::Matrix4f &parent, Pose &trs) const ;
    void computeBoneTransformsRecursive(const Bone *bone, const Pose &p, const Eigen::Matrix4f &parent, std::vector<Eigen::Matrix4f> &trs) const ;
    void computeBoneTransformsRecursive(const Bone *bone, const FullPose &p, const Eigen::Matrix4f &parent, std::vector<Eigen::Matrix4f> &trs) const ;

    Eigen::Matrix4f computeBoneTransformRecursive(const Bone *b, const Pose &p) const ;
    Eigen::Matrix4f computeBoneTransformRecursive(const Bone *b, const FullPose &p) const ;

    void computeBoneRotationDerivativesRecursive(const Pose &pose, const Bone *b, const PoseBone *bv,
                                                  Eigen::Matrix4f dr[], Eigen::Matrix4f &G) const ;

    void computeBoneGlobalDerivativesRecursive(const Pose &pose, const Bone *b,
                                               Eigen::Matrix4f dt[3], Eigen::Matrix4f dr[4], bool nrm) const ;

};

// parameterization of the pose of a set of bones.
// First bone should be the root of the skeleton. Root bone transformation is
// 3 translation parameters + 4 rotation parameters (quaternion).
// The rest of the bones are 4 rotation parameters each

#if 0
class SkeletonParameters {
public:

    SkeletonParameters(const Eigen::VectorXf &c): coeffs_(c) {}
    SkeletonParameters(const Pose &p, const PoseParameterization &bones) ;
    Pose getPose(const PoseParameterization &bones) const ;

    const Eigen::VectorXf &coeffs() const { return coeffs_ ; }
private:
    Eigen::VectorXf coeffs_ ;
};
#endif
