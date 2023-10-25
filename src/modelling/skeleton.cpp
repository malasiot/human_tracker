#include <htrac/model/skeleton.hpp>
#include <htrac/util/mhx2_importer.hpp>

#include <Eigen/Geometry>
#include <iostream>
#include <fstream>

using namespace Eigen ;
using namespace std ;


void skeleton_compute_relative(const MHX2Model &model, const std::string &bone, const Matrix4f &parent_transform, std::map<std::string, Matrix4f> &trs) {
    auto it = model.bones_.find(bone) ;
    assert(it != model.bones_.end());
    const auto &b = it->second ;
    auto mat = parent_transform.inverse() * b.bmat_ ;
    trs.emplace(bone, mat) ;

    for( const auto &bp: model.bones_ ) {
        if ( bp.first == bone ) continue ;
        if ( bp.second.parent_ == bone ) {
            skeleton_compute_relative(model, bp.first, b.bmat_, trs);
        }
    }
}

Matrix4f quat2mat(const Quaternionf &q) {
    Matrix3f mat3 = q.toRotationMatrix();
    Matrix4f mat4 = Matrix4f::Identity();
    mat4.block(0,0,3,3) = mat3;
    return mat4 ;
}

Matrix4f tran2mat(const Vector3f &t) {
    Matrix4f mat4 = Matrix4f::Identity();
    mat4.block(0,3,3,1) = t;
    return mat4 ;
}

void Skeleton::setPoseBones(const PoseParameterization &pp) {
    size_t offset = 0 ;
    for ( const auto &bp: pp ) {
        PoseBone pose_bone ;
        pose_bone.bone_ = findBone(bp.first) ;
        pose_bone.rp_ = bp.second ;
        pose_bone.offset_ = offset ;
        offset += bp.second->dim() ;
        pbone_map_.emplace(bp.first, pbones_.size()) ;
        pbones_.emplace_back(pose_bone) ;
    }
}

void Skeleton::load(const std::string &fileName, bool zUp)
{
    Mhx2Importer importer ;
    if ( importer.load(fileName, zUp) )
        load(importer.getModel()) ;

}

void Skeleton::load(const MHX2Model &model) {
    std::map<string, string> bpmap ;
    // copy bones
    for( const auto &bp: model.bones_ ) {
        const auto &bone = bp.second ;

        Bone nb ;
        nb.name_ = bp.first ;
        nb.offset_ = bone.bmat_ ;
        nb.length_ = (bone.tail_ - bone.head_).norm() ;

        bone_map_.emplace(bp.first, bones_.size()) ;
        bones_.emplace_back(std::move(nb)) ;

        bpmap.emplace(bp.first, bone.parent_) ;
    }

    // build hierarchy
    for( auto &bone: bones_ ) {
        auto it = bpmap.find(bone.name_) ;
        auto parent_name = it->second ;
        if ( parent_name.empty() ) {
            root_ = &bone ;
        } else {
            auto idx = getBoneIndex(parent_name) ;
            Bone &parent_bone = bones_[idx] ;
            bone.parent_ = &parent_bone ;
            parent_bone.children_.push_back(&bone) ;
        }
    }

    //   root_->mat_.block<3, 1>(0, 3) = Vector3f::Zero() ;

    // compute relative transforms

    std::map<std::string, Matrix4f> trs ;
    skeleton_compute_relative(model, root_->name_, Matrix4f::Identity(), trs);

    for( auto &bone: bones_ ) {
        bone.mat_ = trs[bone.name_] ;
    }
}

std::map<string, Vector3f> Skeleton::getJointCoordinates(const std::vector<string> &jnames, const Pose &p) const {
    std::map<string, Vector3f> coords ;

    std::map<string, Matrix4f> trs ;
    computeBoneTransforms(p, trs) ;

    for( const auto &joint: jnames ) {
        auto ptr = trs[joint] ;
        coords.emplace(joint, Vector4f(ptr * Vector4f{0, 0, 0, 1}).head<3>());
    }

    return coords ;
}

void Skeleton::computeBoneTransforms(const Pose &p, map<string, Matrix4f> &trs) const {
    vector<Matrix4f> tvec ;
    computeBoneTransforms(p, tvec) ;
    for( size_t i=0 ; i<tvec.size() ; i++ ) {
        trs.emplace(bones_[i].name_, tvec[i]) ;
    }
}

void Skeleton::computeBoneTransforms(const FullPose &p, map<string, Matrix4f> &trs) const {
    vector<Matrix4f> tvec ;
    computeBoneTransforms(p, tvec) ;
    for( size_t i=0 ; i<tvec.size() ; i++ ) {
        trs.emplace(bones_[i].name_, tvec[i]) ;
    }
}

void Skeleton::computeBoneTransforms(const Pose &p, std::vector<Eigen::Matrix4f> &trs) const {
    trs.resize(bones_.size());
    computeBoneTransformsRecursive(root_, p, Matrix4f::Identity(), trs) ;
}

void Skeleton::computeBoneTransforms(const FullPose &p, std::vector<Eigen::Matrix4f> &trs) const {
    trs.resize(bones_.size());
    computeBoneTransformsRecursive(root_, p, Matrix4f::Identity(), trs) ;
}

void Skeleton::computeBoneTransformsRecursive(const Bone *bone, const Pose &p,
                                              const Eigen::Matrix4f &parent_transform, std::vector<Eigen::Matrix4f> &trs) const {
    string node_name(bone->name_);

    Matrix4f global_tr ;
    if ( bone == root_ ) {
        global_tr = p.getGlobalTransform() *  bone->mat_ ;
    } else {
        Matrix4f node_tr = bone->mat_;

        auto ptr = p.getBoneTransform(node_name) ;
        node_tr = node_tr * ptr ;

        global_tr = parent_transform * node_tr;
    }

    auto idx = getBoneIndex(node_name) ;

    trs[idx] = global_tr ;

    for( Bone *child: bone->children_ ) {
        computeBoneTransformsRecursive(child, p, global_tr, trs) ;
    }
}

void Skeleton::computeBoneTransformsRecursive(const Bone *bone, const FullPose &p,
                                              const Eigen::Matrix4f &parent_transform, std::vector<Eigen::Matrix4f> &trs) const {
    string node_name(bone->name_);

    Matrix4f global_tr ;

    Matrix4f node_tr = bone->mat_, ptr = Matrix4f::Identity();

    auto it = p.find(node_name) ;

    if ( it != p.end() )
        ptr = it->second ;

    node_tr = node_tr * ptr ;

    global_tr = parent_transform * node_tr;


    auto idx = getBoneIndex(node_name) ;

    trs[idx] = global_tr ;

    for( Bone *child: bone->children_ ) {
        computeBoneTransformsRecursive(child, p, global_tr, trs) ;
    }
}

Matrix4f Skeleton::computeBoneTransformRecursive(const Bone *b, const Pose &p) const {
    if ( b == root_ ) {
        return p.getGlobalTransform() * b->mat_ ;
    }
    else {
        Matrix4f node_tr = b->mat_;
        node_tr = node_tr * p.getBoneTransform(b->name_) ;
        return computeBoneTransformRecursive(b->parent_, p) * node_tr  ;
    }
}


Matrix4f Skeleton::computeBoneTransformRecursive(const Bone *b, const FullPose &p) const {
    Matrix4f node_tr = b->mat_ ;

    auto it = p.find(b->name_) ;
    if ( it != p.end() )
        node_tr = node_tr * it->second ;

    if ( b->parent_ == nullptr ) return node_tr ;
    else return computeBoneTransformRecursive(b->parent_, p) * node_tr  ;
}

Matrix4f Skeleton::computeBoneTransform(const Pose &pose, const std::string &boneName) const {
    const Bone *b = findBone(boneName) ;
    if ( b == nullptr )
        return Matrix4f::Identity() ;
    else
        return computeBoneTransformRecursive(b, pose)   ;
}

Matrix4f Skeleton::computeBoneTransform(const Pose &pose, uint boneIdx) const {
    const Bone *b = &bones_[boneIdx] ;
    if ( b == nullptr )
        return Matrix4f::Identity() ;
    else
        return computeBoneTransformRecursive(b, pose)   ;
}

Matrix4f Skeleton::computeBoneTransform(const FullPose &pose, uint boneIdx) const
{
    const Bone *b = &bones_[boneIdx] ;
    if ( b == nullptr )
        return Matrix4f::Identity() ;
    else
        return computeBoneTransformRecursive(b, pose)   ;
}

Matrix4f Skeleton::computeBoneTransform(const FullPose &pose, const std::string &boneName) const {
    const Bone *b = findBone(boneName) ;
    if ( b == nullptr )
        return Matrix4f::Identity() ;
    else
        return computeBoneTransformRecursive(b, pose)   ;
}

void Skeleton::computeBoneRotationDerivatives(const Pose &pose, const Bone *b, const PoseBone *v,  Eigen::Matrix4f dr[]) const {
    Matrix4f G ;
    computeBoneRotationDerivativesRecursive(pose, b, v, dr, G)  ;
}

void Skeleton::computeBoneRotationDerivatives(const Pose &pose, const string &bname, const string &vname,  Eigen::Matrix4f dr[]) const {
    const Bone *b = findBone(bname) ;
    const PoseBone *v = findPoseBone(vname) ;
    assert(b && v) ;
    computeBoneRotationDerivatives(pose, b, v, dr);
}

void Skeleton::computeBoneRotationDerivatives(const Pose &pose, uint bidx, uint vidx, Eigen::Matrix4f dr[]) const {
    const Bone *b = &bones_[bidx] ;
    const PoseBone *v = &pbones_[vidx] ;
    assert(b && v) ;
    computeBoneRotationDerivatives(pose, b, v, dr);
}


void Skeleton::computeGlobalDerivatives(const Pose &pose, const std::string &bname,
                                        Eigen::Matrix4f dt[3], Eigen::Matrix4f dr[4], bool nrm) const {
    const Bone *b = findBone(bname) ;
    assert(b) ;
    computeBoneGlobalDerivativesRecursive(pose, b, dt, dr, nrm)  ;
}

void Skeleton::computeGlobalDerivatives(const Pose &pose, uint bidx,
                                        Eigen::Matrix4f dt[3], Eigen::Matrix4f dr[4],  bool nrm) const {
    const Bone *b = &bones_[bidx] ;
    assert(b) ;
    computeBoneGlobalDerivativesRecursive(pose, b, dt, dr, nrm)  ;
}

const Bone *Skeleton::findBone(const std::string &bn) const {
    auto idx =  getBoneIndex(bn) ;
    return ( idx == -1 ) ? nullptr : &(bones_[idx]) ;
}

int32_t Skeleton::getBoneIndex(const std::string &bn) const {
    auto it = bone_map_.find(bn) ;
    return it == bone_map_.end() ? -1 : it->second ;
}

const PoseBone *Skeleton::findPoseBone(const std::string &bn) const {
    auto it = pbone_map_.find(bn) ;
    return it == pbone_map_.end() ? nullptr : &(pbones_[it->second]) ;
}

uint Skeleton::getNumPoseBoneParams() const {
    uint n = 0 ;
    for( const auto &pb: pbones_ ) {
        n += pb.rp_->dim() ;
    }
    return n ;
}



void split(const Matrix4f &tr, Matrix4f &T, Matrix4f &R, Matrix4f &S) {
    Affine3f a{tr};
    Affine3f rotation{a.rotation()};
    Affine3f translation{Translation3f(a.translation())};

    float Sx = tr.block<3, 1>(0, 0).norm();
    float Sy = tr.block<3, 1>(0, 1).norm();
    float Sz = tr.block<3, 1>(0, 2).norm();

    Affine3f scale{Scaling(Sx, Sy, Sz)};

    R = rotation.matrix() ;
    S = scale.matrix() ;
    T = translation.matrix() ;
}

void Skeleton::computeBoneRotationDerivativesRecursive(const Pose &pose, const Bone *b, const PoseBone *bv,  Matrix4f dr[4], Matrix4f &G) const {

    const RotationParameterization *rp = bv->getParameterization() ;

    uint n = rp->dim() ;

    if ( b == root_ ) {
        G = pose.getGlobalTransform() * b->mat_ ;
        for( int i=0 ; i<n ; i++ )
            dr[i] = Matrix4f::Zero() ;
    } else {
        Matrix4f Gp ;
        computeBoneRotationDerivativesRecursive(pose, b->parent_, bv,  dr, Gp) ;

        auto ptr = pose.getBoneRotation(b->name_) ;
        Matrix4f A = b->mat_ * quat2mat(ptr) ;
        G = Gp * A ;

        if ( b == bv->bone_ ) {
            Matrix4f rr[4] ;
            VectorXf Q = pose.getBoneParams(b->name_) ;

            rp->jacobian(Q, rr) ;

            for(int i=0 ; i<n ; i++)
                dr[i] = dr[i] * A + Gp * b->mat_ * rr[i] ; // product rule
        } else {
            for(int i=0 ; i<n ; i++)
                dr[i] = dr[i] * A ;
        }
    }
}

void Skeleton::computeBoneGlobalDerivativesRecursive(const Pose &pose, const Bone *b,
                                                     Eigen::Matrix4f dt[], Eigen::Matrix4f dr[], bool nrm) const
{
    if ( b == root_ ) {
        Matrix4f T, R ;

        T.setIdentity() ;
        T.block<3, 1>(0, 3) = pose.getGlobalTranslation() ;
        Quaternionf Q = pose.getGlobalRotation() ;
        R.setIdentity() ;
        R.block<3, 3>(0, 0) = Q.toRotationMatrix() ;

        Matrix4f rr[4] ;
        auto q = pose.getGlobalRotationParams() ;

#if POSE_USE_QUAT_PARAM
        QuaternionParameterization qp(nrm) ;
        qp.jacobian(q, rr) ;
#else
        ExponentialParameterization xp ;
        xp.jacobian(q, rr) ;
#endif

        for(int i=0 ; i<Pose::global_rot_params ; i++) {
            dr[i] = T * rr[i] * b->mat_;
        }

        for (uint i=0 ; i<3 ; i++) {
            Matrix4f dT = Matrix4f::Zero() ;
            dT(i, 3) = 1.0 ;
            dt[i] = dT * R * b->mat_ ;
        }

    } else {
        auto ptr = pose.getBoneTransform(b->name_) ;
        computeBoneGlobalDerivativesRecursive(pose, b->parent_,  dt, dr, nrm) ;

        for( size_t i=0 ; i<Pose::global_rot_params ; i++ )
            dr[i] = dr[i] * b->mat_ * ptr ;
        for( size_t i=0 ; i<3 ; i++ )
            dt[i] = dt[i] * b->mat_ * ptr ;

    }

}


vector<string> Skeleton::getBoneNames() const {
    vector<string> bnames ;
    for( const auto &b: bones_ ) {
        bnames.push_back(b.name_) ;
    }
    return bnames ;
}

/*****************************************************************************************/

Model3d Skeleton::toMesh(const Pose &pose) const
{
    map<string, Matrix4f> trs ;
    computeBoneTransforms(pose, trs) ;

    Model3d model ;
    for( const auto &b : bones_ ) {
        auto ptr = trs[b.name_] ;

        Mesh m = Mesh::makeCylinder(0.02, b.length_, 31, 8, true) ;

        Isometry3f offset ;
        offset.setIdentity() ;
        offset.translate(Vector3f{0, b.length_/2.0, 0}) ;
        offset.rotate(AngleAxisf(M_PI/2, Vector3f::UnitX())) ;

        Model3d bm ;
        bm.setGeometry(m).setPose(ptr * offset.matrix()) ;
        model.addChild(std::move(bm)) ;
    }

    return model ;
}


