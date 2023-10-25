#pragma once

#include <htrac/model/skeleton.hpp>
#include <htrac/model/sdf_model.hpp>
#include <htrac/util/mesh.hpp>

#include <map>
#include <Eigen/Geometry>
#include <memory>

struct Primitive {
    virtual ~Primitive() {}
    virtual float sdf(const Eigen::Vector3f &p) const = 0 ;
    virtual Eigen::Vector3f grad(const Eigen::Vector3f &p) const = 0 ;
    virtual Mesh mesh() const = 0 ;
};

// a round cone is defined by swiping a sphere at distance l from an initial radious of r1 to r2
// the coordinate system is on the top of the first sphere with the y axis along the cone axis

struct RoundCone: public Primitive {
public:
    RoundCone(float l, float r1, float r2): l_(l), r1_(r1), r2_(r2) {}

    float sdf(const Eigen::Vector3f &p) const override;
    Eigen::Vector3f grad(const Eigen::Vector3f &p) const override ;

    Mesh mesh() const override ;


    void toMesh(size_t slices, size_t stacks, size_t head_stacks,
                std::vector<Eigen::Vector3f> &vertices,
                std::vector<uint32_t> &indices) const;

    float l_, r1_, r2_ ;
 };

struct Box: public Primitive {
    Box(const Eigen::Vector3f &hs, float r): hs_(hs), r_(r) {}

    float sdf(const Eigen::Vector3f &p) const override;
    Eigen::Vector3f grad(const Eigen::Vector3f &p) const override ;
    Mesh mesh() const override ;

    float r_ ;
    Eigen::Vector3f hs_ ;
};

struct Sphere: public Primitive {
    Sphere(float r): r_(r) {}

    float sdf(const Eigen::Vector3f &p) const override { return p.norm() - r_ ; }
    Eigen::Vector3f grad(const Eigen::Vector3f &p) const override ;
    Mesh mesh() const override ;

    float r_ ;
};

struct ElongatedPrimitive: public Primitive {
    ElongatedPrimitive(Primitive *p, const Eigen::Vector3f &eh): base_(p), eh_(eh) {}

    float sdf(const Eigen::Vector3f &p) const override ;
    Eigen::Vector3f grad(const Eigen::Vector3f &p) const override ;

    std::unique_ptr<Primitive> base_ ;
    Eigen::Vector3f eh_ ;
};


struct ScaledPrimitive: public Primitive {
    ScaledPrimitive(Primitive *p, const Eigen::Vector3f &scale): base_(p), scale_(scale) {}

    float sdf(const Eigen::Vector3f &p) const override ;
    Eigen::Vector3f grad(const Eigen::Vector3f &p) const override ;
    Mesh mesh() const override ;

    std::unique_ptr<Primitive> base_ ;
    Eigen::Vector3f scale_ ;
};

struct TransformedPrimitive: public Primitive {
    TransformedPrimitive(Primitive *p, const Eigen::Isometry3f &tr): base_(p), tr_(tr) {
        itr_ = tr.inverse() ;
    }

    float sdf(const Eigen::Vector3f &p) const override {
        return base_->sdf(itr_ * p) ;
    }

    Eigen::Vector3f grad(const Eigen::Vector3f &p) const override ;

    Mesh mesh() const override ;

    std::unique_ptr<Primitive> base_ ;
    Eigen::Isometry3f tr_, itr_ ;
};

class PrimitiveSDF: public SDFModel {
public:
    void addPrimitive(uint bidx, Primitive *p) {
        primitives_.emplace_back(std::make_pair(bidx, p)) ;
    }

    void readJSON(const Skeleton &sk, const std::string &path);

    uint getNumParts() const override { return primitives_.size() ; }
    uint getPartBone(uint part) const override { return primitives_[part].first; }

    Model3d makeMesh(const Skeleton &sk, const Pose &p) ;

    //float sdf(const Skeleton &sk, const Pose &pose, const Eigen::Vector3f &p) const ;
   // Eigen::Vector3f grad(const Skeleton &sk, const Pose &pose, const Eigen::Vector3f &p) const  ;

    float eval(const Skeleton &sk, const Eigen::Vector3f &v, const Pose &p) const override ;
    Eigen::Vector3f grad(const Skeleton &sk, const Eigen::Vector3f &v, const Pose &p) const override ;

    float evalPart(uint part, const Eigen::Vector3f &v, const Eigen::Matrix4f &imat) const override;
    Eigen::Vector3f gradPart(uint part, const Eigen::Vector3f &v, const Eigen::Matrix4f &imat) const override ;

    // return a M x N matrix with distance to each part
    Eigen::MatrixXf eval(const Eigen::MatrixXf &pts) const override ;

    // return a 3 x N matrix withe columns equals to the gradients of the SDF at given points
    Eigen::MatrixXf grad(const Eigen::MatrixXf &pts, const std::vector<uint> &idxs) const override;

    const Primitive *getPrimitive(uint idx) const { return primitives_[idx].second.get() ; }
private:


    std::vector<std::pair<uint, std::unique_ptr<Primitive>>> primitives_ ;

};
