#ifndef HTRAC_MODEL_SKINNED_MESH_HPP
#define HTRAC_MODEL_SKINNED_MESH_HPP

#include <htrac/model/skeleton.hpp>

// A generic skinned mesh
#define MAX_WEIGHTS_PER_VERTEX 4
class SkinnedMesh {
public:

    SkinnedMesh() {}

    void load(const std::string &fname, bool zUp = false) ;

    void load(const MHX2Model &model);

    // apply the given bone transformations to the mesh and return new vertices and normals.

    void getTransformedVertices(const Pose &p, std::vector<Eigen::Vector3f> &mpos, std::vector<Eigen::Vector3f> &mnorm) const;

    void getTransformedVertices(const Pose &p, std::vector<Eigen::Vector3f> &mpos) const ;

    Model3d toMesh(const Pose &p) const ;

public:



    struct VertexBoneData
    {
        int id_[MAX_WEIGHTS_PER_VERTEX];
        float weight_[MAX_WEIGHTS_PER_VERTEX];

        VertexBoneData() ;

        void reset() ;
        void addBoneData(uint boneID, float w) ;
        void normalize() ;
    };

    // flat version of mesh data (e.g. to be used for rendering)

    std::vector<Eigen::Vector3f> positions_ ;
    std::vector<Eigen::Vector3f> normals_ ;
    std::vector<Eigen::Vector2f> tex_coords_ ;
    std::vector<VertexBoneData> bones_ ;
    std::vector<uint> indices_ ;
    std::vector<double> weights_ ;
    std::map<std::string, uint32_t> bone_idx_ ;

    Skeleton skeleton_ ;


    std::vector<Eigen::Matrix4f> getSkinningTransforms(const Pose &p) const ;
};


#endif
