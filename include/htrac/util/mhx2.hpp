#ifndef MHX2_HPP
#define MHX2_HPP

#include <vector>
#include <map>
#include <deque>
#include <Eigen/Core>


struct MHX2Bone {
    Eigen::Vector3f head_, tail_ ;
    Eigen::Matrix4f bmat_ ; // rotation to align Y axis with the bone
    std::string parent_ ;
    float roll_ ;
};

struct MHX2VertexGroup {
    std::vector<int> idxs_ ;
    std::vector<float> weights_ ;
};

#define MAX_VERTICES_PER_FACE 4
#define UNKNOWN_MATERIAL_INDEX 0xffff

struct MHX2Face {
    MHX2Face(const std::deque<uint> &idx) {
        assert(idx.size() <= MAX_VERTICES_PER_FACE) ;
        num_vertices_ = idx.size() ;
        for(int i=0 ; i<num_vertices_ ; i++)
            indices_[i] = idx[i] ;
    }

    uint num_vertices_ ;
    uint indices_[MAX_VERTICES_PER_FACE] ;
    Eigen::Vector2f tex_coords_[MAX_VERTICES_PER_FACE] ;
};

struct UVFace {
    std::deque<uint> indices_ ;
};

struct MHX2Mesh {
    std::vector<Eigen::Vector3f> vertices_ ;
    std::vector< MHX2Face > faces_ ;
    std::vector<Eigen::Vector2f> uv_coords_ ;
    std::vector< UVFace > uv_faces_ ;
    std::map<std::string, MHX2VertexGroup> groups_ ;
    std::vector<Eigen::Vector3f> normals_ ;

    void computeNormals() ;
};

struct MHX2Geometry {
    Eigen::Vector3f offset_ ;
    float scale_ ;
    std::string material_ ;
    MHX2Mesh mesh_ ;
};

struct MHX2Material {
    Eigen::Vector3f diffuse_color_ ;
    float diffuse_map_intensity_ ;
    Eigen::Vector3f specular_color_ ;
    float specular_map_intensity_ ;
    float shininess_ ;
    float opacity_ ;
    float translucency_ ;
    Eigen::Vector3f emissive_color_ ;
    Eigen::Vector3f ambient_color_ ;
    float transparency_map_intensity_ ;
    bool shadeless_ ;
    bool wireframe_ ;
    bool transparent_ ;
    bool alpha_to_coverage_  ;
    bool backface_cull_ ;
    bool depthless_ ;
    bool cast_shadows_ ;
    bool recieve_shadows_ ;
    bool sss_enabled_ ;
    float sss_R_scale_, sss_G_scale_, sss_B_scale_ ;
    std::string diffuse_texture_ ;
};

struct MHX2Model {
    std::map<std::string, MHX2Bone> bones_ ;
    std::map<std::string, MHX2Geometry> geometries_ ;
    std::map<std::string, MHX2Material> materials_ ;
    Eigen::Vector3f offset_ ;

    const MHX2Bone *getBone(const std::string &name) const {
        auto it = bones_.find(name) ;
        if ( it != bones_.end() ) return &(it->second) ;
        else return nullptr ;
    }
};

#endif
