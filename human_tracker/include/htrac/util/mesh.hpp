#pragma once

#include <vector>
#include <cstdint>
#include <Eigen/Core>

#include <memory>

#include <assimp/mesh.h>
#include <assimp/scene.h>

class Mesh {
public:
    using indices_t = std::vector<uint32_t> ;

    Mesh() = default ;

    using vb3_t = std::vector<Eigen::Vector3f> ;
    using vb2_t = std::vector<Eigen::Vector2f> ;

    vb3_t &vertices() { return vertices_ ; }
    vb3_t &normals() { return normals_ ; }
    vb3_t &colors() { return colors_ ; }
    indices_t &indices() { return indices_ ; }
    const indices_t &indices() const { return indices_ ; }

    const vb3_t &vertices() const { return vertices_ ; }
    const vb3_t &normals() const { return normals_ ; }
    const vb3_t &colors() const { return colors_ ; }

    // primitive shape factories

    static Mesh makeCube(const Eigen::Vector3f &hs) ;

    static Mesh makeSphere(float radius, size_t slices, size_t stacks) ;

    static Mesh makeCylinder(float radius, float height, size_t slices, size_t stacks, bool add_caps = true) ;

    static Mesh makePointCloud(const std::vector<Eigen::Vector3f> &pts) ;
    static Mesh makePointCloud(const std::vector<Eigen::Vector3f> &coords,
                                   const std::vector<Eigen::Vector3f> &clrs) ;

private:

    vb3_t vertices_, normals_, colors_ ;
    indices_t indices_ ;

private:
    friend class Model3d ;
    aiMesh *createAssimpMesh() const;
};

class Model3d {
public:
    Model3d &setGeometry(const Mesh &m) {
        mesh_ = m ; return *this ;
    }

    Model3d &setPose(const Eigen::Matrix4f &pose) {
        pose_ = pose ; return *this ;
    }

    Model3d &addChild(Model3d &&child) {
        children_.emplace_back(std::move(child)) ;
        return *this ;
    }

    void write(const std::string &fname) ;

private:
    std::vector<Model3d> children_ ;
    Mesh mesh_ ;
    Eigen::Matrix4f pose_ = Eigen::Matrix4f::Identity();

private:

    aiNode *createAssimpNode(aiNode *parent, std::vector<aiMesh *> &meshes) const ;
};
