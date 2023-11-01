#include <htrac/util/mesh.hpp>
#include <cvx/misc/strings.hpp>

#include <Eigen/Core>
#include <iostream>

#include <assimp/Exporter.hpp>


using namespace std ;
using namespace Eigen ;

Mesh Mesh::makeCube(const Eigen::Vector3f &hs) {

    Mesh m ;
    m.vertices() = {
    { -hs.x(), +hs.y(), +hs.z() },
    { +hs.x(), +hs.y(), +hs.z() },
    { +hs.x(),  -hs.y(),  +hs.z() },
    { -hs.x(), -hs.y(), +hs.z() },
    { -hs.x(), +hs.y(), -hs.z() },
    { +hs.x(), +hs.y(), -hs.z() },
    { +hs.x(), -hs.y(), -hs.z() },
    { -hs.x(), -hs.y(), -hs.z() } } ;

    m.indices() = {  1, 0, 3,  7, 4, 5,  4, 0, 1,  5, 1, 2,  2, 3, 7,  0, 4, 7,  1, 3, 2,  7, 5, 6,  4, 1, 5,  5, 2, 6,  2, 7, 6, 0, 7, 3};
    return m ;
}

void makeCircleTable(vector<float> &sint, vector<float> &cost, int n, bool half_circle = false) {

    /* Table size, the sign of n flips the circle direction */

    const size_t size = abs(n);

    /* Determine the angle between samples */

    const float angle = (half_circle ? 1 : 2)*M_PI/(float)( ( n == 0 ) ? 1 : n );

    sint.resize(size+1) ; cost.resize(size+1) ;

    /* Compute cos and sin around the circle */

    sint[0] = 0.0;
    cost[0] = 1.0;

    for ( size_t i =1 ; i<size; i++ ) {
        sint[i] = sin(angle*i);
        cost[i] = cos(angle*i);
    }

    /* Last sample is duplicate of the first */

    sint[size] = sint[0];
    cost[size] = cost[0];
}



Mesh Mesh::makeSphere(float radius, size_t slices, size_t stacks) {

    Mesh m ;

    auto &vertices = m.vertices() ;
    auto &normals = m.normals() ;
    auto &indices = m.indices() ;

    int idx = 0;
    float x,y,z;
    int n_vertices ;

    /* Pre-computed circle */
    vector<float> sint1, cost1, sint2, cost2;

    /* number of unique vertices */
    assert (slices !=0 && stacks > 1 );

    n_vertices = slices*(stacks-1) + 2 ;

    makeCircleTable(sint1, cost1, -slices, false) ;
    makeCircleTable(sint2, cost2, stacks, true) ;

    vertices.resize(n_vertices) ;
    normals.resize(n_vertices) ;

    /* top */

    vertices[0] = { 0.f, radius, 0.0f } ;
    normals[0] = { 0.f, 1.0f, 0.0 } ;

    idx = 1;

    /* each stack */
    for( unsigned int i=1; i<stacks; i++ )
    {
        for( unsigned int j=0; j<slices; j++, idx++)
        {
            x = cost1[j]*sint2[i];
            z = sint1[j]*sint2[i];
            y = cost2[i];

            vertices[idx] = { x*radius, y*radius, z*radius } ;
            normals[idx] = { x, z, y } ;
        }
    }

    vertices[idx] = { 0.0f, -radius, 0.0f } ;
    normals[idx] = { 0.0f, -1, 0.0f } ;

    indices.resize(6*slices + 6*(stacks-2)*slices) ;

    /* top stack */

    idx = 0 ;
    for ( unsigned int j=0;  j<slices-1;  j++) {
        indices[idx++] = 0 ;
        indices[idx++] = j+1 ;
        indices[idx++] = j+2;

    }

    indices[idx++] = 1 ;
    indices[idx++] = 0 ;
    indices[idx++] = slices ;

    for ( unsigned int i=0; i< stacks-2 ; i++ )
    {
        unsigned int offset = 1+i*slices;                    /* triangle_strip indices start at 1 (0 is top vertex), and we advance one stack down as we go along */
        unsigned int j ;

        for ( j=0; j<slices-1; j++ ) {
            indices[idx++] = offset + j  + slices;
            indices[idx++] = offset + j + 1 ;
            indices[idx++] = offset + j ;

            indices[idx++] = offset + j + 1 ;
            indices[idx++] = offset + j + slices ;
            indices[idx++] = offset + j + slices + 1;
        }

        indices[idx++] = offset + slices ;
        indices[idx++] = offset  ;
        indices[idx++] = offset + j + slices ;

        indices[idx++] = offset  ;
        indices[idx++] = offset + j ;
        indices[idx++] = offset + j  + slices ;

    }

    /* bottom stack */
    int offset = 1+(stacks-2)*slices;               /* triangle_strip indices start at 1 (0 is top vertex), and we advance one stack down as we go along */

    for ( unsigned int j=0;  j<slices-1;  j++) {
        indices[idx++] = j + offset  ;
        indices[idx++] = n_vertices-1 ;
        indices[idx++] = j + offset + 1;
    }

    indices[idx++] = offset + slices - 1 ;
    indices[idx++] = n_vertices-1 ;
    indices[idx++] = offset ;


    return m ;
}

Mesh Mesh::makeCylinder(float radius, float height, size_t slices, size_t stacks, bool add_caps) {

    Mesh m ;

    auto &vertices = m.vertices() ;
    auto &normals = m.normals() ;
    auto &indices = m.indices() ;

    float z0,z1;

    const float zStep = height / std::max(stacks, (size_t)1) ;

    vector<float> sint, cost;
    makeCircleTable(sint, cost, slices);

    /* Cover the circular base with a triangle fan... */

    z0 = -height/2.0;
    z1 = z0 + zStep;

    for( unsigned int i=0 ; i<slices ; i++ ) {
        vertices.push_back({cost[i]*radius, sint[i]*radius, z0}) ;
        normals.push_back({cost[i], sint[i], 1.0}) ;
    }

    for( size_t j = 1 ;  j <= stacks; j++ ) {

        for( unsigned int i=0 ; i<slices ; i++ ) {
            vertices.push_back({cost[i]*radius, sint[i]*radius, z1}) ;
            normals.push_back({cost[i], sint[i], 1.0}) ;
        }

        for( unsigned int i=0 ; i<slices ; i++ ) {
            size_t pn = ( i == slices - 1 ) ? 0 : i+1 ;
            indices.push_back((j-1)*slices + i) ;
            indices.push_back((j-1)*slices + pn) ;
            indices.push_back((j)*slices + pn) ;

            indices.push_back((j-1)*slices + i) ;
            indices.push_back((j)*slices + pn) ;
            indices.push_back((j)*slices + i) ;
        }

        z1 += zStep;
    }

    size_t offset = vertices.size() ;

    if ( add_caps ) {
        vertices.push_back({0, 0, z0}) ;
        normals.push_back({0, 0, -1}) ;

        for( unsigned int i=0 ; i<slices ; i++ ) {
            vertices.push_back({cost[i]*radius, sint[i]*radius, z0}) ;
            normals.push_back({0, 0, -1}) ;
        }

        for( unsigned int i=0 ; i<slices ; i++ ) {
            indices.push_back(i+offset+1) ;
            indices.push_back(offset) ;
            indices.push_back(i == slices-1 ? offset + 1 : offset + i + 2) ;
        }

        offset = vertices.size() ;

        vertices.push_back({0.f, 0.f, height/2.0f}) ;
        normals.push_back({0, 0, 1}) ;

        for( unsigned int i=0 ; i<slices ; i++ ) {
            vertices.push_back({cost[i]*radius, sint[i]*radius, height/2.0f}) ;
            normals.push_back({0, 0, 1}) ;
        }

        for( unsigned int i=0 ; i<slices ; i++ ) {
            size_t pn = ( i == slices - 1 ) ? 0 : i+1 ;
            indices.push_back(offset + i) ;
            indices.push_back(offset + pn) ;
            indices.push_back(offset + slices) ;
        }
    }

    return m ;
}


aiMesh *Mesh::createAssimpMesh() const {

    aiMesh *mesh = new aiMesh();

    memset(mesh, 0, sizeof(aiMesh)) ;

    mesh->mMaterialIndex = 0 ;
    mesh->mVertices = new aiVector3D[ vertices_.size() ];
    mesh->mNumVertices = vertices_.size();

    uint i=0 ;
    for( const auto &v: vertices_ ) {
        mesh->mVertices[i++] = aiVector3D(v.x(), v.y(), v.z()) ;
    }

    if ( !normals_.empty() ) {
        mesh->mNormals = new aiVector3D[ normals_.size() ] ;
        i = 0 ;
        for( const auto &v: normals_ ) {
            mesh->mNormals[i++] = aiVector3D(v.x(), v.y(), v.z()) ;
        }
    }
    else
        mesh->mNormals = nullptr ;

    if ( !colors_.empty() ) {
        mesh->mColors[0] = new aiColor4D[ vertices_.size() ] ;
        i = 0 ;
        for( const auto &v: colors_ ) {
            mesh->mColors[0][i++] = aiColor4D(v.x(), v.y(), v.z(), 1.0) ;
        }
    }
    else
        mesh->mColors[0] = nullptr ;

    if ( !indices_.empty() ) {
        mesh->mFaces = new aiFace[ indices_.size() / 3 ];
        mesh->mNumFaces = indices_.size() / 3 ;

        uint k=0 ;
        for( uint i=0 ; i<indices_.size() ; i+=3 ) {
            aiFace& face = mesh->mFaces[ k++ ];
            face.mIndices = new unsigned int [3] ;
            face.mNumIndices = 3 ;

            face.mIndices[ 0 ] = indices_[i]  ;
            face.mIndices[ 1 ] = indices_[i+1];
            face.mIndices[ 2 ] = indices_[i+2];
        }

    } else {
        mesh->mFaces = nullptr ;
        mesh->mNumFaces = 0 ;
    }

    return mesh;
}

aiNode *Model3d::createAssimpNode(aiNode *parent, vector<aiMesh *> &meshes ) const {
    aiNode *node = new aiNode ;
    memset(node, 0, sizeof(aiNode)) ;

    if ( !mesh_.vertices().empty() ) {
        aiMesh *mesh = mesh_.createAssimpMesh() ;
        node->mMeshes = new uint [1] ;
        node->mMeshes[0] = meshes.size() ;
        node->mNumMeshes = 1 ;
        meshes.push_back(mesh) ;
    }

    if ( !children_.empty() ) {
        node->mNumChildren = children_.size() ;
        node->mChildren = new aiNode * [children_.size()] ;
        uint i=0 ;
        for( const auto &child: children_ ) {
            aiNode *cnode = child.createAssimpNode(node, meshes) ;
            node->mChildren[i++] = cnode ;
        }
    }

    node->mParent = parent ;

    aiMatrix4x4 m ;

    m.a1 = pose_(0, 0) ; m.a2 = pose_(0, 1) ; m.a3 = pose_(0, 2) ; m.a4 = pose_(0, 3) ;
    m.b1 = pose_(1, 0) ; m.b2 = pose_(1, 1) ; m.b3 = pose_(1, 2) ; m.b4 = pose_(1, 3) ;
    m.c1 = pose_(2, 0) ; m.c2 = pose_(2, 1) ; m.c3 = pose_(2, 2) ; m.c4 = pose_(2, 3) ;
    m.d1 = pose_(3, 0) ; m.d2 = pose_(3, 1) ; m.d3 = pose_(3, 2) ; m.d4 = pose_(3, 3) ;

    node->mTransformation = m ;


    return node ;
}

void Model3d::write(const std::string &fname)
{
    aiScene scene ;
    memset(&scene, 0, sizeof(aiScene)) ;

    vector<aiMesh *> meshes ;

    scene.mRootNode = createAssimpNode(nullptr, meshes) ;
    scene.mNumMeshes = meshes.size() ;
    scene.mMeshes = new aiMesh *[meshes.size()] ;

    scene.mMaterials = new aiMaterial*[1];
    scene.mNumMaterials = 1;
    scene.mMaterials[0] = new aiMaterial();

    for( uint i=0 ; i<meshes.size() ; i++ ) {
        scene.mMeshes[i] = meshes[i] ;
    }

    string fmt ;
    if ( cvx::endsWith(fname, ".dae") ) {
        fmt = "collada" ;
    } else if ( cvx::endsWith(fname, ".stl") ) {
        fmt = "stl" ;
    } else if ( cvx::endsWith(fname, ".ply") ) {
        fmt = "ply" ;
    } else if ( cvx::endsWith(fname, ".obj") ) {
        fmt = "obj" ;
    }

    Assimp::Exporter exporter;
    Assimp::ExportProperties *properties = new Assimp::ExportProperties;
    //    properties->SetPropertyBool(AI_CONFIG_EXPORT_POINT_CLOUDS, true);
    exporter.Export(&scene, fmt, fname, 0, properties ) ;
}
