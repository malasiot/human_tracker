#include <htrac/model/skinned_mesh.hpp>
#include <htrac/util/mhx2_importer.hpp>

#include <set>
#include <iostream>
#include <fstream>

using namespace std ;
using namespace Eigen ;

static float saasin(float fac)
{
    if      (fac <= -1.0f) return (float)-M_PI / 2.0f;
    else if (fac >=  1.0f) return (float) M_PI / 2.0f;
    else return asinf(fac);
}

float angle_normalized_v3v3(const Vector3f &v1, const Vector3f &v2)
{
    /* this is the same as acos(dot_v3v3(v1, v2)), but more accurate */
    if ( v1.dot(v2) < 0 ) {

        Vector3f vec(-v2) ;
        return (float)M_PI - 2.0f * (float)saasin((vec -v1).norm() / 2.0f);
    }
    else
        return 2.0f * (float)saasin((v2-v1).norm() / 2.0f);
}

static Vector3f normal_triangle(const Vector3f &v1, const Vector3f &v2, const Vector3f &v3)
{
    Vector3f n1, n2 ;

    n1 = v1 - v2 ;
    n2 = v1 - v3 ;
    return  n1.cross(n2).normalized() ;

}

static Vector3f normal_quad(const Vector3f &v1, const Vector3f &v2, const Vector3f &v3, const Vector3f &v4)
{
    Vector3f n1, n2 ;

    n1 = v1 - v3 ;
    n2 = v2 - v4 ;

    return  n1.cross(n2).normalized() ;
}

/* Newell's Method */
static Vector3f calc_ngon_normal(const vector<Vector3f> &vtx)
{
    const int nverts = vtx.size() ;

    Vector3f normal = Vector3f::Zero(), v_curr, v_prev = vtx.back() ;

    for (int i = 0; i < nverts; i++) {
        v_curr = vtx[i] ;
        normal += (v_prev - v_curr).cross(v_prev + v_curr) ;
        v_prev = v_curr;
    }

    if ( normal.norm() == 0.0f ) return Vector3f(0, 0, 1.0f);
    else  return normal.normalized() ;
}

static Vector3f calc_face_normal(vector<Vector3f> &vtx)
{
    if ( vtx.size() > 4 ) return calc_ngon_normal(vtx);
    else if ( vtx.size() == 3 ) return normal_triangle(vtx[0], vtx[1], vtx[2]) ;
    else if ( vtx.size() == 4 ) return normal_quad(vtx[0], vtx[1], vtx[2], vtx[3]) ;
    else return Vector3f(0, 0, 1.0) ;
}

void compute_normals(const vector<Vector3f> &vertices, const vector<uint> &indices, vector<Vector3f> &vtx_normals)
{
    vtx_normals.resize(vertices.size()) ;
    for( int i=0 ; i<vertices.size() ; i++ ) vtx_normals[i] = Vector3f::Zero() ;

    for( int i=0 ; i<indices.size() ; i+=3 ) {
        uint idx0 = indices[i] ;
        uint idx1 = indices[i+1] ;
        uint idx2 = indices[i+2] ;
        Vector3f n = normal_triangle(vertices[idx0], vertices[idx1], vertices[idx2]) ;

        vtx_normals[idx0] += n ;
        vtx_normals[idx1] += n ;
        vtx_normals[idx2] += n ;
    }

    for( int i=0 ; i<vertices.size() ; i++ ) vtx_normals[i].normalize() ;

}

SkinnedMesh::VertexBoneData::VertexBoneData() {
    reset();
}

void SkinnedMesh::VertexBoneData::reset() {
    memset(id_, -1, sizeof(uint)*MAX_WEIGHTS_PER_VERTEX) ;
    memset(weight_, 0, sizeof(float)*MAX_WEIGHTS_PER_VERTEX) ;
}

void SkinnedMesh::VertexBoneData::addBoneData(uint boneID, float w) {

    for (uint i = 0 ; i < MAX_WEIGHTS_PER_VERTEX ; i++) {
        if (id_[i] < 0 ) {
            id_[i]     = boneID;
            weight_[i] = w;
            return;
        }
    }
}

void SkinnedMesh::VertexBoneData::normalize()
{
    float w = 0.0 ;

    for(int i=0 ; i<MAX_WEIGHTS_PER_VERTEX ; i++) {
        if ( id_[i] < 0 ) break ;
        w += weight_[i] ;
    }

    if ( w == 0.0 ) return ;

    for(int i=0 ; i<MAX_WEIGHTS_PER_VERTEX ; i++) {
        if ( id_[i] < 0 ) break ;
        weight_[i] /= w  ;
    }
}

vector<Matrix4f> SkinnedMesh::getSkinningTransforms(const Pose &p) const {
    vector<Matrix4f> skinning_trs ;
    skeleton_.computeBoneTransforms(p, skinning_trs) ;

    for( size_t i=0 ; i<skinning_trs.size() ; i++ ) {
         skinning_trs[i] *= skeleton_.getBone(i).offset_.inverse();
    }

    return skinning_trs ;
}

void SkinnedMesh::load(const std::string &fname, bool zUp) {
    Mhx2Importer importer ;
    if ( importer.load(fname, zUp) )
        load(importer.getModel() ) ;
}

void SkinnedMesh::getTransformedVertices(const Pose &p, std::vector<Eigen::Vector3f> &mpos, std::vector<Eigen::Vector3f> &mnorm) const {
   vector<Matrix4f> skinning_trs = getSkinningTransforms(p);

    for(int i=0 ; i<positions_.size() ; i++) {
        const Vector3f &pos = positions_[i] ;
        Vector3f normal ;

        if ( !normals_.empty() ) normal = normals_[i] ;

        const VertexBoneData &bdata = bones_[i] ;

        Matrix4f boneTransform ;

        float w = 0.0 ;

        for( int j=0 ; j<MAX_WEIGHTS_PER_VERTEX ; j++)
        {
            int idx = bdata.id_[j] ;
            if ( idx < 0 ) break ;

            if ( j == 0 ) boneTransform = skinning_trs[idx] * bdata.weight_[j] ;
            else boneTransform += skinning_trs[idx] * (double)bdata.weight_[j] ;
        }

        Vector4f p = boneTransform * Vector4f(pos.x(), pos.y(), pos.z(), 1.0) ;
        mpos.push_back(Vector3f(p.x(), p.y(), p.z())) ;

        if ( !normals_.empty() )
        {
            Vector4f n = boneTransform * Vector4f(normal.x(), normal.y(), normal.z(), 0.0) ;
            Vector3f nrm(Vector3f(n.x(), n.y(), n.z())) ;
            mnorm.push_back(nrm.normalized()) ;
        }
    }

}

Model3d SkinnedMesh::toMesh(const Pose &p) const {
    vector<Vector3f> mpos, mnorm ;
    getTransformedVertices(p, mpos, mnorm) ;

    Model3d model ;

    Mesh mesh ;

    mesh.vertices() = mpos ;
    mesh.normals() = mnorm ;
    mesh.indices() = indices_ ;

    model.setGeometry(mesh) ;
    return model ;
}

void SkinnedMesh::load(const MHX2Model &model) {

    int v_offset = 0 ;

    skeleton_.load(model) ;

    for( const auto &gp: model.geometries_ ){

        const auto &mesh = gp.second.mesh_ ;

        for ( const auto &v: mesh.vertices_ )
            positions_.push_back(v + gp.second.offset_) ;

        vector<int> indices ;
        vector<Vector2f> tcoords ;

        for( const auto &f: mesh.faces_ ) {
            for(int k=0 ; k<f.num_vertices_ ; k++)  {
                indices.push_back(f.indices_[k] + v_offset) ;
                tcoords.push_back(f.tex_coords_[k]);
            }

            indices.push_back(-1) ;
        }

        vector<int> face_vertices ;
        vector<Vector2f> face_tcoords ;

        int c = 0 ;

        for( int i=0 ; i<indices.size() ; i++ )
        {
            int idx = indices[i] ;

            if ( idx == -1 )
            {
                for(int j=1 ; j<face_vertices.size() - 1 ; j++)
                {
                    indices_.push_back(face_vertices[0]) ;
                    indices_.push_back(face_vertices[j]) ;
                    indices_.push_back(face_vertices[j+1]) ;

                    tex_coords_.push_back(face_tcoords[0]) ;
                    tex_coords_.push_back(face_tcoords[j]) ;
                    tex_coords_.push_back(face_tcoords[j+1]) ;
                }

                face_vertices.clear() ;
                face_tcoords.clear() ;

            }
            else {
                face_vertices.push_back(idx) ;
                face_tcoords.push_back(tcoords[c++]) ;
            }

        }


        // create vertex groups

        vector<VertexBoneData> bones ;
        bones.resize(mesh.vertices_.size()) ;

        for( const auto &gp: mesh.groups_ )
        {
            const string &name = gp.first ;
            const MHX2VertexGroup &group = gp.second ;
            uint32_t bidx = skeleton_.getBoneIndex(name) ;

            for( size_t i=0 ; i<group.idxs_.size() ; i++ ) {
                int idx = group.idxs_[i];

                if ( idx < 0 ) continue ;

                SkinnedMesh::VertexBoneData &data = bones[idx] ;

                data.addBoneData(bidx, group.weights_[i]) ;

                data.normalize() ;
            }
        }

        std::copy(bones.begin(), bones.end(), std::back_inserter(bones_)) ;

        v_offset += mesh.vertices_.size() ;
    }

     compute_normals(positions_, indices_, normals_);

}

