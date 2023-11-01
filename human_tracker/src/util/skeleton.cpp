#include <vector>
#include <fstream>
#include <iostream>

#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

using namespace std ;
using namespace Eigen ;

static void make_circle_table(vector<float> &sint, vector<float> &cost, int n, bool half_circle = false) {

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

static void create_cylinder_mesh(float radius, float height, size_t slices, size_t stacks, bool add_caps,
                                 vector<Vector3f> &vertices, vector<Vector3f> &normals,
                                 vector<uint32_t> &vtx_indices, vector<uint32_t> &nrm_indices) {

    float z0,z1;

    const float zStep = height / std::max(stacks, (size_t)1) ;

    vector<float> sint, cost;
    make_circle_table(sint, cost, slices);

    /* Cover the circular base with a triangle fan... */

    z0 = -height/2.0;
    z1 = z0 + zStep;


    unsigned int co = 0 ;

    if ( add_caps ) {
        vertices.push_back({0, 0, z0}) ;
        normals.push_back({0, 0, -1}) ;
        co = 1 ;
    }

    for( unsigned int i=0 ; i<slices ; i++ ) {
        vertices.push_back({cost[i]*radius, sint[i]*radius, z0}) ;
    }

    if ( add_caps ) {
        for( unsigned int i=0 ; i<slices ; i++ ) {

            vtx_indices.push_back(i+1) ;
            vtx_indices.push_back(0) ;
            vtx_indices.push_back(i == slices-1 ? 1 : i+2) ;

            nrm_indices.push_back(0) ;
            nrm_indices.push_back(0) ;
            nrm_indices.push_back(0) ;
        }
    }


    // normals shared by all side vertices

    for( unsigned int i=0 ; i<slices ; i++ ) {
        normals.push_back({cost[i], sint[i], 1.0}) ;
    }

    for( size_t j = 1 ;  j <= stacks; j++ ) {

        for( unsigned int i=0 ; i<slices ; i++ ) {
            vertices.push_back({cost[i]*radius, sint[i]*radius, z1}) ;
        }

        for( unsigned int i=0 ; i<slices ; i++ ) {
            size_t pn = ( i == slices - 1 ) ? 0 : i+1 ;
            vtx_indices.push_back((j-1)*slices + i + co) ;
            vtx_indices.push_back((j-1)*slices + pn + co) ;
            vtx_indices.push_back((j)*slices + pn + co) ;

            vtx_indices.push_back((j-1)*slices + i + co) ;
            vtx_indices.push_back((j)*slices + pn + co) ;
            vtx_indices.push_back((j)*slices + i + co) ;

            nrm_indices.push_back(i + co) ;
            nrm_indices.push_back(pn + co) ;
            nrm_indices.push_back(pn + co) ;

            nrm_indices.push_back(i + co) ;
            nrm_indices.push_back(pn + co) ;
            nrm_indices.push_back(i + co) ;
        }

        z1 += zStep;
    }

    // link apex with last stack

    size_t offset = (stacks)*slices + co;

    if ( add_caps ) {
        vertices.push_back({0.f, 0.f, height/2.0f}) ;
        normals.push_back({0, 0, 1}) ;

        for( unsigned int i=0 ; i<slices ; i++ ) {
            size_t pn = ( i == slices - 1 ) ? 0 : i+1 ;
            vtx_indices.push_back(offset + i) ;
            vtx_indices.push_back(offset + pn) ;
            vtx_indices.push_back(offset + slices) ;

            nrm_indices.push_back(slices+1) ;
            nrm_indices.push_back(slices+1) ;
            nrm_indices.push_back(slices+1) ;
        }
    }
}


void get_bone_geometry(const std::vector<std::pair<int, int>> &skeleton, const std::vector<Vector3f> &joints,
                       std::vector<std::pair<Vector3f, Vector3f>> &coords) {

    for( const auto &bone : skeleton ) {
        int idx0 = bone.first ;
        int idx1 = bone.second ;
        coords.emplace_back(make_pair(joints[idx0], joints[idx1])) ;
    }
}

void skeleton_to_obj(const std::vector<std::pair<int, int>> &skeleton, const std::vector<Vector3f> &joints,
                     const std::string &outpath) {

    size_t voffset = 1, noffset = 1;

    ofstream strm(outpath) ;

    std::vector<std::pair<Vector3f, Vector3f>> coords ;

    get_bone_geometry(skeleton, joints, coords) ;

    for( const auto &jp : coords ) {
        const auto &coords0 = jp.first ;
        const auto &coords1 = jp.second ;

        vector<Vector3f> vertices, normals ;
        vector<uint32_t> vtx_indices, nrm_indices ;

        float len = (coords1-coords0).norm() ;
        create_cylinder_mesh(0.01, len, 16, 8, true,
                             vertices, normals, vtx_indices, nrm_indices ) ;

        Vector3f a{ 0, 0, 1 };
        Vector3f d = (coords1 - coords0).normalized() ;

        // rotate cylinder to align with bone axis
        Matrix3f r(Quaternionf::FromTwoVectors(a, d)) ;
        // translate to bone center
        Vector3f t = coords0 + d * len/2 ;

        for( size_t i = 0 ; i<vertices.size() ; i++ ) {
            const Vector3f &v = vertices[i] ;
            strm << "v " << (r * v + t).adjoint() << endl ;
        }

        for( size_t i = 0 ; i<normals.size() ; i++ ) {
            const Vector3f &n = normals[i] ;
            strm << "vn " << (r * n).adjoint() << endl ;
        }

        for( size_t i=0 ; i<vtx_indices.size() ; ) {
            uint32_t vidx0 = vtx_indices[i] ;
            uint32_t nidx0 = nrm_indices[i++] ;

            uint32_t vidx1 = vtx_indices[i] ;
            uint32_t nidx1 = nrm_indices[i++] ;

            uint32_t vidx2 = vtx_indices[i] ;
            uint32_t nidx2 = nrm_indices[i++] ;

            strm << "f "
                << vidx0 + voffset << "//" << nidx0 + noffset << ' '
                << vidx1 + voffset << "//" << nidx1 + noffset << ' '
                << vidx2 + voffset << "//" << nidx2 + noffset << endl ;
        }

        voffset += vertices.size() ;
        noffset += normals.size() ;
    }

    strm.close() ;
}


void draw_skeleton(const cv::Mat &im, const std::vector<std::pair<int, int>> &skeleton, const std::vector<Eigen::Vector2f> &joints,
                   const cv::Vec3b &clr) {
    for( const auto &l: skeleton ) {
        const auto &j1 = joints[l.first] ;
        const auto &j2 = joints[l.second] ;
        cv::line(im, cv::Point(j1.x(), j1.y()), cv::Point(j2.x(), j2.y()), cv::Scalar(clr[0], clr[1], clr[2])) ;

    }
}


