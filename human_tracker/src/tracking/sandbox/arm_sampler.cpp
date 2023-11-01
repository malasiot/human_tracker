#include <htrac/util/mhx2_importer.hpp>
#include <htrac/util/pose_database.hpp>
#include <htrac/model/skeleton.hpp>

#include <cvx/misc/arg_parser.hpp>
#include <cvx/misc/progress_stream.hpp>

#include <xviz/scene/node.hpp>

#include <iostream>
#include <random>
#include <Eigen/Geometry>
#include <fstream>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

using namespace cvx ;
using namespace std ;
using namespace Eigen ;
using namespace xviz ;

static void spherical(const Vector3f &p, float &theta, float &phi, float &r) {
    r = p.norm() ;
    theta = atan2(p.y(), p.x()) ;
    phi = acos(p.z()/r) ;
}

int main(int argc, char *argv[]) {
    PoseDatabase db ;

    db.connect("/home/malasiot/source/human_tracking/data/poses.sqlite") ;

    Mhx2Importer importer ;
    importer.loadSkeleton("/home/malasiot/source/human_tracking/data/models/human-cmu-low-poly.mhx2");

    Skeleton sk ;
    sk.fromMH(importer.getModel()) ;

    ofstream strm("/tmp/shoulder.txt") ;

    uint ncoords = 10000 ;
    float* coords = new float[2 * ncoords * 2];
    auto cursor = db.getReadCursor() ;
    size_t c = 0, idx = 0 ;
    while ( cursor.next() && c < ncoords ) {
        auto p = cursor.pose() ;

        map<string, Matrix4f> trs ;
        sk.computeBoneTransforms(p, trs);

        Vector3f ls = (trs["LeftArm"] * Vector4f(0, 0, 0, 1)).head<3>();
        Vector3f rs = (trs["RightArm"] * Vector4f(0, 0, 0, 1)).head<3>();

        Vector3f le = (trs["LeftForeArm"] * Vector4f(0, 0, 0, 1)).head<3>();
        Vector3f re = (trs["RightForeArm"] * Vector4f(0, 0, 0, 1)).head<3>();

        Vector3f lh = (trs["LeftHand"] * Vector4f(0, 0, 0, 1)).head<3>();
        Vector3f rh = (trs["RightHand"] * Vector4f(0, 0, 0, 1)).head<3>();

        Vector3f x = (ls - rs).normalized() ;
        Vector3f y { 0, -x.z(), x.y() } ;
        y.normalize() ;
        Vector3f z = x.cross(y).normalized() ;

        Matrix3f R ;
        R.row(0) = x ; R.row(1) = z ; R.row(2) = -y ;

        le = R * (le - ls) ;
        re = R * (re - rs) ; re.x() = -re.x() ;

        lh = R * (lh - ls) ;
        rh = R * (rh - rs) ; rh.x() = -rh.x() ;

        float ltheta, lphi, lr ;
        spherical(le, ltheta, lphi, lr) ;

        float rtheta, rphi, rr ;
        spherical(re, rtheta, rphi, rr) ;

        coords[idx++] = ltheta ; coords[idx++] = lphi ;
        coords[idx++] = rtheta ; coords[idx++] = rphi ;

   //     strm << ltheta << ' ' << lphi << endl ;
    //    strm << rtheta << ' ' << rphi << endl ;

        // -1.8 - 1.8, 0.12 - 3.05
        c++ ;

/*
        map<string, Vector3f> joints ;

        ofstream strm("/tmp/test.obj");
        for( const auto &bp: trs ) {
            Vector3f coords = (bp.second * Vector4f(0, 0, 0, 1)).head<3>() ;
            coords =  R * ( coords - ls ) ;
            joints.emplace(bp.first, coords) ;
            strm << "v " << coords.adjoint() << endl ;
        }
        strm.close();
        */
    }

    uint nc = 100 ;

    std::random_device rd;
    default_random_engine eng{rd()};

    faiss::ClusteringParameters params ;
    params.niter = 5 ;
    params.nredo = 1 ;
    params.verbose = true ;

    params.seed = eng() ;

    faiss::Clustering kmeans(2, nc, params) ;

    // perform k-means
    faiss::IndexFlatL2 index(2);
    kmeans.train(2 * ncoords, coords, index) ;


    for( uint i=0 ; i<nc ; i++ ) {
        strm << kmeans.centroids[2*i] << ' ' << kmeans.centroids[2*i+1] << endl;
    }

}
