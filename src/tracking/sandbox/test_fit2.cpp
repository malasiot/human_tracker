#include <htrac/pose/model_fit.hpp>

#include <htrac/pose/human_detector.hpp>
#include <htrac/pose/dataset.hpp>

#include <cvx/misc/path.hpp>
#include <cvx/math/rng.hpp>
#include <cvx/imgproc/rgbd.hpp>
#include <cvx/misc/format.hpp>
#include <cvx/misc/json_reader.hpp>
#include <cvx/pcl/align.hpp>

#include <htrac/util/io_util.hpp>
#include <htrac/util/pcl_util.hpp>
#include <htrac/model/skeleton.hpp>
#include <htrac/model/skinned_mesh.hpp>
#include <htrac/util/mhx2_importer.hpp>
#include <htrac/pose/keypoint_distance_field.hpp>
#include <htrac/pose/keypoint_detector.hpp>
#include <htrac/pose/pose_from_keypoints.hpp>

#include <xviz/scene/geometry.hpp>
#include <xviz/scene/node_helpers.hpp>
#include <xviz/gui/offscreen.hpp>
#include <xviz/scene/renderer.hpp>
#include <xviz/scene/light.hpp>

#include <QApplication>

#include <fstream>

using namespace std ;
using namespace Eigen ;

vector<Vector3f> pallete = {
    {0.0, 0.0, 0.5},
    {0.0, 0.0, 1.0},
    {0.0, 0.5, 1.0},
    {0.0, 1.0, 1.0},
    {0.5, 1.0, 0.5},
    {1.0, 1.0, 0.0},
    {1.0, 0.5, 0.0},
    {1.0, 0.0, 0.0},
    {0.5, 0.0, 0.0}
};

Vector3f clrmap(float x) {
    x = x * pallete.size() ;
    uint i0 = floor(x) ;
    uint i1 = i0 + 1 ;
    float h = x - i0 ;
    return pallete[i0] * ( 1.f - h ) + pallete[i1] * h ;
}

vector<Vector3f> cloudColors(const Vector3f &c, const vector<Vector3f> &coords) {
    float minv = std::numeric_limits<float>::max(), maxv = -std::numeric_limits<float>::min() ;
    for( const auto &v: coords ) {
        float d = ( v - c ).norm() ;
        minv = std::min(d, minv) ;
        maxv = std::max(d, maxv) ;
    }

    vector<Vector3f> clrs ;
    for( const auto &v: coords ) {
        float d = ( v - c ).norm() ;
        clrs.emplace_back(clrmap((d - minv)/(maxv - minv))) ;
    }

    return clrs;

}

xviz::Image renderMesh(uint width, uint height, const std::vector<Vector3f> &coords, const SkinnedMesh &mesh, const Pose &pose) {
    using namespace xviz ;
    NodePtr scene(new Node) ;

    GeometryPtr cloud_geom(new Geometry(xviz::Geometry::makePointCloud(coords, cloudColors({ 0, 0, 0}, coords)))) ;
    NodePtr cloud_node(new Node) ;

    MaterialPtr cloud_mat(new PerVertexColorMaterial()) ;

    cloud_node->addDrawable(cloud_geom, cloud_mat);

    scene->addChild(cloud_node);

    NodePtr mesh_node(new Node) ;
    vector<Vector3f> mpos, mnorm ;
    mesh.getTransformedVertices(pose, mpos, mnorm) ;

    GeometryPtr mesh_geom(new Geometry(Geometry::Triangles)) ;
    mesh_geom->vertices() = mpos ;
    mesh_geom->normals() = mnorm ;
    mesh_geom->indices() = mesh.indices_ ;
    MaterialPtr mesh_mat(new PhongMaterial(Vector3f{0.5, 0.5, 0.5})) ;
    mesh_node->addDrawable(mesh_geom, mesh_mat) ;
    scene->addChild(mesh_node) ;

    DirectionalLight *dl = new DirectionalLight(Vector3f(1, 1, 1)) ;
    dl->setDiffuseColor(Vector3f(1, 1, 1)) ;
    scene->setLight(LightPtr(dl)) ;

    OffscreenSurface os(QSize(width, height));

    // create a camera

    PerspectiveCamera *pcam = new PerspectiveCamera(1, // aspect ratio
                                                    50*M_PI/180,   // fov
                                                    0.01,        // zmin
                                                    10           // zmax
                                                    ) ;

    CameraPtr cam(pcam) ;

    cam->setBgColor({1, 1, 1, 1});

    // position camera to look at the center of the object

    //  pcam->viewSphere(c, r) ;
    pcam->lookAt({0, 0, 0}, {0, 0, -3}, {0, 1, 0}) ;

    // set camera viewpot

    pcam->setViewport(width, height)  ;

    Renderer rdr ;

    rdr.render(scene, cam) ;

    auto im = os.getImage() ;

    return im ;

}

static KeyPoints load_keypoints(const std::string &fpath) {

    KeyPoints res ;

    ifstream strm(fpath) ;

    try {
        cvx::JSONReader json(strm) ;

        json.beginObject() ;
        while ( json.hasNext() ) {
            string joint_name = json.nextName() ;
            Vector2f coords ;
            float weight ;
            json.beginObject() ;
            while ( json.hasNext() ) {
                string field = json.nextName() ;

                if ( field == "coords" ) {
                    json.beginArray() ;
                    coords.x() = json.nextDouble() ;
                    coords.y() = json.nextDouble() ;
                    json.endArray() ;
                } else if ( field == "weight" ) {
                    weight = json.nextDouble() ;
                }
            }

            res.emplace(joint_name, make_pair(coords, weight)) ;

            json.endObject() ;
        }
        json.endObject() ;
        return res ;
    } catch ( const cvx::JSONParseException &e ) {
        cout << e.what() << endl ;
        return KeyPoints() ;
    }
}




void segmentValidPoints(const cv::Mat &depth, const cvx::PinholeCamera &model, const OcclusionSpace *ocs, cv::Mat &mask)
{
    uint w = depth.cols, h = depth.rows;
    cv::Mat_<ushort> depth_(depth) ;
    mask = cv::Mat(h, w, CV_8UC1, cv::Scalar(0)) ;
    cv::Mat_<uchar> mask_(mask) ;
    std::vector<Vector3f> ipts ;

    for(int i=0 ; i<h ; i++)
        for(int j=0 ; j<w ; j++)  {

            ushort val = depth_[i][j] ;

            if ( val == 0 ) continue ;

            Vector3f p = model.backProject(j, i, val/1000.0) ;
            p.y() =-p.y() ;
            p.z() =-p.z() ;

            if ( !ocs->occluded(p) ) {
                ipts.push_back(p) ;
                mask_[i][j] = 255 ;
            }
        }

    save_point_cloud(ipts, "/tmp/cloud2.obj");

}


int main(int argc, char *argv[]) {
    QApplication qapp(argc, argv) ;
    Config cfg ;

    HumanModelFit::Parameters p ;
    p.lambda_mi_ = 0.1 ;
    //   p.lambda_mi_ = 0;
    //   p.lambda_im_ = 0 ;
    HumanModelFit h(p) ;


    //cvx::PinholeCamera cam(907.4, 906.74, 633.72, 380.77, cv::Size(1280, 720)) ;
    cvx::PinholeCamera cam(604.9, 604.5, 315.8, 253.8, cv::Size(640, 480)) ;

    SkinnedMesh mesh ;
    mesh.load("models/human-cmu.mhx2") ;


    Polytope *ocs = new Polytope{{
        Plane{{0, -1, 0}, 0.8}, // top
        Plane{{0, 1, 0}, 1.2}, // bottom
        Plane{{0, 0, 1}, 2.8}, // back
        Plane{{0, 0, -1}, 2.5}, // front
        Plane{{1, 0, 0}, 0.5}, // left */
        Plane{{-1, 0, 0}, 0.5} // rigtht

    }};

    const Skeleton &skel = h.skeleton() ;


    PoseFromKeyPoints pose_kps(skel, 0.8) ;

    h.setClippingPlanes({
        Plane{{0, -1, 0}, 0.8}, // top
        Plane{{0, 1, 0}, 1.2}, // bottom
        Plane{{0, 0, 1}, 2.8}, // back
        Plane{{0, 0, -1}, 2.5}, // front
        Plane{{1, 0, 0}, 0.5}, // left
                         Plane{{-1, 0, 0}, 0.5}} // rigtht
);

    Pose pose(&skel) ;

    uint start_frame = 305, stop_frame = 500 ;
    for( uint i=start_frame ; i<stop_frame ; i++ ) {

        string depth_file_name = cvx::format("grab_{:03d}_d.png", i) ;
        cv::Mat img = cv::imread("/tmp/" + depth_file_name, -1) ;
        cv::imwrite("/tmp/dim.png", cvx::depthViz(img)) ;

        KeyPoints kpts = load_keypoints("/tmp/" + cvx::format("kpts_{:03d}.json", i)) ;

        if ( !pose_kps.estimate(kpts, cam, img, pose) ) {
            cerr << "pose initialization failed" << endl ;
            continue ;
        }

        skel.toMesh(pose).write("/tmp/init.obj") ;

        cv::Mat mask ;
        segmentValidPoints(img, cam, ocs, mask);
//       cv::Mat smask(img.size(), CV_8UC1, cv::Scalar(255)) ;
//       vector<Vector3f> cloud = depthToPointCloud(img, cam, smask, 1) ;

      //  cv::imwrite("/tmp/mask.png", mask) ;


      //  save_point_cloud(cloud, "/tmp/cloud.obj");


    //    cv::Mat mask ;

        KeyPointListDistanceField *df = new KeyPointListDistanceField(kpts) ;

        Pose result = h.fit(img, cv::Mat(), cam, nullptr, pose) ;

skel.toMesh(result).write("/tmp/result.obj") ;

  //      xviz::Image im = renderMesh(img.cols, img.rows, cloud, mesh, result) ;

  //      im.saveToPNG(cvx::format("/tmp/im_{:03d}.png", i-start_frame)) ;
        h.skeleton().toMesh(result).write("/tmp/result.obj") ;

        pose = result ;
    }
}
