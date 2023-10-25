#include <htrac/pose/model_fit.hpp>

#include <htrac/pose/human_detector.hpp>
#include <htrac/pose/dataset.hpp>

#include <cvx/misc/path.hpp>
#include <cvx/math/rng.hpp>
#include <cvx/imgproc/rgbd.hpp>
#include <cvx/misc/format.hpp>

#include <htrac/util/io_util.hpp>
#include <htrac/util/pcl_util.hpp>
#include <htrac/model/skeleton.hpp>
#include <htrac/model/skinned_mesh.hpp>
#include <htrac/util/mhx2_importer.hpp>
#include <htrac/pose/keypoint_distance_field.hpp>

#include <xviz/scene/geometry.hpp>
#include <xviz/scene/node_helpers.hpp>
#include <xviz/gui/offscreen.hpp>
#include <xviz/scene/renderer.hpp>
#include <xviz/scene/light.hpp>

#include <QApplication>

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

int main(int argc, char *argv[]) {
    QApplication qapp(argc, argv) ;
    Config cfg ;

    HumanModelFit::Parameters p ;
 //   p.lambda_mi_ = 0;
 //   p.lambda_im_ = 0 ;
    HumanModelFit h(p) ;


    std::map<std::string, std::pair<float, Eigen::Vector3f>> kpts ;
  //  kpts.emplace("LeftHand", make_pair(1.0f, Vector3f(0, -0.3, -2.5))) ;

    ITOP dataset("/home/malasiot/source/human_tracking/data/datasets/ITOP/train") ;
    auto cam = dataset.getCamera() ;

    SkinnedMesh mesh ;
    mesh.load("models/human-cmu-low-poly.mhx2") ;


    HumanDetector detector ;

    Plane plane(Vector3f{0, 1, 0}, 1.2) ;
    detector.setGroundPlane(plane);

    Polytope *ocs = new Polytope{{
            Plane{{0, 0, 1}, 0.05}, // bottom
            Plane{{0, 0, -1}, 2.0},  // top
            Plane{{0, -1, 0}, 3.5},  // back
            Plane{{1, 0, 0}, 1.0}, // left
            Plane{{-1, 0, 0}, 0.8} // rigtht
}};
    detector.setOcclusionSpace(ocs) ;

    Pose pose(&h.skeleton()) ;

    pose.setGlobalTranslation(Vector3f{0, -1.3, -3.0});
    pose.setGlobalRotation(Quaternionf{1, 0, 0.3, 0}) ;



    uint start_frame = 130, stop_frame = 500 ;
    for( uint i=start_frame ; i<stop_frame ; i++ ) {
        cv::Mat img = dataset.getDepthImage(i) ;
        cv::imwrite("/tmp/dim.png", cvx::depthViz(img)) ;

        cv::Rect box ;
        cv::Mat mask ;

        detector.detect(img, cam, box, mask);

        cv::Mat smask(img.size(), CV_8UC1, cv::Scalar(255)) ;
        vector<Vector3f> cloud = depthToPointCloud(img, cam, mask, 1) ;



       save_point_cloud(cloud, "/tmp/cloud.obj");



 //  Pose result = h.fit(cloud, mask, cam, pose) ;

        KeyPoints kpts2d ;
    //    kpts2d.emplace("RightHand", make_pair(1.0f, Vector2f(136, 55))) ;

        KeyPointListDistanceField *df = new KeyPointListDistanceField(kpts2d) ;

        Pose result = h.fit(img, mask, cam, df, pose) ;



        xviz::Image im = renderMesh(2 * img.cols, 2 * img.rows, cloud, mesh, result) ;

        im.saveToPNG(cvx::format("/tmp/im_{:03d}.png", i-start_frame)) ;
        h.skeleton().toMesh(result).write("/tmp/result.obj") ;

        pose = result ;
    }
}
