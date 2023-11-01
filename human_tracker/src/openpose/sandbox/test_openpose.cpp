#include <htrac/pose/keypoint_detector_openpose.hpp>

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

#include <cvx/misc/format.hpp>
#include <cvx/misc/json_writer.hpp>

#include <fstream>


const std::string image_folder = "/tmp/" ;
const std::string model_folder =  "/home/malasiot/local/openpose/models/" ;

static const int start_frame = 0 ;
static const int end_frame = 520 ;

using namespace std ;
using namespace cvx ;

static void drawKeyPoints(cv::Mat &im, const KeyPoints &kpts) {
    for( const auto &kp: kpts ) {
        cout << kp.first << endl ;
        cout << kp.second.first.adjoint() << " (" << kp.second.second << ")" << endl ;

        const auto &pos = kp.second.first ;
        cv::drawMarker(im, cv::Point(pos.x(), pos.y()), cv::Scalar(255, 0, 0), cv::MARKER_DIAMOND) ;
    }
}

static void saveKeyPoints(const KeyPoints &kpts, const std::string &out_path) {
    ofstream strm(out_path) ;
    JSONWriter w(strm) ;

    w.beginObject() ;
    for( const auto &kp: kpts ) {
        w.name(kp.first) ;
        w.beginObject() ;
        w.name("coords") ;
        w.beginArray() ;
        const auto &coords = kp.second.first ;
        w.floatValue(coords.x()) ;
        w.floatValue(coords.y()) ;
        w.endArray() ;
        w.name("weight") ;
        w.floatValue(kp.second.second) ;
        w.endObject() ;
    }
    w.endObject() ;
}

int main(int argc, char *argv[])
{
    KeyPointDetectorOpenPose::Parameters params ;
    params.data_folder_ = model_folder ;

    KeyPointDetectorOpenPose ops(params) ;

    ops.init() ;

    for (uint frame = start_frame ; frame < end_frame ; frame ++ ) {
        std::string img_fname = cvx::format("grab_{:03d}_c.png", frame) ;
        std::string kpts_fname = cvx::format("kpts_{:03d}.json", frame) ;
        cv::Mat im = cv::imread(image_folder + '/' + img_fname) ;
        auto kpts = ops.findKeyPoints(im) ;

        cout << frame << endl ;
        saveKeyPoints(kpts, image_folder + '/' + kpts_fname) ;

       //drawKeyPoints(im, kpts) ;

    }

}
