#include <htrac/pose/keypoint_detector.hpp>
#include <opencv2/opencv.hpp>

using namespace Eigen ;

static const std::vector<std::string> knames = {
    "LeftFoot",         /* 0 */
    "RightFoot",        /* 1 */
    "LeftForeArm",      /* 2 */
    "RightForeArm",     /* 3 */
    "eye.L",            /* 4 */
    "eye.R",            /* 5 */
    "LeftUpLeg",        /* 6 */
    "RightUpLeg",       /* 7 */
    "LeftLeg",          /* 8 */
    "RightLeg",         /* 9 */
    "LeftArm",          /* 10 */
    "RightArm",         /* 11 */
    "LeftHand",         /* 12 */
    "RightHand",        /* 13 */
    "Neck",             /* 14 */
};

static const std::vector<cv::Vec3b> colors = {
    { 255, 0,   85},
    { 255, 0,   0},
    { 255, 85,  0},
    { 255, 170, 0},
    { 255, 255, 0},
    { 170, 255, 0},
    { 85,  255, 0},
    { 0,   255, 0},
    { 0,   255, 85},
    { 0,   255, 170},
    { 0,   255, 255},
    { 0,   170, 255},
    { 0,   85,  255},
    { 0,   0,   255},
    { 255, 0,   170},
    { 170, 0,   255},
    { 255, 0,   255},
    { 85,  0,   255},
};

static std::vector<std::pair<int, int>> point_pairs = {
    {6, 7}, {6, 8}, {8, 0}, {7, 9}, {9, 1}, {14, 10}, {14, 11}, {2, 10}, {2, 12}, {3, 11}, {3, 13}
};

void KeyPointDetector::drawKeyPoints(cv::Mat &rgb, const KeyPoints &kpts, float thresh, float thickness, float radius) {

    for( uint i=0 ; i<point_pairs.size() ; i++ ) {

        const auto &kp = point_pairs[i] ;

        int idx1 = kp.first ;
        int idx2 = kp.second ;

        auto it1 = kpts.find(knames[idx1]) ;
        if ( it1 == kpts.end() ) continue ;

        auto it2 = kpts.find(knames[idx2]) ;
        if ( it2 == kpts.end() ) continue ;

        const KeyPoint &kp1 = it1->second ;
        const KeyPoint &kp2 = it2->second ;

        if ( kp1.second < thresh ) continue ;
        if ( kp2.second < thresh ) continue ;

        const Vector2f &p1 = kp1.first ;
        const Vector2f &p2 = kp2.first ;

        cv::Point px1(p1.x(), p1.y()) ;
        cv::Point px2(p2.x(), p2.y()) ;

        const cv::Vec3b &clr = colors[i] ;

        cv::line(rgb, px1, px2, cv::Scalar(clr[0], clr[1], clr[2]), thickness) ;
    }

    for( uint i=0 ; i < knames.size() ; i++ ) {
        auto it = kpts.find(knames[i]) ;
        if ( it == kpts.end() ) continue ;

        const KeyPoint &kp = it->second ;
        float weight = kp.second ;

        if ( weight < thresh ) continue ;

        const Vector2f &pt = kp.first ;

        float radius_scaled = weight * radius ;

        const cv::Vec3b &clr = colors[i] ;

        cv::Point px(pt.x(), pt.y()) ;


        cv::circle(rgb, px, std::round(radius_scaled), cv::Scalar(clr[0], clr[1], clr[2]), thickness) ;

    }

}
