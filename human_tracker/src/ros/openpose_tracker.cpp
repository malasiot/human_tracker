#include <htrac/ros/openpose_tracker.hpp>
#include <cvx/imgproc/rgbd.hpp>

#include "tf2_eigen/tf2_eigen.hpp"
#include "image_geometry/pinhole_camera_model.h"

using std::placeholders::_1, std::placeholders::_2, std::placeholders::_3;

using namespace Eigen ;
using namespace std ;

OpenPoseTracker::OpenPoseTracker()
    : rclcpp::Node("openpose_tracker") {

    declare_parameter("rgb", "/camera/color/image_raw");
    declare_parameter("depth", "/camera/aligned_depth_to_color/image_raw") ;
    declare_parameter("info", "/camera/color/camera_info") ;
    declare_parameter("data", "") ;
    declare_parameter("kp_thresh", 0.5) ;

    const std::string data_folder = get_parameter("data").as_string();

    KeyPointDetectorOpenPose::Parameters params ;
    params.data_folder_ = data_folder ;

    detector_.reset(new KeyPointDetectorOpenPose(params)) ;
    detector_->init() ;

    kp_thresh_ = get_parameter("kp_thresh").as_double();



    subscribe();
}

inline void OpenPoseTracker::subscribe() {
    const std::string rgb_topic = get_parameter("rgb").as_string();
    const std::string caminfo_topic = get_parameter("info").as_string();
    const std::string depth_topic = get_parameter("depth").as_string();

    rgb_sub_.subscribe(this, rgb_topic) ;
    depth_sub_.subscribe(this, depth_topic) ;

    caminfo_sub_ =  create_subscription<sensor_msgs::msg::CameraInfo>(
                caminfo_topic, 1, std::bind(&OpenPoseTracker::camInfoCallback, this, _1)) ;

    sync_.reset(new Synchronizer( SyncPolicy(10), rgb_sub_, depth_sub_ ));
    sync_->registerCallback(std::bind(&OpenPoseTracker::frameCallback,this,_1, _2));

    publisher_ = create_publisher<human_tracker_interfaces::msg::Frame>("frame", 10);

    marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>("skeleton", 10);

    pub_point_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("points", 1);
}

static const std::vector<std::string> s_joint_names = {
    "LeftFoot", // 0
    "RightFoot", // 1
    "LeftForeArm", //2
    "RightForeArm", //3
    "eye.L", // 4
    "eye.R", // 5,
    "LeftUpLeg", //6
    "RightUpLeg", //7
    "LeftLeg", // 8
    "RightLeg", // 9
    "LeftArm", // 10
    "RightArm", // 11
    "LeftHand", // 12
    "RightHand", // 13
    "Neck"  // 14
};
static const std::vector<std::pair<size_t, size_t>> s_bones = {
    { 6, 8 }, { 8, 0 }, { 7, 9 }, { 9, 1 }, {10, 2}, {2, 12}, {11, 3}, {3, 13}, {10, 14}, {11, 14}, {10, 6}, {11, 7}, {6, 7}
};

visualization_msgs::msg::MarkerArray OpenPoseTracker::makeVizMarker(const OpenPoseTracker::KeyPoints3 &kpts) {
    size_t count = 0 ;
    visualization_msgs::msg::MarkerArray markers ;
    for( size_t i=0 ; i<s_bones.size() ; i++ ) {

        visualization_msgs::msg::Marker marker ;

        marker.ns = "ns_skeleton" ;
        marker.id = i ;
        marker.header.frame_id = "/camera_color_optical_frame" ;
        marker.header.stamp = now() ;

        const auto &bp = s_bones[i] ;
        size_t idx1 = bp.first ;
        size_t idx2 = bp.second ;

        const string &j1 = s_joint_names[idx1] ;
        const string &j2 = s_joint_names[idx2] ;

        auto it1 = kpts.find(j1) ;
        auto it2 = kpts.find(j2) ;
        if ( it1 == kpts.end() || it2 == kpts.end() ) {

            marker.type = visualization_msgs::msg::Marker::CYLINDER ;
            marker.action = visualization_msgs::msg::Marker::DELETE;
        } else {
            const auto &v1 = it1->second.first ;
            const auto &v2 = it2->second.first ;

            float len = (v1 - v2).norm() ;
            Vector3d a{ 0, 0, 1 };
            Vector3d d = (v2 - v1).normalized().cast<double>() ;

            // rotate cylinder to align with bone axis
            Quaterniond q = Quaterniond::FromTwoVectors(a, d) ;

            Vector3f t = 0.5*(v1 + v2) ;


            marker.type = visualization_msgs::msg::Marker::CYLINDER ;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = t.x();
            marker.pose.position.y = t.y();
            marker.pose.position.z = t.z();

            marker.pose.orientation = tf2::toMsg(q) ;

            marker.scale.x = 0.01 ;
            marker.scale.y = 0.01 ;
            marker.scale.z = len ;

            marker.color.r = 0.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
            marker.color.a = 1.0;

            marker.lifetime = rclcpp::Duration(10, 0);
        }

        markers.markers.push_back(marker) ;
    }

    return markers ;
}

void OpenPoseTracker::frameCallback(const sensor_msgs::msg::Image::ConstSharedPtr colorMsg, const sensor_msgs::msg::Image::ConstSharedPtr depthMsg) {

    try
    {
        auto colorPtr = cv_bridge::toCvCopy(colorMsg, sensor_msgs::image_encodings::BGR8);

        rgb_ = colorPtr->image;
        frame_number_ ++ ;
    }
    catch (cv_bridge::Exception& e)
    {
        // display the error at most once per 10 seconds
        RCLCPP_ERROR_THROTTLE(get_logger(), *this->get_clock(), 10, "cv_bridge exception %s at line number %d on function %s in file %s", e.what(), __LINE__,
                              __FUNCTION__, __FILE__);
    }

    try
    {
        auto depthPtr = cv_bridge::toCvCopy(depthMsg, sensor_msgs::image_encodings::TYPE_16UC1);

        if (depthMsg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 || depthMsg->encoding == sensor_msgs::image_encodings::MONO16)
            depth_ = depthPtr->image; // no conversion needed
    }
    catch (cv_bridge::Exception& e)
    {
        // display the error at most once per 10 seconds
        RCLCPP_ERROR_THROTTLE(get_logger(), *this->get_clock(), 10, "cv_bridge exception %s at line number %d on function %s in file %s", e.what(), __LINE__,
                              __FUNCTION__, __FILE__);
    }

    if ( camera_info_ ) {
        auto kpts = detector_->findKeyPoints(rgb_) ;

        cvx::PinholeCamera cam(camera_info_->k.at(0), camera_info_->k.at(4), camera_info_->k.at(2), camera_info_->k.at(5), cv::Size(camera_info_->width, camera_info_->height)) ;

        auto kpts3 = getKeyPoints3d(kpts, cam, depth_) ;

        human_tracker_interfaces::msg::Frame frame ;
        frame.header.stamp = now() ;

        visualization_msgs::msg::MarkerArray markers = makeVizMarker(kpts3) ;

        for( const auto &kp: kpts3 ) {
            human_tracker_interfaces::msg::Joint joint ;
            joint.point.x = kp.second.first.x() ;
            joint.point.y = kp.second.first.y() ;
            joint.point.z = kp.second.first.z() ;
            joint.name = kp.first ;
            joint.score = kp.second.second ;

            frame.joints.emplace_back(joint) ;
        }

        publisher_->publish(frame) ;
        marker_publisher_->publish(markers) ;


        // RCLCPP_INFO(get_logger(), "%d", kpts.size()) ;
       //  publishPointCloud(depthMsg) ;
    }


}

void OpenPoseTracker::publishPointCloud(const sensor_msgs::msg::Image::ConstSharedPtr depth_msg) {
    // Update camera model
    image_geometry::PinholeCameraModel cam ;
    cam.fromCameraInfo(camera_info_) ;


    auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();

    cloud_msg->header = depth_msg->header;  // Use depth image time stamp
    cloud_msg->height = depth_msg->height;
    cloud_msg->width = depth_msg->width;
    cloud_msg->is_dense = false;
    cloud_msg->is_bigendian = false;

    sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
    pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

    // Use correct principal point from calibration
    float center_x = cam.cx();
    float center_y = cam.cy();

    // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
    double unit_scaling = 0.001;
    float constant_x = unit_scaling / cam.fx();
    float constant_y = unit_scaling / cam.fy();
    float bad_point = std::numeric_limits<float>::quiet_NaN();

    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
    const uint16_t * depth_row = reinterpret_cast<const uint16_t *>(depth_.data);

    uint step = depth_.step1() ;
    for (int v = 0; v < static_cast<int>(cloud_msg->height); ++v, depth_row += depth_.step1()) {
        for (int u = 0; u < static_cast<int>(cloud_msg->width); ++u, ++iter_x, ++iter_y, ++iter_z) {
            uint16_t depth = depth_row[u];

            // Missing points denoted by NaNs
            if ( depth == 0 ) {
                *iter_x = *iter_y = *iter_z = bad_point;
                continue;
            }

            // Fill in XYZ
            *iter_x = (u - center_x) * depth * constant_x;
            *iter_y = (v - center_y) * depth * constant_y;
            *iter_z = depth * unit_scaling ;

        }
    }

    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud_msg, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud_msg, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud_msg, "b");
    const uint8_t * rgb = rgb_.data ;
    int rgb_skip = rgb_.step1() * 3;
     for (int v = 0; v < static_cast<int>(cloud_msg->height); ++v, rgb += rgb_skip) {
       for (int u = 0; u < static_cast<int>(cloud_msg->width); ++u,
         rgb += 3, ++iter_r, ++iter_g, ++iter_b)
       {
         *iter_r = rgb[0];
         *iter_g = rgb[1];
         *iter_b = rgb[2];
       }
     }


    pub_point_cloud_->publish(*cloud_msg);
}

OpenPoseTracker::KeyPoints3 OpenPoseTracker::getKeyPoints3d(const KeyPoints &kpts, const cvx::PinholeCamera &cam, const cv::Mat &depth) {
    KeyPoints3 res ;

    for( const auto &kp: kpts ) {
        const auto &name = kp.first ;
        const auto &coords = kp.second.first ;
        const auto &weight = kp.second.second ;

        ushort z ;
        if ( cvx::sampleNearestNonZeroDepth(depth, round(coords.x()), round(coords.y()), z, 3) && weight > kp_thresh_ ) {
            Vector3f p = cam.backProject(coords.x(), coords.y(), z/1000.0) ;
            p.y() =-p.y() ;
            p.z() =-p.z() ;
            res.emplace(name, std::make_pair(p, weight)) ;
        }
    }

    return res ;
}



void OpenPoseTracker::camInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr camMsg) {
    camera_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camMsg);

    if ( camera_info_ != nullptr)
        caminfo_sub_.reset() ;
}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OpenPoseTracker>());
    rclcpp::shutdown();
}
