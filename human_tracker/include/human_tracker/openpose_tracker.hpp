#pragma once

// ROS headers
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/subscriber.h"
#include "human_tracker_msgs/msg/frame.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <htrac/pose/keypoint_detector_openpose.hpp>
#include <htrac/pose/pose_from_keypoints.hpp>

#include <opencv2/core/core.hpp>

#include <mutex>
#include <vector>
#include <memory>

class OpenPoseTracker: public rclcpp::Node
{
private:
    cv::Mat rgb_, depth_ ;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_ ;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_ ;
    rclcpp::Publisher<human_tracker_msgs::msg::Frame>::SharedPtr publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_point_cloud_;

    using SyncPolicy =  typename message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> ;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy> ;
    std::unique_ptr<Synchronizer> sync_ ;

    std::unique_ptr<KeyPointDetectorOpenPose> detector_ ;

    size_t frame_number_ = 0;
    double kp_thresh_ = 0.5 ;

    // camera calibration parameters
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;

    using KeyPoints3 = std::map<std::string, std::pair<Eigen::Vector3f, float>> ;

    KeyPoints3 getKeyPoints3d(const KeyPoints &kpts, const cvx::PinholeCamera &cam, const cv::Mat &depth);

    inline void subscribe();
    void camInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr camMsg);
    void frameCallback(const sensor_msgs::msg::Image::ConstSharedPtr colorMsg, const sensor_msgs::msg::Image::ConstSharedPtr depthMsg) ;

    visualization_msgs::msg::MarkerArray makeVizMarker(const OpenPoseTracker::KeyPoints3 &kpts);

    void publishPointCloud(const sensor_msgs::msg::Image::ConstSharedPtr depthMsg) ;
public:

    OpenPoseTracker();
    ~OpenPoseTracker() = default;

};

