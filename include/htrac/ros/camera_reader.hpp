#pragma once

// ROS headers
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

#include <opencv2/core/core.hpp>

// c++ headers
#include <mutex>
#include <vector>
#include <memory>

// define a few datatype
typedef unsigned long long ullong;

class CameraReader
{
private:
    cv::Mat rgb_, depth_ ;
    cv::Mat rgb_used_, depth_used_ ;
    std::string rgb_topic_, depth_topic_, caminfo_topic_ ;
    std::mutex mutex_ ;
    rclcpp::Node::SharedPtr node_handle_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_ ;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_ ;

    ullong mFrameNumber = 0ULL;

    // camera calibration parameters
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;

    inline void subscribe();
    void camInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr camMsg);
    void colorImgCallback(const sensor_msgs::msg::Image::SharedPtr colorMsg);
    void depthImgCallback(const sensor_msgs::msg::Image::SharedPtr depthMsg);

public:
    // we don't want to instantiate using deafult constructor
    CameraReader() = delete;

    // copy constructor
    CameraReader(const CameraReader& other);

    // copy assignment operator
    CameraReader& operator=(const CameraReader& other);

    // main constructor
    CameraReader(rclcpp::Node::SharedPtr nh, const std::string& colorTopic, const std::string& depthTopic,
                 const std::string& camInfoTopic);

    // we are okay with default destructor
    ~CameraReader() = default;

    // returns the current frame number
    // the frame number starts from 0 and increments
    // by 1 each time a frame (color) is received
    ullong getFrameNumber()
    {
        return mFrameNumber;
    }

    // returns the latest color frame from camera
    // it locks the color frame. remember that we
    // are just passing the pointer instead of copying whole data
    const cv::Mat& getColorFrame()
    {
        mutex_.lock();
        rgb_used_ = rgb_;
        mutex_.unlock();
        return rgb_used_;
    }

    // returns the latest depth frame from camera
    // it locks the depth frame. remember that we
    // are just passing the pointer instead of copying whole data
    const cv::Mat& getDepthFrame()
    {
        mutex_.lock();
        depth_used_ = depth_;
        mutex_.unlock();
        return depth_used_;
    }

    // lock the latest depth frame from camera. remember that we
    // are just passing the pointer instead of copying whole data
    void lockLatestDepthImage()
    {
        mutex_.lock();
        depth_used_ = depth_;
        mutex_.unlock();
    }

    // compute the point in 3D space for a given pixel without considering distortion
    void compute3DPoint(const float pixelX, const float pixelY, float (&point)[3])
    {
        // K.at(0) = intrinsic.fx
        // K.at(4) = intrinsic.fy
        // K.at(2) = intrinsic.ppx
        // K.at(5) = intrinsic.ppy

        // our depth frame type is 16UC1 which has unsigned short as an underlying type
        // auto depth = mDepthImageUsed.at<unsigned short>(static_cast<int>(pixelY), static_cast<int>(pixelX));
        auto depth = depth_used_.at<float>(static_cast<int>(pixelY), static_cast<int>(pixelX));

        // no need to proceed further if the depth is zero or less than zero
        // the depth represents the distance of an object placed infront of the camera
        // therefore depth must always be a positive number
        if (depth <= 0)
            return;

        // the following calculation can also be done by image_geometry
        // for example:
        // image_geometry::PinholeCameraModel camModel;
        // camModel.fromCameraInfo(mSPtrCameraInfo);
        // cv::Point2d depthPixel(pixelX, pixelY);
        // auto point3d = camModel.projectPixelTo3dRay(depthPixel)
        // auto depth = mDepthImageUsed.at<unsigned short>(depthPixel);
        // point[0] = depth * point3d.x;
        // point[1] = depth * point3d.y;
        // point[2] = depth * point3d.z;
        // for more info., please see http://wiki.ros.org/image_geometry

        auto x = (pixelX - camera_info_->k.at(2)) / camera_info_->k.at(0);
        auto y = (pixelY - camera_info_->k.at(5)) / camera_info_->k.at(4);

        point[0] = depth * x;
        point[1] = depth * y;
        point[2] = depth;
    }
};

