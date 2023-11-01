#include <htrac/ros/camera_reader.hpp>
using std::placeholders::_1, std::placeholders::_2, std::placeholders::_3;

CameraReader::CameraReader(rclcpp::Node::SharedPtr nh, const std::string& colorTopic, const std::string& depthTopic,
                           const std::string& camInfoTopic)
    : node_handle_(nh), rgb_topic_(colorTopic), depth_topic_(depthTopic), caminfo_topic_(camInfoTopic), rgb_sub_(nh, colorTopic), depth_sub_(nh, depthTopic) {
    // std::cout << "[" << this << "] constructor called" << std::endl;
    subscribe();
}

CameraReader::CameraReader(const CameraReader& other)
    : node_handle_(other.node_handle_), rgb_topic_(other.rgb_topic_), depth_topic_(other.depth_topic_), caminfo_topic_(other.caminfo_topic_) {
    // std::cout << "[" << this << "] copy constructor called" << std::endl;
    subscribe();
}


// we define the subscriber here. we are using TimeSynchronizer filter to receive the synchronized data
inline void CameraReader::subscribe()
{
    caminfo_sub_ =  node_handle_->create_subscription<sensor_msgs::msg::CameraInfo>(
                caminfo_topic_,
                1,
                std::bind(&CameraReader::camInfoCallback, this, _1)) ;


    //            std::bind(&CameraReader::depthImgCallback, this, _1)) ;

    sync_.reset(new Synchronizer( SyncPolicy(10), rgb_sub_, depth_sub_ ));
    sync_->registerCallback(std::bind(&CameraReader::frameCallback,this,_1, _2));

}

void CameraReader::frameCallback(const sensor_msgs::msg::Image::ConstSharedPtr colorMsg, const sensor_msgs::msg::Image::ConstSharedPtr depthMsg) {

    try
    {
        auto colorPtr = cv_bridge::toCvCopy(colorMsg, sensor_msgs::image_encodings::BGR8);

        // it is very important to lock the below assignment operation.
        // remember that we are accessing it from another thread too.
        std::lock_guard<std::mutex> lock(mutex_);
        rgb_ = colorPtr->image;
         mFrameNumber ++ ;
    }
    catch (cv_bridge::Exception& e)
    {
        // display the error at most once per 10 seconds
        RCLCPP_ERROR_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 10, "cv_bridge exception %s at line number %d on function %s in file %s", e.what(), __LINE__,
                           __FUNCTION__, __FILE__);
    }

    try
    {
        auto depthPtr = cv_bridge::toCvCopy(depthMsg, sensor_msgs::image_encodings::TYPE_16UC1);

        // it is very important to lock the below assignment operation.
        // remember that we are accessing it from another thread too.
        std::lock_guard<std::mutex> lock(mutex_);

        // in case of '16UC1' encoding (depth values are in millimeter),
        // a manually conversion from millimeter to meter is required.
        if (depthMsg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 || depthMsg->encoding == sensor_msgs::image_encodings::MONO16)
            // -1 represents no change in datatype
            // src: https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#adf88c60c5b4980e05bb556080916978b
     //       depthPtr->image.convertTo(depth_, -1, 0.001f);
      //  else
            depth_ = depthPtr->image; // no conversion needed
    }
    catch (cv_bridge::Exception& e)
    {
        // display the error at most once per 10 seconds
        RCLCPP_ERROR_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 10, "cv_bridge exception %s at line number %d on function %s in file %s", e.what(), __LINE__,
                           __FUNCTION__, __FILE__);
    }




}

/*
void CameraReader::colorImgCallback(const sensor_msgs::msg::Image::SharedPtr colorMsg) {

    try
    {
        auto colorPtr = cv_bridge::toCvCopy(colorMsg, sensor_msgs::image_encodings::BGR8);

        // it is very important to lock the below assignment operation.
        // remember that we are accessing it from another thread too.
        std::lock_guard<std::mutex> lock(mutex_);
        rgb_ = colorPtr->image;
        mFrameNumber++;
    }
    catch (cv_bridge::Exception& e)
    {
        // display the error at most once per 10 seconds
        RCLCPP_ERROR_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 10, "cv_bridge exception %s at line number %d on function %s in file %s", e.what(), __LINE__,
                           __FUNCTION__, __FILE__);
    }

}

void CameraReader::depthImgCallback(const sensor_msgs::msg::Image::SharedPtr depthMsg) {

    try
    {
        auto depthPtr = cv_bridge::toCvCopy(depthMsg, sensor_msgs::image_encodings::TYPE_16UC1);

        // it is very important to lock the below assignment operation.
        // remember that we are accessing it from another thread too.
        std::lock_guard<std::mutex> lock(mutex_);

        // in case of '16UC1' encoding (depth values are in millimeter),
        // a manually conversion from millimeter to meter is required.
        if (depthMsg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 || depthMsg->encoding == sensor_msgs::image_encodings::MONO16)
            // -1 represents no change in datatype
            // src: https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#adf88c60c5b4980e05bb556080916978b
     //       depthPtr->image.convertTo(depth_, -1, 0.001f);
      //  else
            depth_ = depthPtr->image; // no conversion needed
    }
    catch (cv_bridge::Exception& e)
    {
        // display the error at most once per 10 seconds
        RCLCPP_ERROR_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 10, "cv_bridge exception %s at line number %d on function %s in file %s", e.what(), __LINE__,
                           __FUNCTION__, __FILE__);
    }

}
*/

void CameraReader::camInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr camMsg)
{
    camera_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camMsg);

    // since the calibration parameters are static so we don't need to keep running
    // the subscriber. that is why, we stop the subscriber once we receive
    // the parameters successfully
    if ( camera_info_ != nullptr)
        caminfo_sub_.reset() ;
}

