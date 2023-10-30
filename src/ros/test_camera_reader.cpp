#include <htrac/ros/camera_reader.hpp>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <cvx/imgproc/rgbd.hpp>

void show(const std::shared_ptr<CameraReader> &reader, const std::shared_ptr<rclcpp::Node> &node)
{
    rclcpp::Rate loopRate(10);
    while (rclcpp::ok())
    {
        auto colorImage = reader->getColorFrame();
        auto depthImage = reader->getDepthFrame();

        if (!colorImage.empty())
            cv::imshow("color image", colorImage);
        //  else
        // display the error at most once per 10 seconds
        //    ROS_WARN_THROTTLE(10, "Empty color image frame detected. Ignoring...");

        if (!depthImage.empty())
            cv::imshow("depth image", cvx::depthViz(depthImage));
        //  else
        // display the error at most once per 10 seconds
        //    ROS_WARN_THROTTLE(10, "Empty depth image frame detected. Ignoring...");

        int key = (cv::waitKey(1) & 255);
        if ( key == 27)  // escape key
            break;

        rclcpp::spin_some(node);

        loopRate.sleep();
    }
}

int main(int argc, char *argv[]) {

    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("camera_tester");

    node->declare_parameter("rgb", "/camera/color/image_raw");
    node->declare_parameter("depth", "/camera/aligned_depth_to_color/image_raw") ;
    node->declare_parameter("info", "/camera/color/camera_info") ;


    const std::string colorTopic = node->get_parameter("rgb").as_string();
    const std::string camInfoTopic = node->get_parameter("info").as_string();
    const std::string depthTopic = node->get_parameter("depth").as_string();
    auto camera_reader = std::make_shared<CameraReader>(node, colorTopic, depthTopic, camInfoTopic) ;


    show(camera_reader, node);

    rclcpp::shutdown();


}
