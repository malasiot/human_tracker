#pragma once

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <cvx/camera/camera.hpp>

class ITOP {
public:

    struct Annotation {
        cv::Point2i ipt_ ;
        Eigen::Vector3f coords_ ;
        bool visible_ ;
    };

    ITOP(const std::string &root_dir, const std::string &exclude = {}, bool shuffle = false) ;

    // Get number of samples available
    uint64_t size() const { return ids_.size() ; }

    // Get target associated with the specific sample
    cv::Mat getDepthImage(uint64_t idx) const ;

    std::vector<Annotation> getAnnotation(uint64_t idx) const ;

    cvx::PinholeCamera getCamera() const ;

    std::vector<std::string> jointNames() const { return joint_names_ ; }

private:

    std::vector<Annotation> parseAnnotation(const std::string &p) const ;

private:

    std::vector<std::string> ids_ ;
    std::string root_dir_ ;
    cv::Size img_size_ = cv::Size{320, 240} ;
    static std::vector<std::string> joint_names_ ;
};


class CERTHDataset {
public:

    struct Annotation {
        Eigen::Vector3f ipt_, coords_ ;
        bool visible_ ;
    };

    CERTHDataset(const std::string &root_dir, bool shuffle = false) ;

    // Get number of samples available
    uint64_t size() const { return ids_.size() ; }

    cv::Mat getDepthImage(uint64_t idx) const ;

    cv::Mat getPartLabelImage(uint64_t idx) const ;

    std::vector<Annotation> getAnnotation(uint64_t idx) const ;

    cvx::PinholeCamera getCamera() const ;

private:

    std::vector<Annotation> parseAnnotation(const std::string &jp, const std::string &pp) const ;

private:

    std::vector<std::string> ids_ ;
    std::string root_dir_ ;
    cv::Size img_size_ = cv::Size{256, 256} ;
};

