#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <cvx/camera/camera.hpp>

using PointCloud = std::vector<Eigen::Vector3f> ;
using CoordMap   = cv::Mat_<cv::Vec3f> ;
using BoundingBox = std::pair<Eigen::Vector3f, Eigen::Vector3f> ;

std::vector<Eigen::Vector3f> depth_to_point_cloud(const cv::Mat &im, float camfy) ;

Eigen::Vector3f centroid(const std::vector<Eigen::Vector3f> &pts) ;
Eigen::Matrix3f covariance(const std::vector<Eigen::Vector3f> &pts, const Eigen::Vector3f &c) ;
void eigenvalues(const  Eigen::Matrix3f &cov,  Eigen::Vector3f &eval,  Eigen::Vector3f evec[3]) ;
Eigen::Vector3f box_dimensions(const std::vector<Eigen::Vector3f> &pts, const Eigen::Vector3f &c, const Eigen::Vector3f evec[3]) ;

Eigen::VectorXf pcl_features(const std::vector<Eigen::Vector3f> &pcl) ;

BoundingBox get_bounding_box(const std::vector<Eigen::Vector3f> &pcl) ;

PointCloud depthToPointCloud(const cv::Mat &depth, const cvx::PinholeCamera &cam, uint cell_size = 1) ;
PointCloud depthToPointCloud(const cv::Mat &depth, const cvx::PinholeCamera &cam, const cv::Mat &mask, uint cell_size = 1) ;
