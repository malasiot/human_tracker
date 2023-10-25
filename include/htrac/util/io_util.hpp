#pragma once

#include <Eigen/Core>
#include <cvx/misc/json_reader.hpp>
#include <map>
#include <opencv2/opencv.hpp>

Eigen::MatrixXf parse_json_matrix(cvx::JSONReader &json) ;
void save_joint_coords(const std::map<std::string, Eigen::Vector3f> &joints, const std::string &path);
void save_point_cloud(const std::vector<Eigen::Vector3f> &coords, const std::string &path) ;

void skeleton_to_obj(const std::vector<std::pair<int, int>> &skeleton, const std::vector<Eigen::Vector3f> &joints,
                     const std::string &outpath) ;

void draw_skeleton(const cv::Mat &im, const std::vector<std::pair<int, int>> &skeleton, const std::vector<Eigen::Vector2f> &joints,
                   const std::string &outpath) ;

