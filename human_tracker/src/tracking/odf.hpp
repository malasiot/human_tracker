#ifndef ODF_HPP
#define ODF_HPP

#include <htrac/model/skinned_mesh.hpp>
#include <opencv2/opencv.hpp>
#include <cvx/camera/camera.hpp>

class ObservationsDistanceTransform {
public:
    ObservationsDistanceTransform(float cell_size, float padding): cell_size_(cell_size), padding_(padding) {}

    void compute(const SkinnedMesh &mesh, const Pose &p, const cv::Mat &im, const cvx::PinholeCamera &cam) ;

    bool distance(const Eigen::Vector3f &p, float &dist) const ;
    bool gradient(const Eigen::Vector3f &p, Eigen::Vector3f &g) const ;

private:

    float cell_size_, padding_ ;
    std::unique_ptr<float []> dist_ ;
    std::unique_ptr<float []> gradient_ ;
    Eigen::Vector3f bmin_, bmax_ ;
    int ncells_x_, ncells_y_, ncells_z_ ;

    bool trilinear(float vx, float vy, float vz, float *data, uint offset, float &df) const;
};

#endif
