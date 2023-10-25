#ifndef MODEL_TO_IMAGE_TERM_HPP
#define MODEL_TO_IMAGE_TERM_HPP

#include <memory>

#include "energy_term.hpp"
#include "odf.hpp"

#include <htrac/model/skinned_mesh.hpp>

class ModelToImageTerm: public EnergyTerm {
public:

    ModelToImageTerm(const SkinnedMesh &mesh, const cvx::PinholeCamera &cam):
        mesh_(mesh), cam_(cam) {
    }

    void computeDistanceTransform(const cv::Mat &mask) ;

    double energy(const Pose &pose) ;
    std::pair<double, Eigen::VectorXd> energyGradient(const Pose &pose) ;

private:

    float dtValue(const Eigen::Vector2f &p) ;
    Eigen::Vector2f dtGradient(const Eigen::Vector2f &p) ;

    const SkinnedMesh &mesh_ ;
    cv::Mat dt_, dt_grad_ ;
    cvx::PinholeCamera cam_ ;
};



#endif
