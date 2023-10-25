#ifndef HTRAC_IMAGE_TO_MODEL_TERM_HPP
#define HTRAC_IMAGE_TO_MODEL_TERM_HPP

#include <memory>

#include "energy_term.hpp"
#include <htrac/model/sdf_model.hpp>

class ImageToModelTerm {
public:

    ImageToModelTerm(const Skeleton &sk, const std::shared_ptr<SDFModel> &sdf,  float outlier_threshold):
        skeleton_(sk), sdf_(sdf), outlier_threshold_(outlier_threshold) {}

    void setImageCoords(const std::vector<Eigen::Vector3f> &&icoords) {
        icoords_ = icoords ;
    }

    double energy(const Pose &pose) ;
    std::pair<double, Eigen::VectorXd> energyGradient(const Pose &pose) ;

private:

    float outlier_threshold_ ;
    std::shared_ptr<SDFModel> sdf_ ;
    const Skeleton &skeleton_ ;
    std::vector<Eigen::Vector3f> icoords_ ;
};



#endif
