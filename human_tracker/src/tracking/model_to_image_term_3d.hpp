#ifndef MODEL_TO_IMAGE_TERM_HPP
#define MODEL_TO_IMAGE_TERM_HPP

#include <memory>

#include "energy_term.hpp"
#include "odf.hpp"

#include <htrac/model/skinned_mesh.hpp>

class ModelToImageTerm: public EnergyTerm {
public:

    ModelToImageTerm(const SkinnedMesh &mesh, float cell_size, float padding):
        mesh_(mesh), odf_(cell_size, padding) {
    }

    void initSDF(const Pose &pose, const cv::Mat &im, const cvx::PinholeCamera &cam) {
        odf_.compute(mesh_, pose, im, cam) ;
    }

    double energy(const Pose &pose) ;
    std::pair<double, Eigen::VectorXd> energyGradient(const Pose &pose) ;

    const ObservationsDistanceTransform &odf() const { return odf_ ; }

private:

    ObservationsDistanceTransform odf_ ;
    const SkinnedMesh &mesh_ ;
};



#endif
