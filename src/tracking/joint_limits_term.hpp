#pragma once

#include <htrac/model/skeleton.hpp>

class JointLimitsTerm {
public:

    JointLimitsTerm(const Skeleton &sk);

    float energy(const Pose &pose) const ;
    std::pair<float, Eigen::VectorXf> energyGradient(const Pose &pose) const;

    void energy(const Pose &p, Eigen::VectorXf &e) const  ;
    void jacobian(Eigen::MatrixXf &jac) const ;
    void norm(const Pose &p, Eigen::MatrixXf &jtj, Eigen::VectorXf &jte, float lambda) const ;
    size_t nTerms() const { return n_terms_ ; }


private:

    void compute_energy(const Pose &p, Eigen::VectorXf &e) const;
    void compute_jacobian(const Pose &p, std::vector<std::tuple<uint, uint, float>> &jac, Eigen::VectorXf &e) const ;

    const Skeleton &skeleton_ ;
    uint n_terms_ ;
};
