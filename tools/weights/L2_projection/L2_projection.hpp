#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace L2 {

double hat_phi0(const Eigen::Vector2d& x);
double hat_phi1(const Eigen::Vector2d& x);
double hat_phi2(const Eigen::Vector2d& x);

class Basis {
public:
    std::function<double(const Eigen::Vector2d&)> phi(int i) const;
    Eigen::Vector3d gmapping(const Eigen::Vector3d& bc) const;
    double grad_gmapping(const Eigen::Vector3d& bc) const;

    const int n_bases = 3;
    Eigen::RowVector3i loc_2_glob;
    Eigen::Vector3d v0;
    Eigen::Vector3d v1;
    Eigen::Vector3d v2;

    static std::vector<Basis>
    build_bases(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
};

////////////////////////////////////////////////////////////////////////////////

class Quadrature {
public:
    Quadrature();

    Eigen::Vector3d point(const size_t i) const { return points.row(i); }
    double weight(const size_t i) const { return weights(i); }

    Eigen::MatrixXd points;
    Eigen::MatrixXd weights;
};

////////////////////////////////////////////////////////////////////////////////

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> build_rays(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Basis>& bases,
    const Eigen::MatrixXd& quadrature_points);

Eigen::SparseMatrix<double> compute_mass_mat(
    const int num_nodes,
    const std::vector<Basis>& bases,
    const Quadrature& quadrature = Quadrature());

Eigen::SparseMatrix<double> compute_mass_mat_cross(
    const Eigen::MatrixXd& V_fem,
    const Eigen::MatrixXi& F_fem,
    const std::vector<Basis>& fem_bases,
    const Eigen::MatrixXd& V_coll,
    const Eigen::MatrixXi& F_coll,
    const std::vector<Basis>& coll_bases,
    const std::function<Eigen::MatrixXd(
        const Eigen::MatrixXd&,
        const Eigen::MatrixXd&,
        const Eigen::MatrixXd&,
        const Eigen::MatrixXi)> closest_points,
    const Quadrature& quadrature = Quadrature());

} // namespace L2