#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

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

    static void build_bases(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        std::vector<Basis>& bases);
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

Eigen::SparseMatrix<double> compute_L2_projection_weights(
    const Eigen::MatrixXd& V_fem,
    const Eigen::MatrixXi& F_fem,
    const Eigen::MatrixXd& V_coll,
    const Eigen::MatrixXi& F_coll,
    bool lump_mass_matrix = true);

Eigen::SparseMatrix<double> compute_mass_mat(
    const int num_nodes,
    const std::vector<Basis>& bases,
    const Quadrature& quadrature);

Eigen::SparseMatrix<double> compute_mass_mat_cross(
    const Eigen::MatrixXd& V_fem,
    const Eigen::MatrixXi& F_fem,
    const std::vector<Basis>& fem_bases,
    const Eigen::MatrixXd& V_coll,
    const Eigen::MatrixXi& F_coll,
    const std::vector<Basis>& coll_bases,
    const Quadrature& quadrature);

void eval_phi_j(
    const std::vector<Basis>& bases,
    const size_t index,
    const Eigen::Vector2d& x,
    std::vector<std::pair<size_t, double>>& out);