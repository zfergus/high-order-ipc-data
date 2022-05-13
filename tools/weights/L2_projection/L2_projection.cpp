#include "L2_projection.hpp"

#include <iostream>
#include <vector>
#include <filesystem>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>

#include <igl/readCSV.h>
#include <igl/per_vertex_normals.h>
#include <igl/embree/line_mesh_intersection.h>

template <typename DerivedA, typename DerivedB>
Eigen::Vector3d cross(
    const Eigen::MatrixBase<DerivedA>& a, const Eigen::MatrixBase<DerivedB>& b)
{
    assert(a.size() == 3 && b.size() == 3);
    Eigen::Vector3d c;
    c(0) = a(1) * b(2) - a(2) * b(1);
    c(1) = a(2) * b(0) - a(0) * b(2);
    c(2) = a(0) * b(1) - a(1) * b(0);
    return c;
}

// Compute barycentric coordinates (u, v, w) for
// point p with respect to triangle (a, b, c)
void barycentric_coordinates(
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c,
    Eigen::Vector3d& uvw)
{
    Eigen::Vector3d v0 = b - a, v1 = c - a, v2 = p - a;
    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;
    uvw[1] = (d11 * d20 - d01 * d21) / denom;
    uvw[2] = (d00 * d21 - d01 * d20) / denom;
    uvw[0] = 1.0 - uvw[1] - uvw[2];
}

////////////////////////////////////////////////////////////////////////////////

double hat_phi0(const Eigen::Vector2d& x) { return 1 - x[0] - x[1]; }
double hat_phi1(const Eigen::Vector2d& x) { return x[0]; }
double hat_phi2(const Eigen::Vector2d& x) { return x[1]; }

std::function<double(const Eigen::Vector2d&)> Basis::phi(int i) const
{
    switch (i) {
    case 0:
        return &hat_phi0;
    case 1:
        return &hat_phi1;
    case 2:
        return &hat_phi2;
    default:
        throw "invalid phi i";
    }
}

Eigen::Vector3d Basis::gmapping(const Eigen::Vector3d& bc) const
{
    return v0 * bc[0] + v1 * bc[1] + v2 * bc[2];
}

double Basis::grad_gmapping(const Eigen::Vector3d& bc) const
{
    return cross(v1 - v0, v2 - v0).norm();
}

void Basis::build_bases(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    std::vector<Basis>& bases)
{
    for (int fi = 0; fi < F.rows(); fi++) {
        bases.emplace_back();
        bases.back().loc_2_glob = F.row(fi);
        bases.back().v0 = V.row(F(fi, 0));
        bases.back().v1 = V.row(F(fi, 1));
        bases.back().v2 = V.row(F(fi, 2));
    }
}

////////////////////////////////////////////////////////////////////////////////

Quadrature::Quadrature()
{
    std::filesystem::path quad_data_dir =
        std::filesystem::path(__FILE__).parent_path() / "quadrature";

    igl::readCSV((quad_data_dir / "points.csv").string(), points);
    igl::readCSV((quad_data_dir / "weights.csv").string(), weights);
}

////////////////////////////////////////////////////////////////////////////////

Eigen::SparseMatrix<double> compute_mass_mat(
    const int num_vertices,
    const std::vector<Basis>& bases,
    const Quadrature& quadrature)
{
    std::vector<Eigen::Triplet<double>> tripets;

    for (const Basis& basis : bases) {
        for (int i = 0; i < basis.n_bases; i++) {
            for (int j = 0; j < basis.n_bases; j++) {
                double val = 0;
                for (int qi = 0; qi < quadrature.points.rows(); qi++) {
                    const Eigen::Vector3d& t = quadrature.point(qi);
                    const double w = quadrature.weight(qi);
                    Eigen::Vector2d x = t[0] * Eigen::Vector2d(0, 0)
                        + t[1] * Eigen::Vector2d(1, 0)
                        + t[2] * Eigen::Vector2d(0, 1);
                    val += w * basis.phi(i)(x) * basis.phi(j)(x)
                        * basis.grad_gmapping(t);
                }

                tripets.emplace_back(
                    basis.loc_2_glob(i), basis.loc_2_glob(j), val);
            }
        }
    }

    Eigen::SparseMatrix<double> M;
    M.resize(num_vertices, num_vertices);
    M.setFromTriplets(tripets.begin(), tripets.end());
    return M;
}

////////////////////////////////////////////////////////////////////////////////

void eval_phi_j(
    const std::vector<Basis>& bases,
    const size_t index,
    const Eigen::Vector2d& x,
    std::vector<std::pair<size_t, double>>& out)
{
    const Basis& basis = bases[index];
    out.clear();
    out.reserve(basis.n_bases);
    for (int i = 0; i < basis.n_bases; i++) {
        out.emplace_back(basis.loc_2_glob(i), basis.phi(i)(x));
    }
}

Eigen::SparseMatrix<double> compute_mass_mat_cross(
    const Eigen::MatrixXd& V_fem,
    const Eigen::MatrixXi& F_fem,
    const std::vector<Basis>& fem_bases,
    const Eigen::MatrixXd& V_coll,
    const Eigen::MatrixXi& F_coll,
    const std::vector<Basis>& coll_bases,
    const Quadrature& quadrature)
{
    std::vector<Eigen::Triplet<double>> tripets;

    Eigen::MatrixXd N_coll;
    igl::per_vertex_normals(V_coll, F_coll, N_coll);

    Eigen::MatrixXd origins(
        coll_bases.size() * 3 * quadrature.points.rows(), 3);
    Eigen::MatrixXd rays(origins.rows(), origins.cols());

    size_t ray_i = 0;
    for (const Basis& basis : coll_bases) {
        for (int i = 0; i < basis.n_bases; i++) {
            for (int qi = 0; qi < quadrature.points.rows(); qi++) {
                const Eigen::Vector3d& t = quadrature.point(qi);
                origins.row(ray_i) = basis.gmapping(t);
                rays.row(ray_i) = t[0] * N_coll.row(basis.loc_2_glob(0))
                    + t[1] * N_coll.row(basis.loc_2_glob(1))
                    + t[2] * N_coll.row(basis.loc_2_glob(2));
                ray_i++;
            }
        }
    }

    Eigen::MatrixX3d ids_and_coords =
        igl::embree::line_mesh_intersection(origins, rays, V_fem, F_fem);

    ray_i = 0;
    for (const Basis& basis : coll_bases) {
        for (int i = 0; i < basis.n_bases; i++) {
            std::unordered_map<size_t, double> others;

            for (int qi = 0; qi < quadrature.points.rows(); qi++) {
                const Eigen::Vector3d& t = quadrature.point(qi);
                const double w = quadrature.weight(qi);

                std::vector<std::pair<size_t, double>> bb;
                eval_phi_j(
                    fem_bases, int(ids_and_coords(ray_i, 0)),
                    ids_and_coords.row(ray_i).tail<2>(), bb);
                ray_i++;

                const double phi_i = basis.phi(i)(t.tail<2>());
                for (const auto& [j, phi_j] : bb) {
                    double val = w * phi_i * phi_j * basis.grad_gmapping(t);

                    auto got = others.find(j);
                    if (got == others.end()) {
                        others[j] = val;
                    } else {
                        got->second += val;
                    }
                }
            }

            for (const auto& [j, val] : others) {
                tripets.emplace_back(basis.loc_2_glob(i), j, val);
            }
        }
    }

    Eigen::SparseMatrix<double> A;
    A.resize(V_coll.rows(), V_fem.rows());
    A.setFromTriplets(tripets.begin(), tripets.end());
    return A;
}

////////////////////////////////////////////////////////////////////////////////

Eigen::SparseMatrix<double> compute_L2_projection_weights(
    const Eigen::MatrixXd& V_fem,
    const Eigen::MatrixXi& F_fem,
    const Eigen::MatrixXd& V_coll,
    const Eigen::MatrixXi& F_coll,
    bool lump_mass_matrix)
{
    Quadrature quadrature;

    std::vector<Basis> fem_bases, coll_bases;
    Basis::build_bases(V_fem, F_fem, fem_bases);
    Basis::build_bases(V_coll, F_coll, coll_bases);

    Eigen::SparseMatrix<double> M =
        compute_mass_mat(V_coll.rows(), coll_bases, quadrature);

    if (lump_mass_matrix) {
        Eigen::VectorXd lumped_values = Eigen::VectorXd::Zero(M.rows());
        for (int k = 0; k < M.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it;
                 ++it) {
                lumped_values(it.row()) += it.value();
            }
        }
        M.setZero();
        M = lumped_values.asDiagonal();
    }

    Eigen::SparseMatrix<double> A = compute_mass_mat_cross(
        V_fem, F_fem, fem_bases, V_coll, F_coll, coll_bases, quadrature);

    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
        solver;
    // Compute the ordering permutation vector from the structural pattern of
    solver.analyzePattern(M);
    // Compute the numerical factorization
    solver.factorize(M);

    if (solver.info() != Eigen::Success) {
        Eigen::saveMarket(M, "mass_matrix.mtx");
        throw std::runtime_error("Unable to factorize mass matrix");
    }

    // Use the factors to solve the linear system
    Eigen::SparseMatrix<double> W = solver.solve(A);

    return W;
}