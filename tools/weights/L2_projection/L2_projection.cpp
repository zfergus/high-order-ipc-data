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

namespace L2 {

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
    return 0.5 * cross(v1 - v0, v2 - v0).norm();
}

std::vector<Basis>
Basis::build_bases(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    std::vector<Basis> bases;
    for (int fi = 0; fi < F.rows(); fi++) {
        bases.emplace_back();
        bases.back().loc_2_glob = F.row(fi);
        bases.back().v0 = V.row(F(fi, 0));
        bases.back().v1 = V.row(F(fi, 1));
        bases.back().v2 = V.row(F(fi, 2));
    }
    return bases;
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
                    Eigen::Vector2d x = t.tail<2>();
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

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> build_rays(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Basis>& bases,
    const Eigen::MatrixXd& quadrature_points)
{
    Eigen::MatrixXd N;
    igl::per_vertex_normals(V, F, N);

    Eigen::MatrixXd origins(bases.size() * 3 * quadrature_points.rows(), 3);
    Eigen::MatrixXd rays(origins.rows(), origins.cols());
    int ray_i = 0;
    for (const auto& basis : bases) {
        for (int i = 0; i < basis.n_bases; i++) {
            for (int qi = 0; qi < quadrature_points.rows(); qi++) {
                const Eigen::Vector3d& t = quadrature_points.row(qi);
                origins.row(ray_i) = basis.gmapping(t);
                rays.row(ray_i) = t[0] * N.row(basis.loc_2_glob(0))
                    + t[1] * N.row(basis.loc_2_glob(1))
                    + t[2] * N.row(basis.loc_2_glob(2));
                ray_i++;
            }
        }
    }

    return std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>(origins, rays);
}

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
    const Quadrature& quadrature)
{
    std::vector<Eigen::Triplet<double>> tripets;

    ///////////////////////////////////////////////////////////////////////////

    std::cout << "Computing rays" << std::endl;
    auto origins_and_rays =
        build_rays(V_coll, F_coll, coll_bases, quadrature.points);
    const auto& [origins, rays] = origins_and_rays;

    ///////////////////////////////////////////////////////////////////////////

    std::cout << "Computing closest_points" << std::endl;
    Eigen::MatrixX3d ids_and_coords =
        closest_points(origins, rays, V_fem, F_fem);

    ///////////////////////////////////////////////////////////////////////////

    std::cout << "Evaluating bases" << std::endl;
    int query = 0;
    for (const Basis& basis_i : coll_bases) {
        for (int i = 0; i < basis_i.n_bases; i++) {
            std::unordered_map<size_t, double> others;

            for (int qi = 0; qi < quadrature.points.rows(); qi++) {
                const Eigen::Vector3d& t = quadrature.point(qi);
                const double w = quadrature.weight(qi);

                // evaluate ϕᵢ
                const Eigen::Vector2d x_i = t.tail<2>();
                const double phi_i = basis_i.phi(i)(x_i);

                // Get the basis for j
                const int index = int(ids_and_coords(query, 0));
                if (index < 0 || index >= fem_bases.size()) {
                    throw std::runtime_error("invalid index");
                }
                const Basis& basis_j = fem_bases[index];
                const Eigen::Vector2d x_j = ids_and_coords.row(query).tail<2>();
                query++;

                // Evaluate ϕⱼ
                for (int loc_j = 0; loc_j < basis_j.n_bases; loc_j++) {
                    double phi_j = basis_j.phi(loc_j)(x_j);
                    // what about basis_j.grad_gmapping(t)
                    double val = w * phi_i * phi_j * basis_i.grad_gmapping(t);

                    int j = basis_j.loc_2_glob(loc_j);
                    auto got = others.find(j);
                    if (got == others.end()) {
                        others[j] = val;
                    } else {
                        got->second += val;
                    }
                }
            }

            for (const auto& [j, val] : others) {
                tripets.emplace_back(basis_i.loc_2_glob(i), j, val);
            }
        }
    }

    std::cout << "Building matrix" << std::endl;
    Eigen::SparseMatrix<double> A;
    A.resize(V_coll.rows(), V_fem.rows());
    A.setFromTriplets(tripets.begin(), tripets.end());
    return A;
}

} // namespace L2