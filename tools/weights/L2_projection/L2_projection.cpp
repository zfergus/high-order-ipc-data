#include "L2_projection.hpp"

#include <vector>
#include <filesystem>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <igl/readCSV.h>
#include <igl/per_vertex_normals.h>
#include <igl/embree/line_mesh_intersection.h>

#include <spdlog/spdlog.h>

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

void find_and_eval(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Basis>& bases,
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& normal,
    std::vector<std::pair<size_t, double>>& out)
{
    Eigen::RowVector3d id_and_coord = igl::embree::line_mesh_intersection(
        p.transpose(), normal.transpose(), V, F);

    // find the closest element index
    size_t index(id_and_coord(0));
    // find the coords of the closest point in the closest element
    // Eigen::Vector3d coords;
    // barycentric_coordinates(
    //     p, V.row(F(index, 0)), V.row(F(index, 1)), V.row(F(index, 2)),
    //     coords);
    Eigen::Vector3d coords(
        1 - id_and_coord(1) - id_and_coord(2), //
        id_and_coord(1),                       //
        id_and_coord(2));
    assert(std::isfinite(coords[0]));
    assert(std::isfinite(coords[1]));
    assert(std::isfinite(coords[2]));

    Eigen::Vector2d x = coords[0] * Eigen::Vector2d(0, 0)
        + coords[1] * Eigen::Vector2d(1, 0) + coords[2] * Eigen::Vector2d(0, 1);

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

    for (const Basis& basis : coll_bases) {
        for (int i = 0; i < basis.n_bases; i++) {
            std::unordered_map<size_t, double> others;

            for (int qi = 0; qi < quadrature.points.rows(); qi++) {
                const Eigen::Vector3d& t = quadrature.point(qi);
                const double w = quadrature.weight(qi);

                Eigen::Vector2d x = t[0] * Eigen::Vector2d(0, 0)
                    + t[1] * Eigen::Vector2d(1, 0)
                    + t[2] * Eigen::Vector2d(0, 1);

                Eigen::Vector3d pt = basis.gmapping(t);
                Eigen::Vector3d normal = //
                    t[0] * N_coll.row(basis.loc_2_glob(0))
                    + t[1] * N_coll.row(basis.loc_2_glob(0))
                    + t[2] * N_coll.row(basis.loc_2_glob(0));
                std::vector<std::pair<size_t, double>> bb;
                find_and_eval(V_fem, F_fem, fem_bases, pt, normal, bb);

                for (const auto& [j, phi_j] : bb) {
                    double val =
                        w * basis.phi(i)(x) * phi_j * basis.grad_gmapping(t);

                    auto got = others.find(j);
                    if (got == others.end()) {
                        others[j] = val;
                    } else {
                        got->second += val;
                    }
                }

                for (const auto& [j, val] : others) {
                    tripets.emplace_back(basis.loc_2_glob(i), j, val);
                }
            }
        }
    }

    Eigen::SparseMatrix<double> A;
    A.resize(V_coll.rows(), V_fem.rows());
    A.setFromTriplets(tripets.begin(), tripets.end());
    return A;
}
////////////////////////////////////////////////////////////////////////////////

Eigen::SparseMatrix<double> L2_projection(
    const Eigen::MatrixXd& V_fem,
    const Eigen::MatrixXi& F_fem,
    const Eigen::MatrixXd& V_coll,
    const Eigen::MatrixXi& F_coll)
{
    Quadrature quadrature;

    std::vector<Basis> fem_bases, coll_bases;
    Basis::build_bases(V_fem, F_fem, fem_bases);
    Basis::build_bases(V_coll, F_coll, coll_bases);

    Eigen::SparseMatrix<double> M =
        compute_mass_mat(V_coll.rows(), coll_bases, quadrature);

    Eigen::SparseMatrix<double> A = compute_mass_mat_cross(
        V_fem, F_fem, fem_bases, V_coll, F_coll, coll_bases, quadrature);

    // Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
    //     solver;
    // // Compute the ordering permutation vector from the structural pattern of
    // A solver.analyzePattern(M);
    // // Compute the numerical factorization
    // solver.factorize(M);
    // // Use the factors to solve the linear system
    // Eigen::SparseMatrix<double> W = solver.solve(A);

    return A;
}