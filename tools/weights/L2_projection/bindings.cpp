#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <igl/embree/line_mesh_intersection.h>

#include "L2_projection.hpp"

namespace py = pybind11;

PYBIND11_MODULE(L2, m)
{
    using namespace L2;

    m.doc() = "L2 Projection";

    ///////////////////////////////////////////////////////////////////////////

    m.def("hat_phi0", &hat_phi0);
    m.def("hat_phi1", &hat_phi1);
    m.def("hat_phi2", &hat_phi2);

    py::class_<Basis>(m, "Basis")
        .def(py::init())
        .def("phi", &Basis::phi)
        .def("gmapping", &Basis::gmapping)
        .def("grad_gmapping", &Basis::grad_gmapping)
        .def_readonly("n_bases", &Basis::n_bases)
        .def_readwrite("loc_2_glob", &Basis::loc_2_glob)
        .def_readwrite("v0", &Basis::v0)
        .def_readwrite("v1", &Basis::v1)
        .def_readwrite("v2", &Basis::v2);

    m.def(
        "build_bases", Basis::build_bases,
        R"L2_Qu8mg5v7(
        Build the basis.
        )L2_Qu8mg5v7",
        py::arg("V"), py::arg("F"));

    ///////////////////////////////////////////////////////////////////////////

    py::class_<Quadrature>(m, "Quadrature")
        .def(py::init())
        .def("point", &Quadrature::point)
        .def("weight", &Quadrature::weight)
        .def_readwrite("points", &Quadrature::points)
        .def_readwrite("weights", &Quadrature::weights);

    ///////////////////////////////////////////////////////////////////////////

    m.def(
        "compute_mass_mat", compute_mass_mat,
        R"L2_Qu8mg5v7(
        Compute the compute mass matrix.
        )L2_Qu8mg5v7",
        py::arg("num_nodes"), py::arg("bases"),
        py::arg("quadrature") = Quadrature());

    m.def(
        "compute_mass_mat_cross", compute_mass_mat_cross,
        R"L2_Qu8mg5v7(
        Compute the compute mass matrix cross.
        )L2_Qu8mg5v7",
        py::arg("V_fem"), py::arg("F_fem"), py::arg("fem_bases"),
        py::arg("V_coll"), py::arg("F_coll"), py::arg("coll_bases"),
        py::arg("closest_points"), py::arg("quadrature") = Quadrature());

    ///////////////////////////////////////////////////////////////////////////

    m.def(
        "build_rays", build_rays,
        R"L2_Qu8mg5v7(
        Build ray origins and directions
        )L2_Qu8mg5v7",
        py::arg("V"), py::arg("F"), py::arg("bases"),
        py::arg("quadrature_points"));

    ///////////////////////////////////////////////////////////////////////////

    m.def(
        "igl_embree_line_mesh_intersection",
        igl::embree::line_mesh_intersection<Eigen::MatrixXd, Eigen::MatrixXi>,
        R"L2_Qu8mg5v7(
        Project the point cloud V_source onto the triangle mesh
        V_target,F_target. 
        A ray is casted for every vertex in the direction specified by 
        N_source and its opposite.

        Input:
        V_source: #Vx3 Vertices of the source mesh
        N_source: #Vx3 Normals of the point cloud
        V_target: #V2x3 Vertices of the target mesh
        F_target: #F2x3 Faces of the target mesh

        Output:
        #Vx3 matrix of baricentric coordinate. Each row corresponds to 
        a vertex of the projected mesh and it has the following format:
        id b1 b2. id is the id of a face of the source mesh. b1 and b2 are 
        the barycentric coordinates wrt the first two edges of the triangle
        To convert to standard global coordinates, see barycentric_to_global.h
        )L2_Qu8mg5v7",
        py::arg("V_source"), py::arg("N_source"), py::arg("V_target"),
        py::arg("F_target"));
}
