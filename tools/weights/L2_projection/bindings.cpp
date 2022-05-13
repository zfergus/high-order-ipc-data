#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "L2_projection.hpp"

namespace py = pybind11;

PYBIND11_MODULE(L2_projection, m)
{
    m.doc() = "L2 Projection";

    m.def(
        "L2_projection", L2_projection,
        R"L2_Qu8mg5v7(
        Compute the L2 projection weight matrix.
        )L2_Qu8mg5v7",
        py::arg("V_fem"), py::arg("F_fem"), py::arg("V_coll"),
        py::arg("F_coll"));

    m.def(
        "compute_mass_mat", compute_mass_mat,
        R"L2_Qu8mg5v7(
        Compute the compute mass matrix.
        )L2_Qu8mg5v7",
        py::arg("num_nodes"), py::arg("bases"), py::arg("quadrature"));

    m.def(
        "compute_mass_mat_cross", compute_mass_mat_cross,
        R"L2_Qu8mg5v7(
        Compute the compute mass matrix cross.
        )L2_Qu8mg5v7",
        py::arg("V_fem"), py::arg("F_fem"), py::arg("fem_bases"),
        py::arg("V_coll"), py::arg("F_coll"), py::arg("coll_bases"),
        py::arg("quadrature"));

    m.def(
        "build_bases",
        [](const Eigen::MatrixXd& V,
           const Eigen::MatrixXi& F) -> std::vector<Basis> {
            std::vector<Basis> bases;
            Basis::build_bases(V, F, bases);
            return bases;
        },
        R"L2_Qu8mg5v7(
        Build the basis.
        )L2_Qu8mg5v7",
        py::arg("V"), py::arg("F"));

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

    py::class_<Quadrature>(m, "Quadrature")
        .def(py::init())
        .def("point", &Quadrature::point)
        .def("weight", &Quadrature::weight)
        .def_readwrite("points", &Quadrature::points)
        .def_readwrite("weights", &Quadrature::weights);
}
