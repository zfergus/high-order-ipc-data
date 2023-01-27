import numpy as np

import meshio
import igl

from weights.bases import (
    basis_order_to_gmsh,
    nodes_count_to_order,
)
from .attach_higher_order_nodes import attach_higher_order_nodes
from .utils import faces, boundary_to_full

tetra_type_to_order = {
    "tetra": 1,
    "tetra10": 2,
    "tetra20": 3,
    "tetra35": 4,
}

order_to_tetra_type = {
    1: "tetra",
    2: "tetra10",
    3: "tetra20",
    4: "tetra35",
}


class FEMesh:
    def __init__(self, filename, cell_i=0) -> None:
        mesh = meshio.read(filename)

        self.path = filename
        self.name = filename.stem

        self.V_geometry = mesh.points
        self.V_displace = mesh.points

        assert(len(mesh.cells) > cell_i)

        for cell in mesh.cells:
            if cell.type == "triangle":
                T = cell.data
                self.T_geometry = cell.data
                break
            elif "tetra" in cell.type:
                T = cell.data
                self.T_geometry = cell.data
                break
        self.T_displace = T

        *_, J = igl.remove_unreferenced(self.V, self.P1())
        self.in_vertex_ids = J
        self.in_faces = self.faces()
        self.in_edges = self.edges()

    def dim(self):
        return self.V_geometry.shape[1]

    def n_geom_nodes(self):
        return self.V_geometry.shape[0]

    def n_disp_nodes(self):
        return self.V_displace.shape[0]

    def n_vertices(self):
        return self.in_vertex_ids.size

    def n_edges(self):
        return self.E.shape[0]

    def n_faces(self):
        return self.F.shape[0]

    def P1(self):
        return self.T_displace[:, :self.dim()+1]

    def faces(self, T=None):
        return faces(T if T is not None else self.P1())

    def edges(self, T=None):
        return igl.edges(T if T is not None else self.P1())

    def geometric_order(self):
        return nodes_count_to_order[self.T_geometry.shape[1]]

    def geometric_order(self):
        return nodes_count_to_order[self.T_displace.shape[1]]

    def boundary_faces(self):
        BF = igl.boundary_facets(self.P1())
        BF2T = boundary_to_full(BF, self.P1())
        return BF, BF2T

    def attach_higher_order_nodes(self, order):
        """Insert higher order indices at end."""
        # Remove any currently attached HO nodes
        V, T, *_ = igl.remove_unreferenced(self.V_displace, self.P1())
        # Replace the nodes and tets with the HO ones
        self.V, self.T = attach_higher_order_nodes(
            V, self.edges(T), self.faces(T), T, order)

    def save(self, filename):
        print(f"saving FEM mesh to {filename}")
        meshio.write(filename, meshio.Mesh(
            self.V, [(order_to_tetra_type[self.order],
                      self.T[:, basis_order_to_gmsh[self.order]])]),
                     file_format="gmsh")
