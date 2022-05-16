import numpy as np

import meshio
import igl

from .write_obj import write_obj
from .utils import attach_higher_order_nodes, boundary_to_full, faces_to_edges, faces


class FEMesh:
    def __init__(self, filename) -> None:
        # Triangular mesh
        mesh = meshio.read(filename)

        self.V = mesh.points

        self.T = mesh.cells[0].data

        if mesh.cells[0].type == "triangle":
            self.F = self.T
            self.E = np.sort(igl.edges(self.F))
            self.BE = np.sort(igl.boundary_facets(self.F))
            self.BE2E = boundary_to_full(self.BE, self.E)
            self.BF = None
            self.BF2F = None
        elif mesh.cells[0].type == "tetra":
            self.F = np.sort(faces(self.T))
            # F = F[:, ::-1]  # Flip normals
            self.E = igl.edges(self.F)
            self.BE = None
            self.BE2E = None
            self.BF = np.sort(igl.boundary_facets(self.T))
            # BF = BF[:, ::-1]  # Flip normals
            self.BF2F = boundary_to_full(self.BF, self.F)

        self.F2E = faces_to_edges(self.F, self.E)

        self.order = 1
        self.V_HO, self.T_HO = self.V, self.T

    def dim(self):
        return self.V.shape[1]

    def n_nodes(self):
        return self.V_HO.shape[0]

    def n_vertices(self):
        return self.V.shape[0]

    def n_edges(self):
        return self.E.shape[0]

    def n_faces(self):
        return self.F.shape[0]

    def attach_higher_order_nodes(self, order):
        """Insert higher order indices at end."""
        self.order = order
        self.V_HO, self.T_HO = attach_higher_order_nodes(
            self.V, self.E, self.F, self.T, self.order)

    def save(self, filename):
        print(f"saving FEM mesh to {filename}")
        write_obj(filename, self.V_HO,
                  E=self.E if self.BF is None else None,
                  F=self.BF)
