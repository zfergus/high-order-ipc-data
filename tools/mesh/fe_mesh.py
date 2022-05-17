import numpy as np

import meshio
import igl

from weights.bases import gmsh_to_basis_order, basis_order_to_gmsh
from .attach_higher_order_nodes import attach_higher_order_nodes
from .utils import faces
from .write_obj import write_obj

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
        # Triangular mesh
        mesh = meshio.read(filename)

        self.V = mesh.points

        # breakpoint()
        assert(len(mesh.cells) > cell_i)
        cell_type = mesh.cells[cell_i].type
        self.order = tetra_type_to_order[cell_type]
        if cell_type == "triangle":
            self.T = mesh.cells[cell_i].data
        elif "tetra" in cell_type:
            self.T = (
                mesh.cells[cell_i].data[:, gmsh_to_basis_order[self.order]])
        else:
            raise NotImplementedError(
                f"FEMesh not implemented for {cell_type}")

        self.T = self.T.reshape(1, -1)

        self._n_vertices = len(set(self.P1().flatten()))

        if self.order != 1:
            self.save("before.msh")
            # Reorder the nodes to be [V; E; F; C] and set to input positions
            self.attach_higher_order_nodes(self.order)

    def dim(self):
        return self.V.shape[1]

    def n_nodes(self):
        return self.V.shape[0]

    def n_vertices(self):
        return self._n_vertices

    def n_edges(self):
        return self.E.shape[0]

    def n_faces(self):
        return self.F.shape[0]

    def P1(self):
        return self.T[:, :self.dim()+1]

    def attach_higher_order_nodes(self, order):
        """Insert higher order indices at end."""
        if order == self.order:
            V_old = self.V.copy()
            T_old = self.T.copy()
        # Remove any currently attached HO nodes
        V, T, IM, J = igl.remove_unreferenced(self.V, self.P1())
        T = T.reshape(1, 4)
        # Compute the edges in faces in a fixed order
        F = faces(T)
        F = np.sort(F)
        E = igl.edges(F)
        # Replace the nodes and tets with the HO ones
        self.V, self.T = attach_higher_order_nodes(V, E, F, T, order)
        assert((self.V[:V.shape[0]] == V).all())
        # self.F = faces(self.P1())
        self.F = faces(self.P1())
        self.F = np.sort(self.F)
        self.E = igl.edges(self.F)
        if order == self.order:
            self.save("test_before.msh")
            V_new = self.V.copy()
            for i, vi_new in np.ndenumerate(self.T):
                assert(i[1] > 3 or (self.V[vi_new] == V_old[T_old[i]]).all())
                V_new[vi_new] = V_old[T_old[i]]
            self.V = V_new
            breakpoint()
            self.save("test.msh")
        self.order = order

    def save(self, filename):
        print(f"saving FEM mesh to {filename}")
        meshio.write(filename, meshio.Mesh(
            self.V, [(order_to_tetra_type[self.order],
                      self.T[:, basis_order_to_gmsh[self.order]])]),
                     file_format="gmsh")
