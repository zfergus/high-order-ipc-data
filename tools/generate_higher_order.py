import argparse
import pathlib

import numpy as np
import scipy.sparse
import meshio

import igl

from weights.higher_order_2D import build_phi_2D
from weights.higher_order_3D import build_phi_3D
from weights.utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=pathlib.Path)
    parser.add_argument('-o,--order', dest="order", type=int, default=2,
                        choices=[2, 3])
    parser.add_argument('-m', dest="div_per_edge", type=int, default=10)

    args = parser.parse_args()

    # Triangular mesh
    mesh = meshio.read(args.mesh)

    V = mesh.points
    num_vertices = V.shape[0]

    if mesh.cells[0].type == "triangle":
        F = mesh.cells[0].data
        E = igl.edges(F)
        BE = igl.boundary_facets(F)
        BE2E = boundary_to_full(BE, E)
        BF = None
    elif mesh.cells[0].type == "tetra":
        T = mesh.cells[0].data
        F = np.sort(faces(T))
        # F = F[:, ::-1]  # Flip normals
        E = igl.edges(F)
        F2E = faces_to_edges(F, E)
        BE = None
        BF = np.sort(igl.boundary_facets(T))
        # BF = BF[:, ::-1]  # Flip normals
        BF2F = boundary_to_full(BF, F)

    # insert higher order indices at end
    V_fem = attach_higher_order_nodes(V, E, F, args.order)

    if mesh.cells[0].type == "triangle":
        # get Φ matrix
        phi, E_col = build_phi_2D(
            V_fem.shape[0], num_vertices, BE, BE2E, args.order,
            args.div_per_edge)
        F_col = None
    elif mesh.cells[0].type == "tetra":
        # get phi matrix
        phi, F_col = build_phi_3D(
            num_vertices, E.shape[0], V_fem, BF, BF2F, F2E, args.order,
            args.div_per_edge)
        E_col = None

    # compute collision vertices
    phi = scipy.sparse.csc_matrix(phi)
    V_col = phi @ V_fem

    # Check for duplicate vertices
    print("Checking for duplicate vertices... ", end="")
    _V_col, _, _, _ = igl.remove_duplicate_vertices(
        V_col, F_col if E_col is None else E_col, 1e-7)
    print(f"{V_col.shape[0] - _V_col.shape[0]} duplicate vertices found")

    ###########################################################################
    # Output

    root_dir = pathlib.Path(__file__).parents[1]

    out_weight = (root_dir / "weights" / "higher_order" /
                  f"{args.mesh.stem}-P{args.order}.hdf5")
    print(f"saving weights to {out_weight}")
    save_weights(out_weight, scipy.sparse.csc_matrix(phi), edges=E, faces=F)

    out_coll_mesh = args.mesh.parent / f"{args.mesh.stem}-collision-mesh.obj"
    print(f"saving collision mesh to {out_coll_mesh}")
    write_obj(out_coll_mesh, V_col, E=E_col, F=F_col)

    out_fem_mesh = "fem_mesh.obj"
    print(f"saving FEM mesh to {out_fem_mesh}")
    write_obj(out_fem_mesh, V_fem, E=E if BF is None else None, F=BF)


if __name__ == '__main__':
    main()
