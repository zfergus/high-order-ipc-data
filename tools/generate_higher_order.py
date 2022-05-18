import argparse
import pathlib

import numpy as np
import scipy.sparse
import meshio

import igl

from weights.barycentric import compute_barycentric_weights
from weights.higher_order_2D import build_phi_2D
from weights.higher_order_3D import build_phi_3D
from weights.utils import *
from mesh.fe_mesh import FEMesh
from mesh.write_obj import write_obj
from mesh.invert_gmapping import *
from weights.bases import *


def build_collision_mesh(fe_mesh, div_per_edge):
    """get Φ matrix"""
    if fe_mesh.dim() == 2:
        Phi, E_col = build_phi_2D(
            fe_mesh.n_nodes(), fe_mesh.n_vertices(), fe_mesh.BE, fe_mesh.BE2E,
            fe_mesh.order, div_per_edge)
        F_col = None
    else:
        assert(fe_mesh.dim() == 3)
        Phi, F_col = build_phi_3D(fe_mesh, div_per_edge)
        E_col = None

    # compute collision vertices
    Phi = scipy.sparse.csc_matrix(Phi)
    V_col = Phi @ fe_mesh.V

    # Check for duplicate vertices
    print("Checking for duplicate vertices... ", end="", flush=True)
    _V_col, _, _, _ = igl.remove_duplicate_vertices(
        V_col, F_col if E_col is None else E_col, 1e-7)
    print(f"{V_col.shape[0] - _V_col.shape[0]} duplicate vertices found")

    return Phi, V_col, E_col, F_col


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=pathlib.Path)
    parser.add_argument('-c,--collision-mesh',
                        dest="collision_mesh", type=pathlib.Path, default=None)
    parser.add_argument('-o,--order', dest="order",
                        help="order of the displacement",
                        type=int, default=2, choices=range(1, 5))
    parser.add_argument('-g,--geometric', dest="geom_order",
                        help="order of the geometric basis",
                        type=int, default=2, choices=range(1, 5))
    parser.add_argument('-m', dest="div_per_edge", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = pathlib.Path(__file__).parents[1]

    # Triangular mesh
    fe_mesh = FEMesh(args.mesh)
    if fe_mesh.order != args.order:
        fe_mesh.attach_higher_order_nodes(args.order)
    fe_mesh.save("fem_mesh.msh")

    if args.collision_mesh is None:
        Phi, V_col, E_col, F_col = build_collision_mesh(
            fe_mesh, args.div_per_edge)

        out_coll_mesh = (
            args.mesh.parent / f"{args.mesh.stem}-P{args.order}-collision-mesh.obj")
        print(f"saving collision mesh to {out_coll_mesh}")
        write_obj(out_coll_mesh, V_col, E=E_col, F=F_col)

        out_weight = (root_dir / "weights" / "higher_order" /
                      f"{args.mesh.stem}-P{args.order}.hdf5")
    else:
        V_col = meshio.read(args.collision_mesh).points.astype(float)
        Phi = compute_barycentric_weights(
            V_col, fe_mesh.V, fe_mesh.P1(), fe_mesh.T,
            order=fe_mesh.order)

        out_weight = (root_dir / "weights" / "higher_order" /
                      f"{args.mesh.stem}-P{args.order}-to-{args.collision_mesh.stem}.hdf5")

    print(f"saving weights to {out_weight}")
    save_weights(out_weight, Phi, vertices=fe_mesh.in_vertex_ids,
                 edges=fe_mesh.in_edges, faces=fe_mesh.in_faces)

    print("W Error:", np.linalg.norm(Phi @ fe_mesh.V - V_col, np.inf))


if __name__ == '__main__':
    main()
