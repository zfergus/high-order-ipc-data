import argparse
import pathlib

import numpy as np
import scipy
import scipy.sparse
import meshio

import igl

from weights.barycentric import compute_barycentric_weights
from weights.mean_value import compute_mean_value_weights
from L2 import compute_L2_projection_weights
from weights.higher_order_3D import regular_2D_grid
from weights.utils import save_weights, load_weights, write_obj


def test(v_tet, f_tet, f_tri, weights):
    from scipy.spatial.transform import Rotation as R

    # T = np.array([
    #     [1, 1, 0],
    #     [0, 1, 0],
    #     [0, 0, 1],
    # ])
    # T = R.from_rotvec(np.pi * np.random.random(3)).as_matrix() @ T
    T = np.eye(3)

    deformed_v_tet = v_tet @ T.T
    mesh = meshio.Mesh(deformed_v_tet, [("tetra", f_tet)])
    mesh.write("deformed_coarse.msh", file_format="gmsh")

    deformed_v_tri = weights @ deformed_v_tet
    mesh = meshio.Mesh(deformed_v_tri, [("triangle", f_tri)])
    mesh.write("deformed_dense.obj")


def test2(coll_mesh, fem_mesh, W):
    if scipy.sparse.issparse(W):
        W_dense = W.A
    else:
        W_dense = W

    # coll_mesh.point_data = {
    #     f"w{i:02d}": W[:, i].reshape(-1, 1) for i in range(W.shape[1])
    # }

    U, Σ, Vᵀ = np.linalg.svd(W_dense, full_matrices=False)
    Σ[Σ != 0] = 1 / Σ
    Winv = Vᵀ.T @ np.diag(Σ) @ U.T

    v = coll_mesh.points.copy()
    v[0, 1] *= 0.75
    u_coll = v - coll_mesh.points
    u_fem = Winv @ u_coll
    print(Winv.shape, np.product(Winv.shape), (Winv > 0).sum())

    coll_mesh.point_data["displacement"] = u_coll
    fem_mesh.point_data["displacement"] = u_fem

    meshio.write("fem_mesh.vtu", fem_mesh)
    meshio.write("coll_mesh.vtu", coll_mesh)


def eliminate_near_zeros(A, tol=1e-12):
    A.data[np.abs(A.data) < tol] = 0
    A.eliminate_zeros()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('contact_mesh', type=pathlib.Path)
    parser.add_argument('fem_mesh', type=pathlib.Path)
    parser.add_argument('-m,--method', dest="method",
                        default="MVC", choices=["BC", "MVC", "L2"])
    parser.add_argument('--force-recompute', default=False,
                        # action=argparse.BooleanOptionalAction
                        type=bool)

    args = parser.parse_args()

    coll_mesh = meshio.read(args.contact_mesh)
    v_coll = coll_mesh.points
    assert(coll_mesh.cells[0].type == "triangle")  # Triangular mesh
    f_coll = coll_mesh.cells[0].data.astype(int)
    # Removed unreferenced vertices
    v_coll, f_coll, _, _ = igl.remove_unreferenced(v_coll, f_coll)

    fem_mesh = meshio.read(args.fem_mesh)
    v_fem = fem_mesh.points
    assert(fem_mesh.cells[0].type == "tetra")  # Tetrahedral Mesh
    f_fem = fem_mesh.cells[0].data.astype(int)

    root = pathlib.Path(__file__).parents[1]
    method2dir = {"MVC": "mean_value", "BC": "barycentric", "L2": "L2"}
    out_dir = pathlib.Path(root / "weights" / method2dir[args.method])
    out_dir.mkdir(exist_ok=True, parents=True)
    hdf5_path = (
        out_dir / f'{args.fem_mesh.stem}-to-{args.contact_mesh.stem}.hdf5')

    if args.force_recompute or not hdf5_path.exists():
        if args.method == "MVC":
            W = compute_mean_value_weights(v_coll, v_fem, f_fem, quiet=False)
        elif args.method == "BC":
            W = compute_barycentric_weights(
                v_coll, v_fem, f_fem, quiet=False)
        elif args.method == "L2":
            f_fem = igl.boundary_facets(f_fem)
            bv_fem, bf_fem, _, J = igl.remove_unreferenced(v_fem, f_fem)
            W = compute_L2_projection_weights(
                bv_fem, bf_fem, v_coll, f_coll, lump_mass_matrix=True)
            W_full = np.zeros([v_coll.shape[0], v_fem.shape[0]])
            W_full[:, J] = W.A
            W = W_full
        W = scipy.sparse.csc_matrix(W)
        eliminate_near_zeros(W)
        breakpoint()
        print(f"Saving weights to {hdf5_path}")
        save_weights(hdf5_path, W)
    else:
        print(f"Loading weights from {hdf5_path}")
        W = scipy.sparse.csc_matrix(load_weights(hdf5_path))

    # Checks error of mapping
    print("Error:", np.linalg.norm(W @ v_fem - v_coll, np.inf))

    # test(v_tet, f_tet, f_tri, W)
    # test2(coll_mesh, fem_mesh, W)


if __name__ == "__main__":
    main()
