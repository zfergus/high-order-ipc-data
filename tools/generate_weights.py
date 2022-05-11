import argparse
import pathlib

import numpy as np
import scipy
import scipy.sparse
import meshio

import igl

from weights.barycentric import compute_barycentric_weights
from weights.mean_value import compute_mean_value_weights
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


def upsample(V, F, div_per_edge):
    """Build Φ_3D"""
    upsampled_V = [V]
    downsample_buddies = [[[i, 1.0]] for i in range(V.shape[0])]
    v_count = V.shape[0]

    E = igl.edges(F)
    edge_alphas = np.linspace(0, 1, div_per_edge)[1:-1]
    for e in E:
        v0, v1 = V[e]
        for alpha in edge_alphas:
            downsample_buddies[e[0]].append([v_count, 1 - alpha])
            downsample_buddies[e[1]].append([v_count, alpha])
            upsampled_V.append((v1 - v0) * alpha + v0)
            v_count += 1

    face_alphas, _ = regular_2D_grid(div_per_edge)
    face_alphas = face_alphas[3 + 3 * (div_per_edge - 2):]
    for f in F:
        v0, v1, v2 = V[f]
        for alpha, beta in face_alphas:
            # no edge values
            assert(alpha > 0 and alpha < 1)
            assert(beta > 0 and beta < 1)
            downsample_buddies[f[0]].append([v_count, alpha])
            downsample_buddies[f[1]].append([v_count, beta])
            downsample_buddies[f[2]].append([v_count, 1 - alpha - beta])
            upsampled_V.append((v1 - v0) * alpha + (v2 - v0) * beta + v0)
            v_count += 1

    return np.vstack(upsampled_V), downsample_buddies


def downsample(W, downsample_buddies):
    W_downsampled = np.vstack([
        sum(weight * W[buddy] for buddy, weight in buddies).A
        for buddies in downsample_buddies])
    W_downsampled /= W_downsampled.sum(axis=1)[:, None]
    return W_downsampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('contact_mesh', type=pathlib.Path)
    parser.add_argument('fem_mesh', type=pathlib.Path)
    parser.add_argument('-m,--method', dest="method",
                        default="MVC", choices=["BC", "MVC"])
    parser.add_argument('--force-recompute', default=False,
                        # action=argparse.BooleanOptionalAction
                        type=bool)

    args = parser.parse_args()

    coll_mesh = meshio.read(args.contact_mesh)
    assert(coll_mesh.cells[0].type == "triangle")  # Triangular mesh
    f_coll = coll_mesh.cells[0].data.astype(int)
    v_coll = coll_mesh.points

    fem_mesh = meshio.read(args.fem_mesh)
    assert(fem_mesh.cells[0].type == "tetra")  # Tetrahedral Mesh
    f_fem = fem_mesh.cells[0].data.astype(int)
    v_fem = fem_mesh.points

    root = pathlib.Path(__file__).parents[1]
    if args.method == "MVC":
        out_dir = pathlib.Path(root / "weights" / "mean_value")
    elif args.method == "BC":
        out_dir = pathlib.Path(root / "weights" / "barycentric")
    out_dir.mkdir(exist_ok=True, parents=True)
    hdf5_path = (out_dir /
                 f'{args.fem_mesh.stem}-to-{args.contact_mesh.stem}.hdf5')
    # f'{args.contact_mesh.stem}-to-{args.fem_mesh.stem}.hdf5')

    # if(v_fem.shape[0] > v_coll.shape[0]):
    upsampled_V, downsample_buddies = upsample(
        v_coll, f_coll, 10)  # int(np.ceil(v_fem.shape[0] / v_coll.shape[0])))
    # write_obj("test.obj", upsampled_V)
    # exit(0)
    # else:
    #     upsampled_V = v_coll
    #     downsample_buddies = None

    if args.force_recompute or not hdf5_path.exists():
        if args.method == "MVC":
            W = compute_mean_value_weights(v_coll, v_fem, f_fem, quiet=False)
        elif args.method == "BC":
            W = compute_barycentric_weights(
                upsampled_V, v_fem, f_fem, quiet=False)
            if downsample_buddies is not None:
                W = downsample(W, downsample_buddies)
        W = scipy.sparse.csc_matrix(W)
        print(f"Saving weights to {hdf5_path}")
        save_weights(hdf5_path, W)
    else:
        print(f"Loading weights from {hdf5_path}")
        W = scipy.sparse.csc_matrix(load_weights(hdf5_path))

    # Checks error of mapping
    # print("Error:", np.linalg.norm(W @ v_fem - v_coll, np.inf))

    # test(v_tet, f_tet, f_tri, W)
    # test2(coll_mesh, fem_mesh, W)


if __name__ == "__main__":
    main()
