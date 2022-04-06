import numpy as np
import meshio
import h5py
import argparse
import pathlib
import scipy
import scipy.sparse

from weights.barycentric import compute_barycentric_weights
from weights.mean_value import compute_mean_value_weights


def save_weights(path, W):
    h5f = h5py.File(path, 'w')

    if scipy.sparse.issparse(W):
        # Saving as sparse matrix
        W_coo = W.tocoo()
        g = h5f.create_group('weight_triplets')
        g.create_dataset('values', data=W_coo.data)
        g.create_dataset('rows', data=W_coo.row)
        g.create_dataset('cols', data=W_coo.col)
        g.attrs['shape'] = W_coo.shape
    else:
        h5f.create_dataset('weights', data=W)

    h5f.close()


def load_weights(path):
    with h5py.File(path, 'r') as h5f:
        if "weight_triplets" in h5f:
            g = h5f['weight_triplets']
            return scipy.sparse.coo_matrix(
                (g['values'][:], (g['rows'][:], g['cols'][:])), g.attrs['shape']
            ).tocsc()
        else:
            assert("weights" in h5f)
            return h5f['weights']


def test(v_tet, f_tet, f_tri, weights):
    from scipy.spatial.transform import Rotation as R

    T = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    T = R.from_rotvec(np.pi * np.random.random(3)).as_matrix() @ T

    deformed_v_tet = v_tet @ T.T
    mesh = meshio.Mesh(deformed_v_tet, [("tetra", f_tet)])
    mesh.write("deformed_coarse.msh")

    deformed_v_tri = weights @ deformed_v_tet
    mesh = meshio.Mesh(deformed_v_tri, [("triangle", f_tri)])
    mesh.write("deformed_dense.obj")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dense_mesh', type=pathlib.Path)
    parser.add_argument('coarse_mesh', type=pathlib.Path)
    parser.add_argument('-m,--method', dest="method",
                        default="MVC", choices=["BC", "MVC"])
    parser.add_argument('--force-recompute', default=False,
                        # action=argparse.BooleanOptionalAction
                        type=bool)

    args = parser.parse_args()

    # Triangular mesh
    mesh = meshio.read(args.dense_mesh)
    f_tri = np.array(mesh.cells[0].data)
    v_tri = np.array(mesh.points)

    # Tetrahedral Mesh
    mesh = meshio.read(args.coarse_mesh)
    f_tet = np.array(mesh.cells[0].data)
    v_tet = np.array(mesh.points)

    root = pathlib.Path(__file__).parents[1]
    if args.method == "MVC":
        out_dir = pathlib.Path(root / "weights" / "mean_value")
    elif args.method == "BC":
        out_dir = pathlib.Path(root / "weights" / "barycentric")
    out_dir.mkdir(exist_ok=True, parents=True)
    hdf5_path = (out_dir /
                 f'{args.coarse_mesh.stem}-to-{args.dense_mesh.stem}.hdf5')

    if args.force_recompute or not hdf5_path.exists():
        if args.method == "MVC":
            W = compute_mean_value_weights(v_tri, v_tet, f_tet, quiet=False)
        elif args.method == "BC":
            W = compute_barycentric_weights(v_tri, v_tet, f_tet, quiet=False)
        W = scipy.sparse.csc_matrix(W)
        save_weights(hdf5_path, W)
    else:
        W = scipy.sparse.csc_matrix(load_weights(hdf5_path))

    # Checks error of mapping
    print("Error:", np.linalg.norm(W @ v_tet - v_tri, np.inf))

    # test(v_tet, f_tet, f_tri, W)
    # if scipy.sparse.issparse(W):
    #     W = W.A
    #
    # mesh = meshio.read(args.dense_mesh)
    # mesh.point_data = {
    #     f"w{i:02d}": W[:, i].reshape(-1, 1) for i in range(W.shape[1])
    # }
    #
    # V_fem = v_tet.copy()
    # V_fem[0] *= 0.1
    # mesh.point_data["displacement"] = np.array(
    #     W @ V_fem - mesh.points, dtype=float)
    #
    # meshio.write(f"out_{args.method.lower()}.vtu", mesh)


if __name__ == "__main__":
    main()
