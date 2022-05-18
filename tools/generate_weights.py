import argparse
import pathlib

import numpy as np
import scipy
import scipy.sparse
import meshio

import igl

from weights.barycentric import compute_barycentric_weights
from weights.mean_value import compute_mean_value_weights
from weights.L2 import compute_L2_projection_weights
from weights.utils import save_weights, load_weights


def eliminate_near_zeros(A, tol=1e-12):
    A.data[np.abs(A.data) < tol] = 0
    A.eliminate_zeros()


def density(A):
    return f"{(A.nnz / np.product(A.shape)) * 100:.2f}%"


def compute_pseudoinverse(W, tol=1e-6):
    W_dense = W if not scipy.sparse.issparse(W) else W.A

    U, Σ, Vᵀ = np.linalg.svd(W_dense, full_matrices=False)

    Σ[abs(Σ) <= tol] = 0
    Σ[Σ != 0] = 1 / Σ[Σ != 0]

    Winv = Vᵀ.T @ np.diag(Σ) @ U.T

    Winv = scipy.sparse.csc_matrix(Winv)
    eliminate_near_zeros(Winv)

    print(np.linalg.norm((W @ Winv @ W - W).A))

    return Winv


def remove_unreferenced_vertices(mesh, filename=None):
    V = mesh.points
    assert(len(mesh.cells) == 1)
    F = mesh.cells[0].data.astype(int)
    # Removed unreferenced vertices
    V, F, _, J = igl.remove_unreferenced(V, F)
    unref_removed = mesh.points.shape[0] - V.shape[0]
    assert(unref_removed != 0 or (F - mesh.cells[0].data).sum() == 0)
    print(f"Found and removed {unref_removed} unreferenced vertices")
    mesh.points = V
    mesh.cells[0].data = F.astype("int32")
    if filename is not None and unref_removed != 0:
        print(f"Saving unreference free mesh to {filename}")
        if filename.suffix == ".msh":
            mesh.write(filename, file_format="gmsh22")
        else:
            mesh.write(filename)
    return J


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('coarse_mesh', type=pathlib.Path)
    parser.add_argument('fine_mesh', type=pathlib.Path)
    parser.add_argument('-m,--method', dest="method",
                        default="BC", choices=["BC", "MVC", "L2", "identity"])
    parser.add_argument('-f,--force-recompute', action="store_true",
                        default=False, dest="force_recompute")
    return parser.parse_args()


def read_mesh(filename):
    print(f"Loading {filename}")
    mesh = meshio.read(filename)
    # remove_unreferenced_vertices(mesh, filename)
    V = mesh.points
    F = []
    cells_type = None
    for cells in mesh.cells:
        if cells_type == None and cells.type == "triangle":
            cells_type = cells.type
            F = [cells.data.astype(int)]
        elif cells.type == "tetra":
            if cells_type == "tetra":
                F.append(cells.data.astype(int))
            else:
                F = [cells.data.astype(int)]
    assert(len(F) > 0)
    F = np.vstack(F)

    if cells_type == "tetra":
        BF = igl.boundary_facets(F)
        bmesh = meshio.Mesh(V, [("triangle", BF)])
        BV2V = remove_unreferenced_vertices(
            bmesh, filename.with_suffix(".ply"))
        BV = bmesh.points
        BF = bmesh.cells[0].data.astype(int)
    else:
        BV2V = None
        BV = V
        BF = F

    return BV, BF, BV2V, V, F


def compute_W(BV_from, BF_from, V_from, F_from, BV2V_from,
              BV_to, BF_to, method):
    if method == "MVC":
        W = compute_mean_value_weights(BV_to, V_from, F_from, quiet=False)
    elif method == "BC":
        W = compute_barycentric_weights(BV_to, V_from, F_from, quiet=False)
    elif method == "L2":
        W = compute_L2_projection_weights(
            BV_to, BF_to, BV_from, BF_from, lump_mass_matrix=False)
        W_full = scipy.sparse.lil_matrix((BV_to.shape[0], V_from.shape[0]))
        W_full[:, BV2V_from] = W
        W = W_full
    elif method == "identity":
        assert(BV_from.shape == BV_to.shape)
        W = scipy.sparse.eye(BV_from.shape[0])
        W_full = scipy.sparse.lil_matrix((BV_to.shape[0], V_from.shape[0]))
        W_full[:, BV2V_from] = W
        W = W_full

    W = scipy.sparse.csc_matrix(W)
    eliminate_near_zeros(W)

    return W


def main():
    args = parse_args()

    BV_coarse, BF_coarse, BV2V_coarse, V_coarse, F_coarse = read_mesh(
        args.coarse_mesh)
    assert(F_coarse.shape[1] == 4)

    BV_fine, BF_fine, BV2V_fine, V_fine, F_fine = read_mesh(args.fine_mesh)

    root = pathlib.Path(__file__).parents[1]
    method2dir = {"MVC": "mean_value", "BC": "barycentric",
                  "L2": "L2", "identity": "identity"}
    out_dir = pathlib.Path(root / "weights" / method2dir[args.method])
    out_dir.mkdir(exist_ok=True, parents=True)

    hdf5_path = (
        out_dir / f'{args.coarse_mesh.stem}-to-{args.fine_mesh.stem}.hdf5')

    if args.force_recompute or not hdf5_path.exists():
        W = compute_W(BV_coarse, BF_coarse, V_coarse, F_coarse, BV2V_coarse,
                      BV_fine, BF_fine, args.method)
        print(f"Saving W to {hdf5_path}")
        save_weights(hdf5_path, W, V_coarse.shape[0])
    else:
        print(f"Loading W from {hdf5_path}")
        W = scipy.sparse.csc_matrix(load_weights(hdf5_path))

    # Checks error of mapping
    print(f"density(W)={density(W)} W.nnz={W.nnz}")
    print("W Error:", np.linalg.norm(W @ V_coarse - BV_fine, np.inf))

    if F_fine.shape[1] == 4:
        hdf5_path = (
            out_dir / f'{args.fine_mesh.stem}-to-{args.coarse_mesh.stem}.hdf5')
        if args.force_recompute or not hdf5_path.exists():
            Winv = compute_W(BV_fine, BF_fine, V_fine, F_fine, BV2V_fine,
                             BV_coarse, BF_coarse, args.method)
            # W_full = compute_W(BV_coarse, BF_coarse, V_coarse, F_coarse, BV2V_coarse,
            #                    V_fine, F_fine, args.method)
            # Winv = compute_pseudoinverse(W_full)[BV2V_coarse]

            print(f"Saving W⁻¹ to {hdf5_path}")
            save_weights(hdf5_path, Winv, V_fine.shape[0])
        else:
            print(f"Loading W⁻¹ from {hdf5_path}")
            Winv = scipy.sparse.csc_matrix(load_weights(hdf5_path))

        print(f"density(W⁻¹)={density(Winv)} W⁻¹.nnz={Winv.nnz}")
        print("W⁻¹ Error:", np.linalg.norm(Winv @ V_fine - BV_coarse, np.inf))


if __name__ == "__main__":
    main()
