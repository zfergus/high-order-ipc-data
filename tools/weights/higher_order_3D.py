import itertools
import pathlib

import numpy as np
import scipy.sparse
import trimesh

import igl

from .bases import *
from .utils import labeled_tqdm
from mesh.utils import boundary_to_full, faces_to_edges, sorted_tuple, faces
from mesh.write_obj import write_obj


def regular_2D_grid(n):
    delta = 1 / (n - 1)
    # map from (i, j) coordinates to vertex id
    ij2v = np.full((n, n), -1)
    V = []

    # Corner Vertices
    V.extend([[0, 0], [1, 0], [0, 1]])
    ij2v[0, 0] = 0
    ij2v[-1, 0] = 1
    ij2v[0, -1] = 2

    # Edge vertices
    # [0, 0] -> [1, 0]
    ij2v[1:-1, 0] = np.arange(n - 2) + 3
    V.extend([[x, 0] for x in np.linspace(0, 1, n)[1:-1]])
    # [1, 0] -> [0, 1]
    ij2v[np.arange(n - 2, 0, -1), np.arange(1, n - 1)] = (
        np.arange(n - 2) + (3 + n - 2))
    V.extend([[1 - x, x] for x in np.linspace(0, 1, n)[1:-1]])
    # [0, 1] -> [0, 0]
    ij2v[0, -2:0:-1] = np.arange(n - 2) + (3 + 2 * (n - 2))
    V.extend([[0, 1 - y] for y in np.linspace(0, 1, n)[1:-1]])

    # Interior vertices
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            if i + j >= n - 1:
                break
            ij2v[i, j] = len(V)
            V.append([i * delta, j * delta])

    # Create triangulated faces
    F = []
    for i, j in itertools.product(range(n - 1), range(n - 1)):
        for f in [ij2v[[i, i + 1, i], [j, j, j + 1]],
                  ij2v[[i + 1, i + 1, i], [j, j + 1, j + 1]]]:
            if (f >= 0).all():
                F.append(f)

    return np.array(V), np.array(F)


def find_row_in_array(A, a):
    for i, ai in enumerate(A):
        if sorted_tuple(a) == sorted_tuple(ai):
            return i
    raise ValueError("unable to find row")


def uv_to_uvw(u, v, fi):
    assert(fi >= 0 and fi < 4)
    if fi == 0:
        return u, v, np.zeros_like(u)
    elif fi == 1:
        return u, np.zeros_like(u), v
    elif fi == 2:
        return 1 - u - v, u, v
    elif fi == 3:
        return np.zeros_like(u), u, v


def build_phi_3D(mesh, div_per_edge):
    """Build Φ_3D"""
    V_grid, F_grid = regular_2D_grid(div_per_edge)
    alphas, betas = np.hsplit(V_grid, 2)
    n_grid_v = V_grid.shape[0]
    n_grid_f = F_grid.shape[0]
    BE_grid = igl.boundary_facets(F_grid)
    n_grid_be = BE_grid.shape[0]

    # Boundary in full ids
    BF = igl.boundary_facets(mesh.P1())
    BF2T = boundary_to_full(BF, mesh.P1())

    # The order of Φ's rows will be: corners then edge interior then face interior
    phi = scipy.sparse.lil_matrix((n_grid_v * BF.shape[0], mesh.n_nodes()))

    # Faces of the collision mesh (initially not stitched together)
    F_coll = np.zeros((n_grid_f * BF.shape[0], 3), dtype=int)

    # Edges cooresponding to the edges on the higher order mesh
    E_higher = np.zeros((n_grid_be * BF.shape[0], 2), dtype=int)

    for fi, f in enumerate(labeled_tqdm(BF, "Building Φ")):
        nodes = mesh.T[BF2T[fi]]
        fi_in_tet = find_row_in_array(nodes[:4][faces_3D_order], f)
        assert(len(nodes) == len(hat_phis_3D[mesh.order]))

        rows = np.arange(fi * n_grid_v, (fi + 1) * n_grid_v)

        for i, node_i in enumerate(nodes):
            phi[rows, node_i] = hat_phis_3D[mesh.order][i](
                *uv_to_uvw(alphas, betas, fi_in_tet))

        F_coll[fi * n_grid_f:(fi + 1) * n_grid_f] = (
            F_grid.copy() + fi * n_grid_v)

        E_higher[fi * n_grid_be:(fi+1) * n_grid_be] = (
            BE_grid.copy() + fi * n_grid_v)

    # Compute V_coll to remove duplicates
    phi = phi.tocsc()
    V_coll = phi @ mesh.V

    # Stitch faces together
    print("Building collision mesh")
    ordering_filename = (
        mesh.path.parent / f"order_mesh={mesh.name}_m={div_per_edge}.npz")
    try:
        print(f"\tTrying to loading unique order {ordering_filename}")
        ordering = np.load(ordering_filename)
        indices = ordering["indices"]
        inverse = ordering["inverse"]

        F_coll = np.hstack([inverse[F_coll[:, j]].reshape(-1, 1)
                            for j in range(F_coll.shape[1])])
        print("\tSuccess")
    except Exception as e:
        print(f"\tFailed to load from file because {e}")
        print("\tCreating unique order")
        _, indices, inverse, _ = igl.remove_duplicate_vertices(
            V_coll, F_coll, epsilon=1e-7)
        # _, indices, inverse = (
        #     np.unique(V_coll, return_index=True, return_inverse=True, axis=0))
        indices, arg_indices = np.sort(indices), np.argsort(indices)
        for i, inv in enumerate(inverse):
            new_inv = np.where(inv == arg_indices)[0]
            assert(new_inv.size == 1)
            inverse[i] = new_inv[0]

        print(f"\tSaving unique order to {ordering_filename}")
        np.savez(ordering_filename, indices=indices, inverse=inverse)

        F_coll = np.hstack([inverse[F_coll[:, j]][:, None]
                            for j in range(F_coll.shape[1])])

    V_coll = V_coll[indices]

    print("Updating Φ")
    new_phi = phi[indices]
    assert(abs(new_phi[inverse] - phi).max() < 1e-13)
    phi = new_phi.tocsc()
    assert(np.linalg.norm(phi @ mesh.V - V_coll, ord=np.Inf) < 1e-13)

    E_higher = np.hstack([inverse[E_higher[:, j]].reshape(-1, 1)
                          for j in range(E_higher.shape[1])])
    E_higher = np.unique(np.sort(E_higher), axis=0)

    # Fix face orientation
    print("Checking face orientation")
    mesh = trimesh.Trimesh(V_coll, F_coll)
    trimesh.repair.fix_normals(mesh)
    F_coll = mesh.faces

    print(f"|F|={F_coll.shape} |V|={V_coll.shape} |Φ|={phi.shape}")

    return phi, F_coll, E_higher
