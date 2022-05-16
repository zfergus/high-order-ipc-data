import itertools

import numpy as np
import scipy.sparse
import trimesh

import igl

from .bases import hat_phis_2D
from .utils import labeled_tqdm
from mesh.utils import faces_to_edges


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


def build_phi_3D(num_vertices, num_edges, V, BF_in, BF2F, F2E, order, div_per_edge):
    """Build Φ_3D"""
    V_grid, F_grid = regular_2D_grid(div_per_edge)
    alphas = V_grid[:, 0]
    betas = V_grid[:, 1]

    num_nodes = V.shape[0]
    n_nodes_per_edge = order - 1

    BV, BF, _, J = igl.remove_unreferenced(V, BF_in)
    BE = igl.edges(BF)
    BF2BE = faces_to_edges(BF, BE)
    nBV = BV.shape[0]
    nBE = BE.shape[0]
    nBF = BF.shape[0]

    n_grid_boundary_vertices = 3 + 3 * (div_per_edge - 2)
    n_grid_interior_vertices = V_grid.shape[0] - n_grid_boundary_vertices
    num_coll_vertices = (
        nBV + (div_per_edge - 2) * nBE + nBF * n_grid_interior_vertices)

    # The order of Φ's rows will be: corners then edge interior then face interior
    phi = scipy.sparse.lil_matrix((num_coll_vertices, num_nodes))

    for fi, f in enumerate(labeled_tqdm(BF_in, "Building Φ")):
        # Construct a list of nodes associated with face f
        nodes = f.tolist()
        # Edge nodes
        offset = num_vertices
        delta = n_nodes_per_edge
        for ei in F2E[BF2F[fi]]:
            nodes.extend(offset + ei * delta + np.arange(delta))
        # Face nodes
        if order == 3:
            offset += delta * num_edges
            nodes.append(offset + BF2F[fi])

        # Map from local collision vertices to global
        rows = BF[fi].tolist()
        offset = nBV
        delta = div_per_edge - 2
        for ei in BF2BE[fi]:
            rows.extend(offset + ei * delta + np.arange(delta))
        offset += delta * nBE
        delta = n_grid_interior_vertices
        rows.extend(offset + fi * delta + np.arange(delta))

        # evaluate ϕᵢ on the face nodes
        for i, node_i in enumerate(nodes):
            phi[rows, node_i] = hat_phis_2D[order][i](alphas, betas)

    phi = phi.tocsc()
    V_col = phi @ V

    # Stitch faces together
    F_col = []
    for fi, f in enumerate(labeled_tqdm(BF, "Building collision mesh")):
        v0, v1, v2 = BV[f]
        f_coll = F_grid.copy()
        for i, vi in np.ndenumerate(f_coll):
            if vi < 3:
                f_coll[i] = f[vi]
            elif vi < n_grid_boundary_vertices:
                offset = nBV
                delta = div_per_edge - 2
                local_ei = (vi - 3) // (div_per_edge - 2)
                ei = BF2BE[fi, local_ei]
                # ej = (vi - 3) % delta

                point = alphas[vi] * (v1 - v0) + betas[vi] * (v2 - v0) + v0
                ej = np.linalg.norm(
                    V_col[offset + ei * delta + np.arange(delta)] - point,
                    axis=1).argmin()

                f_coll[i] = offset + ei * delta + ej
            else:
                offset = nBV + nBE * (div_per_edge - 2)
                delta = n_grid_interior_vertices
                f_coll[i] = (
                    offset + fi * delta + (vi - n_grid_boundary_vertices))
        F_col.append(f_coll)
    F_col = np.vstack(F_col)

    # Fix face orientation
    mesh = trimesh.Trimesh(V_col, F_col)
    trimesh.repair.fix_normals(mesh)
    F_col = mesh.faces

    return phi, F_col
