import itertools

import numpy as np
import scipy.sparse
import trimesh

import igl


from .bases import *
from .utils import labeled_tqdm
from mesh.utils import boundary_to_full, faces_to_edges, sorted_tuple, faces
from mesh.invert_gmapping import sample_tet
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


def build_phi_3D(mesh, div_per_edge):
    """Build Φ_3D"""
    V_grid, F_grid = regular_2D_grid(div_per_edge)
    alphas, betas = numpy.hsplit(V_grid, 2)

    # Boundary in full ids
    BF_full = np.sort(igl.boundary_facets(mesh.P1()))
    BF_full = np.sort(BF_full)

    # Boundary in local ids
    BV, BF, *_ = igl.remove_unreferenced(mesh.V, BF_full)
    BE = igl.edges(BF)
    BF2BE = faces_to_edges(BF, BE)
    nBV = BV.shape[0]
    nBE = BE.shape[0]
    nBF = BF.shape[0]

    n_grid_boundary_vertices = 3 + 3 * (div_per_edge - 2)
    n_grid_interior_vertices = V_grid.shape[0] - n_grid_boundary_vertices
    num_coll_vertices = (
        nBV + (div_per_edge - 2) * nBE + nBF * n_grid_interior_vertices)

    BF2F = boundary_to_full(BF_full, mesh.F)
    F2E = faces_to_edges(mesh.F, mesh.E)

    # The order of Φ's rows will be: corners then edge interior then face interior
    phi = scipy.sparse.lil_matrix((num_coll_vertices, mesh.n_nodes()))

    for fi, f in enumerate(labeled_tqdm(BF_full, "Building Φ")):
        # Construct a list of nodes associated with face f
        nodes = f.tolist()
        # Edge nodes
        offset = mesh.n_vertices()
        delta = nodes_per_edge[mesh.order]
        for ei in F2E[BF2F[fi]]:
            nodes.extend(offset + ei * delta + np.arange(delta))
        # Face nodes
        if mesh.order >= 3:
            offset += delta * mesh.n_edges()
            delta = nodes_per_face[mesh.order]
            nodes.extend(offset + fi * delta + np.arange(delta))
        # Cell nodes (not needed for 2D basis)

        # Map from local collision vertices to global
        rows = BF[fi].tolist()
        offset = nBV
        delta = div_per_edge - 2
        for ei in BF2BE[fi]:
            rows.extend(offset + ei * delta + np.arange(delta))
        offset += delta * nBE
        delta = n_grid_interior_vertices
        rows.extend(offset + fi * delta + np.arange(delta))

        assert(len(nodes) == len(hat_phis_2D[mesh.order]))
        for i, node_i in enumerate(nodes):
            phi[rows, node_i] = hat_phis_2D[mesh.order][i](alphas, betas)

    phi = phi.tocsc()
    V_col = phi @ mesh.V

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
                    V_col[offset + ei * delta + np.arange(delta)] - point, axis=1).argmin()

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
    write_obj("test.obj", V_col, F=F_col)

    return phi, numpy.array([], dtype=int)
