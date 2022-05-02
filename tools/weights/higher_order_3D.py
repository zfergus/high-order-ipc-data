import argparse
import pathlib
import itertools

import numpy as np
import scipy.sparse
import meshio
import trimesh

import igl

from utils import *


hat_phis = {
    1: [
        lambda x, y: -x - y + 1,
        lambda x, y: x,
        lambda x, y: y
    ],
    2: [
        lambda x, y: (x + y - 1) * (2 * x + 2 * y - 1),
        lambda x, y: x * (2 * x - 1),
        lambda x, y: y * (2 * y - 1),
        lambda x, y: -4 * x * (x + y - 1),
        lambda x, y: 4 * x * y,
        lambda x, y: -4 * y * (x + y - 1),
    ],
    3: [
        lambda x, y: (-27.0 / 2.0 * x**2 * y + 9 * x**2 - 27.0 / 2.0 * y**2 * x + 9 * y**2 -
                      9.0 / 2.0 * x**3 + 18 * x * y - 11.0 / 2.0 * x - 9.0 / 2.0 * y**3 - 11.0 / 2.0 * y + 1),
        lambda x, y: (1.0 / 2.0) * x * (9 * x**2 - 9 * x + 2),
        lambda x, y: (1.0 / 2.0) * y * (9 * y**2 - 9 * y + 2),
        lambda x, y: (9.0 / 2.0) * x * (x + y - 1) * (3 * x + 3 * y - 2),
        lambda x, y: -9.0 / 2.0 * x * (3 * x**2 + 3 * x * y - 4 * x - y + 1),
        lambda x, y: (9.0 / 2.0) * x * y * (3 * x - 1),
        lambda x, y: (9.0 / 2.0) * x * y * (3 * y - 1),
        lambda x, y: (9.0 / 2.0) * y * (x + y - 1) * (3 * x + 3 * y - 2),
        lambda x, y: -9.0 / 2.0 * y * (3 * x * y - x + 3 * y**2 - 4 * y + 1),
        lambda x, y: -27 * x * y * (x + y - 1),
    ]
}


def regular_2d_grid(n):
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
    V_grid, F_grid = regular_2d_grid(div_per_edge)
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
            phi[rows, node_i] = hat_phis[order][i](alphas, betas)

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


def tet_edges(tet):
    yield tet[[0, 1]]
    yield tet[[1, 2]]
    yield tet[[2, 0]]
    yield tet[[0, 3]]
    yield tet[[1, 3]]
    yield tet[[2, 3]]


def tet_faces(tet):
    yield tet[[0, 1, 2]]
    yield tet[[0, 1, 3]]
    yield tet[[1, 2, 3]]
    yield tet[[2, 0, 3]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=pathlib.Path)
    parser.add_argument('-o,--order', dest="order", type=int, default=2,
                        choices=[2, 3])
    parser.add_argument('-m', dest="div_per_edge", type=int, default=10)

    args = parser.parse_args()

    # Triangular mesh
    mesh = meshio.read(args.mesh)
    assert(mesh.cells[0].type == "tetra")
    T = np.array(mesh.cells[0].data)
    V = np.array(mesh.points)
    num_vertices = V.shape[0]

    order = int(args.order)
    div_per_edge = int(args.div_per_edge)

    F = np.sort(faces(T))
    # F = F[:, ::-1]  # Flip normals
    E = igl.edges(F)
    F2E = faces_to_edges(F, E)
    F_boundary = np.sort(igl.boundary_facets(T))
    # F_boundary = F_boundary[:, ::-1]  # Flip normals
    F_boundary_to_F_full = boundary_to_full(F_boundary, F)

    # insert higher order indices at end
    V_fem = attach_higher_order_nodes(V, E, F, order)

    # get phi matrix
    phi, F_col = build_phi_3D(
        num_vertices, E.shape[0], V_fem, F_boundary, F_boundary_to_F_full, F2E, order, div_per_edge)

    # compute collision vertices
    phi = scipy.sparse.csc_matrix(phi)
    V_col = phi @ V_fem

    # Removing duplicate rows (kind of hacky)
    print("Checking for duplicate vertices... ", end="")
    n_vertices_before = V_col.shape[0]
    _V_col, _, _, _F_col = igl.remove_duplicate_vertices(V_col, F_col, 1e-7)
    print(f"{n_vertices_before - _V_col.shape[0]} duplicate vertices found")

    # ordering = polyfem_ordering_3D(V_fem.shape[0], E.shape[0], T, order)

    root_dir = pathlib.Path(__file__).parents[2]

    out_weight = (root_dir / "weights" / "higher_order" /
                  f"{args.mesh.stem}-P{order}.hdf5")
    print(f"saving weights to {out_weight}")
    save_weights(out_weight, phi, edges=E, faces=F)

    out_coll_mesh = args.mesh.parent / f"{args.mesh.stem}-collision-mesh.obj"
    print(f"saving collision mesh to {out_coll_mesh}")
    write_obj(out_coll_mesh, V_col, F=F_col)

    out_fem_mesh = "fem_mesh.obj"
    print(f"saving FEM mesh to {out_fem_mesh}")
    write_obj(out_fem_mesh, V_fem, F=F_boundary)


if __name__ == '__main__':
    main()
