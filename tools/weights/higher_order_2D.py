import numpy as np
import igl
import meshio
import argparse
import pathlib
import scipy.sparse
import itertools

from utils import (
    save_weights,
    write_obj,
    boundary_to_full,
    attach_higher_order_nodes,
    faces_to_edges
)


hat_phis = {
    1: [lambda x: 1 - x, lambda x: x],
    2: [
        lambda x: 2 * (x - 0.5) * (x - 1),
        lambda x: 2 * (x - 0) * (x - 0.5),
        lambda x: -4 * (x - 0.5)**2 + 1
    ],
    3: [
        lambda x: -9 / 2 * (x - 1 / 3) * (x - 2 / 3) * (x - 1),
        lambda x: 9 / 2 * (x - 0) * (x - 1 / 3) * (x - 2 / 3),
        lambda x: 27 / 2 * (x - 0) * (x - 2 / 3) * (x - 1),
        lambda x: -27 / 2 * (x - 0) * (x - 1 / 3) * (x - 1),
    ]
}


def get_phi_2d(num_nodes, num_vertices, E_boundary, E_boundary_to_E_full,
               order, div_per_edge):
    assert(div_per_edge > 2)
    alpha = np.linspace(0, 1, div_per_edge)
    alpha = alpha[1:-1]

    n_el = E_boundary.shape[0]
    num_boundary_vertices = len(set(E_boundary.flatten()))
    num_coll_nodes = num_boundary_vertices + n_el * (div_per_edge - 2)
    n_edge_nodes = order - 1

    E_col = np.empty((n_el * (div_per_edge - 1), 2), dtype=int)

    V_edge_to_V_collision = {}
    for e in E_boundary:
        for i in e:
            if i not in V_edge_to_V_collision:
                V_edge_to_V_collision[i] = len(V_edge_to_V_collision)
    assert(len(V_edge_to_V_collision) == num_boundary_vertices)

    phi = np.zeros((num_coll_nodes, num_nodes))
    for fem_i, coll_i in V_edge_to_V_collision.items():
        phi[coll_i, fem_i] = 1

    start_vi = num_boundary_vertices
    delta_vi = alpha.size
    start_ei = 0
    delta_ei = (div_per_edge - 1)
    for ei, e in enumerate(E_boundary):
        vertex_nodes = e.tolist()
        edge_nodes = (
            num_vertices + E_boundary_to_E_full[ei] *
            n_edge_nodes + np.arange(n_edge_nodes)
        ).tolist()
        nodes = vertex_nodes + edge_nodes

        for i in range(order + 1):
            val = hat_phis[order][i](alpha)
            ind = nodes[i]
            phi[start_vi:(start_vi + delta_vi), ind] = val

        E_col[start_ei] = [V_edge_to_V_collision[e[0]], start_vi]
        start_ei += 1
        for i in range(delta_ei - 2):
            E_col[start_ei + i] = [start_vi + i, start_vi + i + 1]
        start_ei += delta_ei - 2
        E_col[start_ei] = [start_vi + delta_vi - 1, V_edge_to_V_collision[e[1]]]
        start_ei += 1

        start_vi += delta_vi

    return phi, E_col


def polyfem_ordering_2D(num_vertices, E, F, order):
    ordering = []
    F2E = faces_to_edges(F, E)
    processed_vertices = set()
    processed_edges = set()
    num_nodes_per_edge = order - 1
    for fi, f in enumerate(F):
        for vi in f:
            if vi not in processed_vertices:
                ordering.append(vi)
                processed_vertices.add(vi)
        for i in range(3):
            ei = F2E[fi, i]
            if ei not in processed_edges:
                ordering.extend(num_vertices + num_nodes_per_edge * ei
                                + np.arange(num_nodes_per_edge))
                processed_edges.add(ei)
        if order == 3:
            ordering.append(num_vertices + E.shape[0] + fi)
    return ordering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=pathlib.Path)
    parser.add_argument('-o,--order', dest="order", type=int,
                        default="2", choices=[2, 3])
    parser.add_argument('-m', dest="div_per_edge", type=int,
                        default=10)

    args = parser.parse_args()

    # Triangular mesh
    mesh = meshio.read(args.mesh)
    assert(mesh.cells[0].type == "triangle")
    F = mesh.cells[0].data
    V = mesh.points
    num_vertices = V.shape[0]

    E_full = igl.edges(F)
    E_boundary = igl.boundary_facets(F)
    E_boundary_to_E_full = boundary_to_full(E_boundary, E_full)

    # insert higher order indices at end
    V = attach_higher_order_nodes(V, E_full, F, args.order)

    # get phi matrix
    phi, E_col = get_phi_2d(V.shape[0], num_vertices, E_boundary,
                            E_boundary_to_E_full, args.order, args.div_per_edge)

    ordering = polyfem_ordering_2D(num_vertices, E_full, F, args.order)
    assert(len(ordering) == phi.shape[1])
    phi = phi[:, ordering]
    V = V[ordering]
    for i, e in enumerate(E_full):
        for j, v in enumerate(e):
            E_full[i, j] = ordering.index(v)

    # center = np.zeros(3)
    # radius = 0.5
    # for i in range(E_boundary.shape[0]):
    #     for j in range(args.order - 1):
    #         vi = num_vertices + (args.order - 1) * boundary_to_full[i] + j
    #         print(vi)
    #         point = V[vi]
    #         center_to_point = point - center
    #         center_to_point /= np.linalg.norm(center_to_point)
    #         V[vi] = center + center_to_point * radius

    save_weights("phi.hdf5", scipy.sparse.csc_matrix(phi))
    write_obj("fem_mesh.obj", V, E_full)

    # compute collision matrix
    V_col = phi @ V

    # test code to visualise
    write_obj("coll_mesh.obj", V_col, E_col)


if __name__ == '__main__':
    main()
