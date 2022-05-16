import numpy as np
import scipy.sparse

from .bases import hat_phis_1D
from mesh.utils import faces_to_edges


def build_phi_2D(num_nodes, num_vertices, E_boundary, E_boundary_to_E_full,
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

    phi = scipy.sparse.lil_matrix((num_coll_nodes, num_nodes))
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
            val = hat_phis_1D[order][i](alpha)
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

    return phi.tocsc(), E_col


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
