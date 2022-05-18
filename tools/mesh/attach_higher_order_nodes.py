import itertools
import numpy as np

import igl

from weights import bases
from mesh.utils import faces


def attach_higher_order_nodes(V_P1, E, F, T_P1, order):
    if order == 1:
        return V_P1, T_P1

    if order > 4:
        raise RuntimeError(
            f"attach_higher_order_nodes does not work for P{order}")

    V = [V_P1]

    for e in E:
        V.extend(bases.edge_nodes(*V_P1[e], order))
    for f in F:
        V.extend(bases.face_nodes(*V_P1[f], order))
    for tet in T_P1:
        V.extend(bases.cell_nodes(*V_P1[tet], order))

    edges_per_element = 3 if V_P1.shape[1] == 2 else 6
    faces_per_element = 1 if V_P1.shape[1] == 2 else 4

    nodes_per_edge = bases.nodes_per_edge[order]
    nodes_per_face = bases.nodes_per_face[order]
    nodes_per_cell = bases.nodes_per_cell[order]
    nodes_per_element = bases.nodes_per_element[order]

    T = np.full(
        (T_P1.shape[0], nodes_per_element), -1, dtype=T_P1.dtype)
    T[:, :T_P1.shape[1]] = T_P1

    edge_offset = T_P1.shape[1]
    face_offset = edge_offset + nodes_per_edge * edges_per_element
    cell_offset = face_offset + nodes_per_face * faces_per_element

    edge_to_id = {tuple(e): i for i, e in enumerate(E)}
    face_to_id = {tuple(f): i for i, f in enumerate(F)}
    for ti, t in enumerate(T_P1):
        for i, e in enumerate(t[bases.edges_3D_order]):
            tmp = np.arange(order - 1)
            if tuple(e) in edge_to_id:
                ei = edge_to_id[tuple(e)]
            else:
                ei = edge_to_id[tuple(e)[::-1]]
                tmp = tmp[::-1]
            T[ti, edge_offset + nodes_per_edge * i + np.arange(nodes_per_edge)] = (
                V_P1.shape[0] + nodes_per_edge * ei + tmp)

        if order >= 3:
            for i, f in enumerate(t[bases.faces_3D_order]):
                found = False
                for perm in itertools.permutations(range(3), 3):
                    perm = np.array(perm)
                    if tuple(f[perm]) in face_to_id:
                        fi = face_to_id[tuple(f[perm])]
                        found = True
                assert(found)
                T[ti, face_offset + nodes_per_face * i + np.arange(nodes_per_face)] = (
                    V_P1.shape[0]
                    + nodes_per_edge * E.shape[0]
                    + nodes_per_face * fi
                    + (0 if order == 3 else perm[bases.face_node_3D_order[order][i]]))

        if order >= 4:
            assert(order == 4)
            T[ti, cell_offset] = (V_P1.shape[0]
                                  + nodes_per_edge * E.shape[0]
                                  + nodes_per_face * F.shape[0]
                                  + ti)

    V = np.vstack(V)
    assert((T >= 0).all() and (T <= V.shape[0]).all())
    return V, T
