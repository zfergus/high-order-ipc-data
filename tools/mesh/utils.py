import itertools
import numpy as np

import igl

from weights.bases import edges_3D_order, faces_3D_order  # noqa


def sorted_tuple(vals):
    return tuple(sorted(vals))


def faces(T):
    BF = igl.boundary_facets(T)
    OF = igl.oriented_facets(T)  # boundary facets + duplicated interior faces
    assert((OF.shape[0] + BF.shape[0]) % 2 == 0)
    num_faces = (OF.shape[0] + BF.shape[0]) // 2

    F = np.empty((num_faces, 3), dtype=int)
    F[:BF.shape[0]] = BF
    processed_faces = set([sorted_tuple(f) for f in BF])
    for f in OF:
        if sorted_tuple(f) not in processed_faces:
            F[len(processed_faces)] = f
            processed_faces.add(sorted_tuple(f))
    assert(F.shape[0] == len(processed_faces))
    return F


def faces_to_edges(F, E):
    if F.size == 0:
        return np.array()
    assert(E.size != 0)

    edge_to_id = {sorted_tuple(e): i for i, e in enumerate(E)}

    F2E = np.empty(F.shape, dtype=F.dtype)
    for i, f in enumerate(F):
        for j in range(f.size):
            F2E[i, j] = edge_to_id[sorted_tuple((f[j], f[(j + 1) % f.size]))]

    return F2E


def boundary_to_full(boundary, full):
    element_to_id = {
        sorted_tuple(b): i
        for i, f in enumerate(full)
        for b in itertools.combinations(f, boundary.shape[1])
    }
    return np.array([element_to_id[sorted_tuple(e)] for e in boundary])


def attach_higher_order_nodes(V, E, F, T, order):
    if T is None:
        T = F  # 2D case

    if order == 1:
        return V, T

    V_HO = [V]

    for ei, e in enumerate(E):
        v1, v2 = V[e]
        if order == 2:
            V_HO.append((v1 + v2) / 2)
        elif order == 3:
            V_HO.append(v1 + (v2 - v1) / 3)
            V_HO.append(v1 + 2 * (v2 - v1) / 3)
        else:
            raise NotImplementedError(f"P{order} not implemented!")

    for fi, f in enumerate(F):
        v1, v2, v3 = V[f]
        if order == 2:
            break
        elif order == 3:
            V_HO.append((v1 + v2 + v3) / 3)
        else:
            raise NotImplementedError(f"P{order} not implemented!")

    # for tet in T:
    #     if order == 4:
    #         V_new.append(sum(V[tet]) / 4)

    edges_per_element = 3 if V.shape[1] == 2 else 6
    faces_per_element = 1 if V.shape[1] == 2 else 4
    nodes_per_element = (
        T.shape[1] + (order - 1) * edges_per_element + (order - 2) * faces_per_element)

    T_HO = np.full((T.shape[0], nodes_per_element), -1, dtype=T.dtype)
    T_HO[:, :T.shape[1]] = T
    edge_offset = T.shape[1]
    face_offset = T.shape[1] + (order - 1) * edges_per_element

    edge_to_id = {tuple(e): i for i, e in enumerate(E)}
    face_to_id = {sorted_tuple(f): i for i, f in enumerate(F)}
    for ti, t in enumerate(T):
        for i, e in enumerate(t[edges_3D_order]):
            tmp = np.arange(order - 1)
            if tuple(e) in edge_to_id:
                ei = edge_to_id[tuple(e)]
            else:
                ei = edge_to_id[tuple(e)[::-1]]
                tmp = tmp[::-1]
            T_HO[ti, (order - 1) * i + np.arange(order - 1) +
                 edge_offset] = V.shape[0] + (order - 1) * ei + tmp
        if order >= 3:
            for i, f in enumerate(t[faces_3D_order]):
                fi = face_to_id[sorted_tuple(f)]
                T_HO[ti, i + face_offset] = (
                    V.shape[0] + (order - 1) * E.shape[0] + fi)

    assert(not (T_HO == -1).any())
    V_HO = np.vstack(V_HO)
    assert(not (T_HO >= V_HO.shape[0]).any())
    return V_HO, T_HO
