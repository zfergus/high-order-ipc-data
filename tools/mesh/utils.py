import itertools
import numpy as np

import igl


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
