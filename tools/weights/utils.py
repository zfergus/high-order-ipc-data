import itertools
import numpy as np
import scipy.sparse
import h5py
import igl
from tqdm import tqdm


def quiet_tqdm(x, quiet):
    return x if quiet else tqdm(x)


def labeled_tqdm(data, label):
    pbar = tqdm(data)
    pbar.set_description(label)
    return pbar


def save_weights(path, W):
    h5f = h5py.File(path, 'w')

    if scipy.sparse.issparse(W):
        # Saving as sparse matrix
        W_coo = W.tocoo()
        g = h5f.create_group('weight_triplets')
        g.create_dataset('values', data=W_coo.data)
        g.create_dataset('rows', data=W_coo.row)
        g.create_dataset('cols', data=W_coo.col)
        g.attrs['shape'] = W_coo.shape
    else:
        h5f.create_dataset('weights', data=W)

    h5f.close()


def write_obj(filename, V, E=None, F=None):
    with open(filename, 'w') as f:
        for v in V:
            f.write("v {:.16f} {:.16f} {:.16f}\n".format(*v))
        if E is not None:
            for e in E:
                f.write("l {:d} {:d}\n".format(*(e + 1)))
        if F is not None:
            for face in F:
                f.write("f {:d} {:d} {:d}\n".format(*(face + 1)))


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
        return numpy.array()
    assert(E.size != 0)

    edge_to_id = {sorted_tuple(e): i for i, e in enumerate(E)}

    F2E = np.empty(F.shape, dtype=F.dtype)
    for i, f in enumerate(F):
        for j in range(f.size):
            F2E[i, j] = edge_to_id[sorted_tuple((f[j], f[(j + 1) % f.size]))]

    return F2E


def boundary_to_full(boundary, full):
    element_to_id = {sorted_tuple(e): i for i, e in enumerate(full)}
    return np.array([element_to_id[sorted_tuple(e)] for e in boundary])


def attach_higher_order_nodes(V, E, F, order):
    V_new = V.tolist()

    for e in E:
        v1, v2 = V[e]
        if order == 2:
            V_new.append((v1 + v2) / 2)
        elif order == 3:
            V_new.append(v1 + (v2 - v1) / 3)
            V_new.append(v1 + 2 * (v2 - v1) / 3)
        else:
            raise NotImplementedError(f"P{order} not implemented!")

    for f in F:
        v1, v2, v3 = V[f]
        if order == 2:
            break
        elif order == 3:
            V_new.append((v1 + v2 + v3) / 3)
        else:
            raise NotImplementedError(f"P{order} not implemented!")

    # for tet in T:
    #     if order == 4:
    #         V_new.append(sum(V[tet]) / 4)

    return np.array(V_new)
