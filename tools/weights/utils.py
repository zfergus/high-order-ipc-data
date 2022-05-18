import numpy as np
import scipy.sparse
import h5py
from tqdm import tqdm


def quiet_tqdm(x, quiet):
    return x if quiet else tqdm(x)


def labeled_tqdm(data, label):
    pbar = tqdm(data)
    pbar.set_description(label)
    return pbar


def save_weights(path, W, n_fem_vertices, vertices=None, edges=None, faces=None):
    """
    Save a weight matrix.
    Optionally: save the edge and/or face matricies
    """
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

    h5f.attrs["n_fem_vertices"] = n_fem_vertices

    if vertices is not None:
        h5f.create_dataset("ordered_vertices", data=vertices)
    if edges is not None:
        h5f.create_dataset("ordered_edges", data=edges)
    if faces is not None:
        h5f.create_dataset("ordered_faces", data=faces)

    h5f.close()


def load_weights(path):
    with h5py.File(path, 'r') as h5f:
        if "weight_triplets" in h5f:
            g = h5f['weight_triplets']
            return scipy.sparse.coo_matrix(
                (g['values'][:], (g['rows'][:], g['cols'][:])), g.attrs['shape']
            ).tocsc()
        else:
            assert("weights" in h5f)
            return h5f['weights']
