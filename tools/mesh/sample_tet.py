import numpy as np
import igl

from .invert_gmapping import gmapping
from weights.bases import nodes_count_to_order
from weights.utils import labeled_tqdm


def add_tet(tmp, V):
    if tmp[0] >= 0 and tmp[1] >= 0 and tmp[2] >= 0 and tmp[3] >= 0:
        e0 = V[tmp[1], :] - V[tmp[0], :]
        e1 = V[tmp[2], :] - V[tmp[0], :]
        e2 = V[tmp[3], :] - V[tmp[0], :]

        vol = np.dot(np.cross(e0, e1), e2)
        if vol < 0:
            T = np.array([tmp[0], tmp[1], tmp[2], tmp[3]])
        else:
            T = np.array([tmp[0], tmp[1], tmp[3], tmp[2]])

        return T
    else:
        return None


def sample_tet(nn):
    n = nn
    delta = 1. / (n - 1.)

    T = np.zeros(((n - 1) * (n - 1) * (n - 1) * 6, 4))
    V = np.zeros((n * n * n, 3))
    mmap = -np.ones(n * n * n, dtype=int)

    index = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i + j + k >= n:
                    continue
                mmap[(i + j * n) * n + k] = index
                V[index, :] = [i * delta, j * delta, k * delta]
                index += 1

    V = V[:index, :]

    index = 0
    for i in range(n-1):
        for j in range(n-1):
            for k in range(n-1):
                indices = [(i + j * n) * n + k,
                           (i + 1 + j * n) * n + k,
                           (i + 1 + (j + 1) * n) * n + k,
                           (i + (j + 1) * n) * n + k,
                           (i + j * n) * n + k + 1,
                           (i + 1 + j * n) * n + k + 1,
                           (i + 1 + (j + 1) * n) * n + k + 1,
                           (i + (j + 1) * n) * n + k + 1]

                tmp = [mmap[indices[1 - 1]], mmap[indices[2 - 1]],
                       mmap[indices[4 - 1]], mmap[indices[5 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

                tmp = [mmap[indices[6 - 1]], mmap[indices[3 - 1]],
                       mmap[indices[7 - 1]], mmap[indices[8 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

                tmp = [mmap[indices[5 - 1]], mmap[indices[2 - 1]],
                       mmap[indices[6 - 1]], mmap[indices[4 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

                tmp = [mmap[indices[5 - 1]], mmap[indices[4 - 1]],
                       mmap[indices[8 - 1]], mmap[indices[6 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

                tmp = [mmap[indices[4 - 1]], mmap[indices[2 - 1]],
                       mmap[indices[6 - 1]], mmap[indices[3 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

                tmp = [mmap[indices[3 - 1]], mmap[indices[4 - 1]],
                       mmap[indices[8 - 1]], mmap[indices[6 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

    T = T[:index, :]

    return V, T


def upsample_mesh(V, T, samples):
    order = nodes_count_to_order[T.shape[1]]

    UVW, Tref = sample_tet(samples)

    Vup = []
    Tup = []
    offset = 0
    for tet in labeled_tqdm(T, "Upsampling tet mesh"):
        XYZ = gmapping(order, UVW, V[tet])
        Vup.extend(XYZ)
        Tup.extend(Tref + offset)
        offset += XYZ.shape[0]
    nVup = len(Vup)

    Vup = np.array(Vup)
    Tup = np.array(Tup, dtype=int)
    assert(nVup == Vup.shape[0])
    assert(Tup.max() <= Vup.shape[0])

    Tup_to_T = np.arange(Tup.shape[0], dtype=int) // Tref.shape[0]

    Vup_new, indices, inverse, Tup_new = igl.remove_duplicate_vertices(
        Vup, Tup, epsilon=1e-8)
    Vup = Vup_new
    Tup = Tup_new
    assert(Tup.max() <= Vup.shape[0])

    return Vup, Tup, Tup_to_T
