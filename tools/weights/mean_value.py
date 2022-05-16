import numpy
import scipy.sparse
from numba import jit, prange

import igl

from .utils import labeled_tqdm

EPSILON = 1e-10
EPSILON_FACE = 1e-10


@jit(nopython=True, fastmath=True)
def get_imp1(i, complex_dim):
    ip1 = 0
    im1 = 0
    if i + 1 >= complex_dim:
        ip1 = 0
    else:
        ip1 = i + 1
    if i > 0:
        im1 = i - 1
    else:
        im1 = complex_dim - 1

    return ip1, im1


@jit(nopython=True, fastmath=True)
def MorphPoint(vertices, point, faces, dim=3, complex_dim=3):
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]

    assert(n_faces > 0)
    assert(n_vertices > 0)

    total_f = numpy.zeros(dim)
    total_w = 0.0

    weights = numpy.zeros(n_vertices)

    for i in range(n_vertices):
        d = numpy.linalg.norm(vertices[i] - point)
        if d < EPSILON:
            weights[i] = 1
            return weights

    vert = numpy.zeros((complex_dim, dim))
    u = numpy.zeros((complex_dim, dim))
    d = numpy.zeros(complex_dim)
    l = numpy.zeros(complex_dim)
    theta = numpy.zeros(complex_dim)
    c = numpy.zeros(complex_dim)
    s = numpy.zeros(complex_dim)
    w = numpy.zeros(complex_dim)

    IP1 = []
    IM1 = []
    for i in range(complex_dim):
        ip1, im1 = get_imp1(i, complex_dim)
        IP1.append(ip1)
        IM1.append(im1)

    for j in prange(n_faces):
        for i in range(complex_dim):
            vert[i] = vertices[faces[j][i]]
            d[i] = numpy.linalg.norm(vert[i] - point)
            u[i] = (vert[i] - point) / d[i]

        h = 0
        for i in range(complex_dim):
            l[i] = numpy.linalg.norm(u[IP1[i]] - u[IM1[i]])

            if abs(l[i] - 2) < EPSILON:
                l[i] = 2
            elif abs(l[i] + 2) < EPSILON:
                l[i] = -2

            theta[i] = 2 * numpy.arcsin(l[i] / 2.)

            h += theta[i] / 2.

        if numpy.pi - h < EPSILON_FACE:
            weights = numpy.zeros(n_vertices)
            for i in range(complex_dim):
                w[i] = numpy.sin(theta[i]) * d[IP1[i]] * d[IM1[i]]
                weights[faces[j][i]] = w[i]

            return weights

        flag = 0
        for i in range(complex_dim):
            c[i] = (2 * numpy.sin(h) * numpy.sin(h - theta[i])) / \
                (numpy.sin(theta[IP1[i]]) * numpy.sin(theta[IM1[i]])) - 1
            s[i] = numpy.sign(numpy.linalg.det(u)) * numpy.sqrt(1 - c[i]**2)

            if s[i] != s[i] or abs(s[i]) < EPSILON:
                flag = 1
                break

        if flag == 1:
            continue

        for i in range(complex_dim):
            w[i] = (theta[i] - c[IP1[i]] * theta[IM1[i]] - c[IM1[i]] *
                    theta[IP1[i]]) / (d[i] * numpy.sin(theta[IP1[i]]) * s[IM1[i]])
            weights[faces[j][i]] += w[i]

    return weights


def compute_mean_value_weights(P, V, F, quiet=True):
    """
    P: points of dense mesh; 2d matrix (|V_dense|, 3)
    V: vertices of tet mesh; 2d matrix (|V_tet|, 3)
    F: faces of tet mesh; 2d matrix (|F|, 4)
    returns mapping matrix
    """
    assert(F.shape[1] == 4)
    BF = igl.boundary_facets(F)  # get surface
    # NV, NF, IM, _ = igl.remove_unreferenced(V, surface)  # Remove unreferenced

    W = []

    for i, p in enumerate(labeled_tqdm(P, "Building W")):
        w = MorphPoint(V, p, BF)
        total_w = w.sum()
        # assert(total_w > 0)
        W.append(w / total_w)

    return scipy.sparse.csc_matrix(numpy.vstack(W))
