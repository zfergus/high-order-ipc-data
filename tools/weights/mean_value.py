import numpy
import scipy
from scipy.sparse import lil_matrix, csc_matrix, coo_matrix

import igl

from .utils import quiet_tqdm

from numba import jit

EPSILON = 1e-10
EPSILON_FACE = 1e-10


@jit(nopython=True, parallel=False)
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


@jit(nopython=True, parallel=False)
def MorphPoint(vertices, values, point, faces, n_vertices, n_faces, dim=3, complex_dim=3):
    total_f = numpy.zeros(dim)
    total_w = 0.0

    weights = numpy.zeros(n_vertices)
    # weights = lil_matrix((n_vertices, 1))

    for i in range(n_vertices):
        d = numpy.linalg.norm(vertices[i] - point)
        if d < EPSILON:
            weights[i] = 1
            # weights[(i, 0)] = 1
            return values[i], weights

    vert = numpy.zeros((complex_dim, dim))
    val = numpy.zeros((complex_dim, dim))
    u = numpy.zeros((complex_dim, dim))
    d = numpy.zeros(complex_dim)
    l = numpy.zeros(complex_dim)
    theta = numpy.zeros(complex_dim)
    c = numpy.zeros(complex_dim)
    s = numpy.zeros(complex_dim)
    w = numpy.zeros(complex_dim)

    for j in range(n_faces):
        for k in range(complex_dim):
            vert[k] = vertices[faces[j][k]]
            val[k] = values[faces[j][k]]

        for i in range(complex_dim):
            d[i] = numpy.linalg.norm(vert[i] - point)
            u[i] = (vert[i] - point) / d[i]

        for i in range(complex_dim):
            ip1, im1 = get_imp1(i, complex_dim)

            l[i] = numpy.linalg.norm(u[ip1] - u[im1])

            if abs(l[i] - 2) < EPSILON:
                l[i] = 2
            elif abs(l[i] + 2) < EPSILON:
                l[i] = -2

            theta[i] = 2 * numpy.arcsin(l[i] / 2.)

        h = numpy.sum(theta) / 2.

        if numpy.pi - h < EPSILON_FACE:
            weights = numpy.zeros(n_vertices)
            # weights = lil_matrix((n_vertices, 1))
            for i in range(complex_dim):
                ip1, im1 = get_imp1(i, complex_dim)

                w[i] = numpy.sin(theta[i]) * d[ip1] * d[im1]
                weights[faces[j][i]] = w[i]
                # weights[(faces[j][i], 0)] = w[i]
                total_f += w[i] * val[i]

            total_f /= numpy.sum(w)
            return total_f, weights

        flag = 0
        for i in range(complex_dim):
            ip1, im1 = get_imp1(i, complex_dim)

            c[i] = (2 * numpy.sin(h) * numpy.sin(h - theta[i])) / \
                (numpy.sin(theta[ip1]) * numpy.sin(theta[im1])) - 1
            s[i] = numpy.sign(numpy.linalg.det(u)) * numpy.sqrt(1 - c[i]**2)

            if s[i] != s[i] or abs(s[i]) < EPSILON:
                flag = 1

        if flag == 1:
            continue

        for i in range(complex_dim):
            ip1, im1 = get_imp1(i, complex_dim)

            w[i] = (theta[i] - c[ip1] * theta[im1] - c[im1] *
                    theta[ip1]) / (d[i] * numpy.sin(theta[ip1]) * s[im1])
            weights[faces[j][i]] += w[i]
            # weights[(faces[j][i], 0)] += w[i]
            total_f += w[i] * val[i]
            total_w += w[i]

    if total_w == 0:
        print("Error")
        return total_f, weights

    return total_f / total_w, weights


def compute_mean_value_weights(P, V, F, quiet=True):
    """
    P: points of dense mesh; 2d matrix (|V_dense|, 3)
    V: vertices of tet mesh; 2d matrix (|V_tet|, 3)
    F: faces of tet mesh; 2d matrix (|F|, 4)
    returns mapping matrix
    """
    surface = igl.boundary_facets(F)  # get surface

    # Handling sparse matrices
    rows = numpy.array([])
    cols = numpy.array([])
    data = numpy.array([])

    # TODO: use igl::remove_unreferenced to remove interior points in V
    # NV : NV by dim list of mesh vertex positions
    # NF : NF by ss list of simplices
    # IM : V by 1 list of indices such that: NF = IM(F) and NT = IM(T) and V(find(IM<=size(NV,1)),:) = NV
    # J  : RV by 1 list, such that RV = V(J,:)
    # NV, NF, IM, J = igl.remove_unreferenced(V, surface)

    for i, p in enumerate(quiet_tqdm(P, quiet)):
        # _, w = MorphPoint(NV, NV, p, NF, NV.shape[0], NF.shape[0])
        _, w = MorphPoint(V, V, p, surface, V.shape[0], surface.shape[0])
        # TODO: map from reduced to full vertices
        w = coo_matrix(w)
        rows = numpy.append(rows, i * numpy.ones(w.row.shape[0]))
        cols = numpy.append(cols, w.row)
        data = numpy.append(data, w.data / numpy.sum(w.data))

    return csc_matrix((data, (rows, cols)), shape=(P.shape[0], V.shape[0]))
