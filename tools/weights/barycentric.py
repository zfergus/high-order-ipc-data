import numpy

from .utils import quiet_tqdm


def compute_bary(p, v):
    """
    p: point of dense mesh; vector (3)
    v: vertices of single tet; 2d matrix (4, 3)
    returns the bary coords wrt v
    """
    v_ap = p - v[0]
    v_bp = p - v[1]

    v_ab = v[1] - v[0]
    v_ac = v[2] - v[0]
    v_ad = v[3] - v[0]

    v_bc = v[2] - v[1]
    v_bd = v[3] - v[1]

    V_a = 1 / 6 * numpy.dot(v_bp, numpy.cross(v_bd, v_bc))
    V_b = 1 / 6 * numpy.dot(v_ap, numpy.cross(v_ac, v_ad))
    V_c = 1 / 6 * numpy.dot(v_ap, numpy.cross(v_ad, v_ab))
    V_d = 1 / 6 * numpy.dot(v_ap, numpy.cross(v_ab, v_ac))
    V = 1 / 6 * numpy.dot(v_ab, numpy.cross(v_ac, v_ad))

    bary = numpy.zeros(4)
    bary[0] = V_a / numpy.abs(V)
    bary[1] = V_b / numpy.abs(V)
    bary[2] = V_c / numpy.abs(V)
    bary[3] = V_d / numpy.abs(V)

    assert(abs(bary.sum() - 1) < 1e-10)

    return bary


def compute_barycentric_weights(P, V, F, quiet=True):
    """
    P: points of dense mesh; 2d matrix (|V_dense|, 3)
    V: vertices of tet mesh; 2d matrix (|V_tet|, 3)
    F: faces of tet mesh; 2d matrix (|F|, 4)
    returns mapping matrix
    """
    M = numpy.zeros((P.shape[0], V.shape[0]))

    for idx1, p in enumerate(quiet_tqdm(P, quiet)):
        flag = 0
        for idx2, f in enumerate(F):
            tet_vert = []
            for i in range(4):
                tet_vert.append(V[f[i]])
            tet_vert = numpy.array(tet_vert)

            bary = compute_bary(p, tet_vert)
            if numpy.all(bary >= 0):
                flag = 1
                for i in range(4):
                    M[idx1][f[i]] = bary[i]
                break

        # If the vertex is not present inside any tet
        # Finds the tet such that min number of bary coords are negative
        # Possible ISSUE: Doesn't handle the case of tie i.e if for two tets equal number of bary coords are negative elements
        tot_count = 0  # Sets the number of negative bary coords that we require
        while(flag == 0):
            tot_count += 1  # In first itration, tot_count = 1
            for idx2, f in enumerate(F):
                tet_vert = []
                for i in range(4):
                    tet_vert.append(V[f[i]])
                tet_vert = numpy.array(tet_vert)

                bary = compute_bary(p, tet_vert)
                count = 0
                for i in range(bary.shape[0]):  # counts negative terms
                    if bary[i] < 0:
                        count += 1
                if count == tot_count:
                    flag = 1
                    for i in range(4):
                        M[idx1][f[i]] = bary[i]
                    break

    return M
