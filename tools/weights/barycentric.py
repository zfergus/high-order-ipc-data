import itertools
import numpy
import scipy.sparse
from numba import jit, prange

import igl
import trimesh
from aabbtree import AABB, AABBTree

from .utils import quiet_tqdm, labeled_tqdm, sorted_tuple
from .point_triangle_distance import pointTriangleDistance


@jit
def compute_barycentric_coords(p, v):
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


# @jit(nopython=True)
def distance_to_tet(p, tet, bc):
    if (0 <= bc).all() and (bc <= 1).all() and abs(bc.sum() - 1) < 1e-12:
        return numpy.linalg.norm(p - bc.dot(tet))
    faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    return min(pointTriangleDistance(tet[f], p) for f in faces)


def compute_barycentric_weights(P, V, F, quiet=True):
    """
    P: points of dense mesh; 2d matrix (|V_dense|, 3)
    V: vertices of tet mesh; 2d matrix (|V_tet|, 3)
    F: faces of tet mesh; 2d matrix (|F|, 4)
    returns mapping matrix
    """
    M = scipy.sparse.lil_matrix((P.shape[0], V.shape[0]))

    # Build a tree for fast interior checks
    tree = AABBTree()
    for fi, f in enumerate(labeled_tqdm(F, "Build AABB Tree")):
        limits = numpy.vstack([V[f].min(axis=0), V[f].max(axis=0)]).T
        tree.add(AABB(limits), fi)

    # Extract the surface for closest proximity checks of exterior points
    BF = igl.boundary_facets(F)
    face_to_tet_id = {
        sorted_tuple(f): i for i, tet in enumerate(F) for f in itertools.combinations(tet, 3)
    }
    BF2F = numpy.array([face_to_tet_id[sorted_tuple(f)] for f in BF])
    surface = trimesh.Trimesh(V, BF)

    for pi, p in enumerate(labeled_tqdm(P, "Compute W")):
        found = False

        limits = numpy.vstack([p, p]).T
        fis = tree.overlap_values(AABB(limits))
        bcs = [compute_barycentric_coords(p, V[F[fi]]) for fi in fis]
        for fi, bc in zip(fis, bcs):
            if (0 <= bc).all() and (bc <= 1).all() and abs(bc.sum() - 1) < 1e-12:
                found = True
                M[pi, F[fi]] = bc
                d1 = 0
                break

        if not found:
            # point must be on the exterior of the mesh, so find the closest point on the surface
            closest, distance, triangle_id = (
                trimesh.proximity.ProximityQuery(surface).on_surface(
                    p.reshape(1, 3)))
            fi = BF2F[triangle_id[0]]
            M[pi, F[fi]] = compute_barycentric_coords(p, V[F[fi]])
            # d1 = distance_to_tet(
            #     p, V[F[fi]], compute_barycentric_coords(p, V[F[fi]]))

        # Brute force
        # all_bcs = [compute_barycentric_coords(p, V[f]) for f in F]
        # all_distances = numpy.array([distance_to_tet(p, V[f], bc)
        #                              for bc, f in zip(all_bcs, F)])
        # all_fi = all_distances.argmin()
        #
        # d2 = all_distances[fi]
        # if fi != all_fi and d2 < d1:
        #     print(f"{fi} {all_fi}:{d2} < {d1}! diff={abs(d2 - d1)}")

    return M
