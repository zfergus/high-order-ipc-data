import numpy
import scipy.sparse
from numba import jit, prange

import igl
import trimesh
from aabbtree import AABB, AABBTree

from .bases import hat_phis_3D
from .point_triangle_distance import pointTriangleDistance
from .utils import quiet_tqdm, labeled_tqdm
from mesh.utils import boundary_to_full


def distance_to_tet(p, tet, bc):
    if (0 <= bc).all() and (bc <= 1).all() and abs(bc.sum() - 1) < 1e-12:
        return numpy.linalg.norm(p - bc.dot(tet))
    faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    return min(pointTriangleDistance(tet[f], p) for f in faces)


class VolumetricClosestPointQuery:
    def __init__(self, V, F) -> None:
        """
        V: vertices of tet mesh; 2d matrix (|V_tet|, 3)
        F: tets of tet mesh; 2d matrix (|F|, 4)
        """
        self.V = V
        self.F = F

        # Build a tree for fast interior checks
        self.tree = AABBTree()
        for fi, f in enumerate(labeled_tqdm(F, "Build AABB Tree")):
            limits = numpy.vstack([V[f].min(axis=0), V[f].max(axis=0)]).T
            self.tree.add(AABB(limits), fi)

        # Extract the surface for closest proximity checks of exterior points
        BF = igl.boundary_facets(F)
        self.BF2F = boundary_to_full(BF, F)
        self.proximity_query = trimesh.proximity.ProximityQuery(
            trimesh.Trimesh(V, BF))

    def __call__(self, p) -> tuple[int, numpy.ndarray]:
        """Find the closest tet and barycentric coordinates to the given point."""
        limits = numpy.vstack([p, p]).T
        fis = self.tree.overlap_values(AABB(limits))
        bcs = igl.barycentric_coordinates_tet(
            numpy.tile(p, (len(fis), 1)),
            self.V[self.F[fis, 0]], self.V[self.F[fis, 1]],
            self.V[self.F[fis, 2]], self.V[self.F[fis, 3]])
        bcs = bcs.reshape(-1, 4)

        for fi, bc in zip(fis, bcs):
            if (0 <= bc).all() and (bc <= 1).all() and abs(bc.sum() - 1) < 1e-12:
                return fi, bc

        # point must be on the exterior of the mesh, so find the closest point on the surface
        _, _, triangle_id = self.proximity_query.on_surface(p.reshape(1, 3))
        fi = self.BF2F[triangle_id[0]]
        bc = igl.barycentric_coordinates_tet(
            p.reshape(-1, 3).copy(),
            self.V[self.F[fi, 0]].copy().reshape(1, 3).copy(),
            self.V[self.F[fi, 1]].copy().reshape(1, 3).copy(),
            self.V[self.F[fi, 2]].copy().reshape(1, 3).copy(),
            self.V[self.F[fi, 3]].copy().reshape(1, 3).copy())
        return fi, bc

    def brute_force(self, p) -> tuple[int, numpy.ndarray]:
        """Brute force"""
        bcs = igl.barycentric_coordinates_tet(
            numpy.tile(p, (self.F.shape[0], 1)),
            self.V[self.F[:, 0]], self.V[self.F[:, 1]],
            self.V[self.F[:, 2]], self.V[self.F[:, 3]])
        distances = numpy.array([
            distance_to_tet(p, self.V[f], bc) for bc, f in zip(bcs, self.F)])
        fi = distances.argmin()
        return fi, bcs[fi]


def compute_barycentric_weights(P, V, F, F_HO=None, order=1, quiet=True):
    """
    P: points of dense mesh; 2d matrix (|V_dense|, 3)
    V: vertices of tet mesh; 2d matrix (|V_tet|, 3)
    F: faces of tet mesh; 2d matrix (|F|, 4)
    F_HO: Tets with higher order nodes attached
    order: basis order
    returns mapping matrix
    """
    M = scipy.sparse.lil_matrix((P.shape[0], V.shape[0]))

    if F_HO is None:
        F_HO = F

    closest_point = VolumetricClosestPointQuery(V, F)

    for pi, p in enumerate(labeled_tqdm(P, "Compute W")):
        fi, bc = closest_point(p)
        uvw = bc[1:]
        M[pi, F_HO[fi]] = [phi(*uvw) for phi in hat_phis_3D[order]]

    return M
