import pathlib
import meshio
import numpy as np
import scipy.sparse

import igl
import trimesh
from aabbtree import AABB, AABBTree
from tqdm import tqdm

from .bases import hat_phis_3D, nodes_count_to_order
from .point_triangle_distance import pointTriangleDistance
from .utils import labeled_tqdm
from mesh.sample_tet import upsample_mesh
from mesh.invert_gmapping import invert_gmapping
from mesh.utils import boundary_to_full

from subprocess import Popen, DEVNULL
import pickle


def uvw_to_bc(uvw):
    if len(uvw.shape) == 1:
        return np.array([[1-uvw.sum(), uvw[0], uvw[1], uvw[2]]])
    return np.hstack([(1 - uvw.sum(axis=1)).reshape(-1, 1), uvw])


def bc_to_uvw(bc):
    if len(bc.shape) == 1:
        return bc[1:]
    return bc[:, 1:].flatten()


def distance_heuristic(bc):
    if len(bc) == 3:
        bc = uvw_to_bc(bc)
    return np.linalg.norm(bc, ord=1)


def is_inside_tet(bc):
    assert(len(bc) == 4)
    return (bc >= 0).all()


def distance_to_P1_tet(p, tet, bc):
    if VolumetricClosestPointQuery.inside_tet(bc):
        return np.linalg.norm(p - bc.dot(tet))
    faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    return min(pointTriangleDistance(tet[f], p) for f in faces)


class VolumetricClosestPointQuery:
    def __init__(self, V, T, samples=4) -> None:
        """
        V: vertices of tet mesh; 2d matrix (|V_tet|, 3)
        T: tets of tet mesh; 2d matrix (|T|, 4)
        """
        self.V = V
        self.T = T

        self.mesh_order = nodes_count_to_order[self.T.shape[1]]

        if self.mesh_order > 1:
            self.V_upsampled, self.T_upsampled, self.Tup2T = upsample_mesh(
                self.V, self.T, samples)
            # TODO viz the mesh
        else:
            self.V_upsampled, self.T_upsampled = self.V, self.T
            self.Tup2T = np.arange(self.T.shape[0])

        # Build a tree for fast interior checks
        self.tree = AABBTree()
        for tet_i, tet in enumerate(labeled_tqdm(self.T_upsampled, "Build AABB Tree")):
            limits = np.vstack([self.V_upsampled[tet].min(axis=0),
                                self.V_upsampled[tet].max(axis=0)]).T
            self.tree.add(AABB(limits), tet_i)
        # with open("tree.pkl", "wb") as f:
        #     pickle.dump(self.tree, f)
        # with open("tree.pkl", "rb") as f:
        #     self.tree = pickle.load(f)

        # Extract the surface for closest proximity checks of exterior points
        if self.mesh_order == 1:
            BF = igl.boundary_facets(self.T_upsampled)
            self.BF2T = self.Tup2T[boundary_to_full(BF, self.T_upsampled)]
            self.bmesh = trimesh.Trimesh(self.V_upsampled, BF)
            self.proximity_query = trimesh.proximity.ProximityQuery(self.bmesh)

    def __call__(self, p, pi=None) -> tuple[int, np.ndarray]:
        """Find the closest tet and barycentric coordinates to the given point."""
        if self.mesh_order == 1:
            return self.closest_point_P1(p)
        return self.closest_point_PN(p)

    def closest_point_P1(self, p) -> tuple[int, np.ndarray]:
        assert(self.T_upsampled.shape == self.T.shape)
        limits = np.vstack([p, p]).T
        tis = self.tree.overlap_values(AABB(limits))
        bcs = igl.barycentric_coordinates_tet(
            np.tile(p, (len(tis), 1)),
            self.V[self.T[tis, 0]], self.V[self.T[tis, 1]],
            self.V[self.T[tis, 2]], self.V[self.T[tis, 3]])
        bcs = bcs.reshape(-1, 4)

        for ti, bc in zip(tis, bcs):
            if is_inside_tet(bc):
                return ti, bc

        # point must be on the exterior of the mesh, so find the closest point on the surface
        _, _, bfis = self.proximity_query.on_surface(p.reshape(1, 3))
        ti = self.BF2T[bfis[0]]
        bc = igl.barycentric_coordinates_tet(
            p.reshape(-1, 3).copy(),
            self.V[self.T[ti, 0]].copy().reshape(1, 3).copy(),
            self.V[self.T[ti, 1]].copy().reshape(1, 3).copy(),
            self.V[self.T[ti, 2]].copy().reshape(1, 3).copy(),
            self.V[self.T[ti, 3]].copy().reshape(1, 3).copy())
        return ti, bc

    def uvw0(self, p, ti):
        # ti = self.Tup2T[ti_up]
        return bc_to_uvw(igl.barycentric_coordinates_tet(
            p.reshape(-1, 3).copy(),
            self.V[self.T[ti, 0]].reshape(-1, 3).copy(),
            self.V[self.T[ti, 1]].reshape(-1, 3).copy(),
            self.V[self.T[ti, 2]].reshape(-1, 3).copy(),
            self.V[self.T[ti, 3]].reshape(-1, 3).copy()))

    def closest_point_PN(self, p) -> tuple[int, np.ndarray]:
        inflation = 1e-3
        tis = []
        while len(tis) < 3:
            limits = np.vstack([p - inflation, p + inflation]).T
            tis_up = self.tree.overlap_values(AABB(limits))
            tis = self.Tup2T[tis_up]
            tis = np.unique(tis)
            inflation *= 1.1

        bcs = uvw_to_bc(np.vstack([
            invert_gmapping(
                self.mesh_order, self.uvw0(p, ti), self.V[self.T[ti]], p)
            for ti in tis
        ]))

        i = np.argmin([distance_heuristic(bc) for bc in bcs])
        return tis[i], bcs[i]


def foo(pi0) -> tuple[int, np.ndarray]:
    pi1 = pi0 + 1000
    with open("VolumetricClosestPointQuery.pkl", "rb") as f:
        closest_point, P = pickle.load(f)
    data = []
    for pi, p in zip(range(pi0, min(pi1, P.shape[0])), P[pi0:pi1]):
        print(pi)
        ti, bc = closest_point(p)
        data.append((pi, ti, bc.tolist()))
    return data


def compute_barycentric_weights(P, V_geom, T_geom, V_disp=None, T_disp=None):
    """
    P: points of dense mesh; 2d matrix (|P|, 3)
    V: vertices of tet mesh; 2d matrix (|V|, 3)
    T_geom: Tets of the geometry (possibly with higher order nodes attached)
    T_disp: Tets of the displacement (possibly with higher order nodes attached)
    order: basis order
    returns mapping matrix
    """
    if V_disp is None:
        V_disp = V_geom
    if T_disp is None:
        T_disp = T_geom

    M = scipy.sparse.lil_matrix((P.shape[0], V_disp.shape[0]))

    geom_order = nodes_count_to_order[T_geom.shape[1]]
    disp_order = nodes_count_to_order[T_disp.shape[1]]

    closest_point = VolumetricClosestPointQuery(V_geom, T_geom)

    # query_path = pathlib.Path("VolumetricClosestPointQuery.pkl")
    # if query_path.exists():
    #     with open("VolumetricClosestPointQuery.pkl", "rb") as f:
    #         closest_point, P = pickle.load(f)
    # else:
    #     closest_point = VolumetricClosestPointQuery(V_geom, T_geom)
    #     with open("VolumetricClosestPointQuery.pkl", "wb") as f:
    #         pickle.dump((closest_point, P), f)

    # n = 500
    # for i in range(0, P.shape[0], n):
    #     Popen(['nohup', 'python', 'tools/worker.py', str(i), str(i+n)],
    #           stdout=DEVNULL, stderr=DEVNULL)
    # print("spawned and exiting")
    # exit(0)

    # pis = set()
    # for i in range(0, P.shape[0], n):
    #     with open(f"rows/{i}-{i+n}_updated.pkl", "rb") as f:
    #         data = pickle.load(f)
    #     updated_data = []
    #     for pi, ti, bc in labeled_tqdm(data, "Compute W"):
    #         assert(pi not in pis)
    #         pis.add(pi)

    #         if not np.isfinite(bc).all():
    #             print(f"recomputing {pi}")
    #             ti, bc = closest_point(P[pi])

    #         M[pi, T_disp[ti]] = [phi(*bc[1:])
    #                              for phi in hat_phis_3D[disp_order]]

    #         # err = np.linalg.norm(P[pi] - M[pi] @ V_disp, ord=np.Inf)
    #         # if err > 1e-4:
    #         #     print(f"recomputing {pi}")
    #         #     ti, bc = closest_point(P[pi])
    #         #     new_err = np.linalg.norm(P[pi] - M[pi] @ V_disp, ord=np.Inf)
    #         #     breakpoint()

    #         updated_data.append((pi, ti, bc))
    #     # with open(f"rows/{i}-{i+n}_updated.pkl", "wb") as f:
    #     #     pickle.dump(updated_data, f)
    # for pi in range(P.shape[0]):
    #     assert(pi in pis)

    for pi, p in enumerate(labeled_tqdm(P, "Compute W")):
        ti, bc = closest_point(p)
        M[pi, T_disp[ti]] = [phi(*bc_to_uvw(bc).flatten())
                             for phi in hat_phis_3D[disp_order]]

    return M
