import sys
import pathlib

import numpy
import scipy.sparse
from scipy.sparse.linalg import spsolve

import quadpy
import trimesh
import igl

import sys
import pathlib

L2_dir = pathlib.Path(__file__).parent / "L2_projection"

build_dir = L2_dir / "build"
if not build_dir.exists():
    raise RuntimeError(
        "L2 projection needs to be built first:\ncd tools/weights/L2_projection; mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release; make -j4")

sys.path.append(str(build_dir))

import L2  # noqa


def phong_projection(origins, rays, V_target, F_target):
    # Embree ray cast
    print("Computing line mesh intersection")
    ids_and_coords = L2.igl_embree_line_mesh_intersection(
        origins, rays, V_target, F_target)
    # ids_and_coords = numpy.empty(origins.shape)

    missed_rays_ids = [
        query_i
        for query_i, (id, *_) in enumerate(ids_and_coords) if int(id) < 0
    ]
    # missed_rays_ids = numpy.arange(origins.shape[0])

    if len(missed_rays_ids) > 0:
        print(f"In total {len(missed_rays_ids)} / {rays.shape[0]} rays missed")
        print("Building ProximityQuery of trimesh")
        prox_query = trimesh.proximity.ProximityQuery(
            trimesh.Trimesh(V_target, F_target))

        print("Querying closest point on surface")
        closest_points, distances, triangle_ids = (
            prox_query.on_surface(origins[missed_rays_ids]))

        ids_and_coords[missed_rays_ids, 0] = triangle_ids
        print("Computing barycentric coordinates of closest point")
        ids_and_coords[missed_rays_ids, 1:] = igl.barycentric_coordinates_tri(
            closest_points,
            V_target[F_target[triangle_ids, 0]],
            V_target[F_target[triangle_ids, 1]],
            V_target[F_target[triangle_ids, 2]])[:, 1:]

    return ids_and_coords


def eliminate_near_zeros(A, tol=1e-12):
    A.data[numpy.abs(A.data) < tol] = 0
    A.eliminate_zeros()


def compute_L2_projection_weights(v_fem, f_fem, v_coll, f_coll, lump_mass_matrix=True):
    # some quadrature rule
    scheme = quadpy.t2.get_good_scheme(48)
    numpy.savetxt(str(L2_dir / "quadrature" / "points.csv"),
                  scheme.points.T.tolist(), delimiter=",")
    numpy.savetxt(str(L2_dir / "quadrature" / "weights.csv"),
                  scheme.weights.tolist(), delimiter=",")

    quadrature = L2.Quadrature()

    fem_bases = L2.build_bases(v_fem, f_fem)
    coll_bases = L2.build_bases(v_coll, f_coll)

    M = L2.compute_mass_mat(v_coll.shape[0], coll_bases, quadrature)
    assert(M.shape == (v_coll.shape[0], v_coll.shape[0]))

    A = L2.compute_mass_mat_cross(
        v_fem, f_fem, fem_bases, v_coll, f_coll, coll_bases, phong_projection,
        quadrature)
    assert(A.shape == (v_coll.shape[0], v_fem.shape[0]))

    if(lump_mass_matrix):
        M_lumped = scipy.sparse.csc_matrix(M.shape)
        M_lumped.setdiag(M.sum(axis=1))
        M = M_lumped

    W = spsolve(M, A)
    eliminate_near_zeros(W)

    return W
