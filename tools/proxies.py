import argparse
import pathlib

import numpy
import scipy.sparse

from tqdm import tqdm
from tqdm.contrib import tzip

import pymesh

from weights.barycentric import compute_barycentric_weights
from weights.utils import save_weights


def compute_kDOPs(mesh, k=6):
    comps = pymesh.separate_mesh(mesh)
    for comp in comps:
        points = comp.vertices
        center = points.mean(axis=1)
        min_values = numpy.full(k, numpy.inf)
        max_values = numpy.full(k, -numpy.inf)
        directions = numpy.empty((k, 3))
        kdops = []
        for i, point in enumerate(points):
            for j, dir in enumerate(directions):
                t = (point.dot(dir) - center.dot(dir)) / (dir.dot(dir))
                min_values[j] = min(min_values[j], t)
                max_values[j] = max(max_values[j], t)


CUBE_FACE_INDICES = numpy.array((
    (0, 1, 3, 2),
    (2, 3, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (2, 6, 4, 0),
    (7, 3, 1, 5),
))


def gen_cube_verts():
    for x in [-1.0, 1.0]:
        for y in [-1.0, 1.0]:
            for z in [-1.0, 1.0]:
                yield x, y, z


CUBE_TETS = numpy.array([
    [1, 3, 8, 7],
    [1, 3, 4, 8],
    [1, 4, 2, 8],
    [6, 1, 2, 8],
    [6, 1, 8, 5],
    [8, 1, 7, 5],
]) - 1


def tet_volume(V, T):
    a = V[T[1]] - V[T[0]]
    b = V[T[2]] - V[T[0]]
    c = V[T[3]] - V[T[0]]
    return 1 / 6 * numpy.cross(a, b).dot(c)


def OBB(mesh):
    vertices = pymesh.convex_hull(mesh).vertices

    cov_mat = numpy.cov(vertices, rowvar=False, bias=True)
    eig_vals, eig_vecs = numpy.linalg.eigh(cov_mat)

    change_of_basis_mat = eig_vecs
    inv_change_of_basis_mat = numpy.linalg.inv(change_of_basis_mat)

    aligned = vertices @ inv_change_of_basis_mat.T

    box_min = aligned.min(axis=0)
    box_max = aligned.max(axis=0)

    center = (box_max + box_min) / 2
    center_world = center @ change_of_basis_mat.T

    box_points = numpy.array(list(gen_cube_verts()), dtype=numpy.float64)
    box_points *= (box_max - box_min) / 2
    box_points = box_points @ change_of_basis_mat.T
    box_points += center_world

    tets = CUBE_TETS.copy()
    orientations = pymesh.get_tet_orientations_raw(box_points, tets)
    for ori, tet in zip(orientations, tets):
        if(ori < 0):
            tet[[0, 1]] = tet[[1, 0]]
    assert(all(pymesh.get_tet_orientations_raw(box_points, tets) > 0))

    return pymesh.form_mesh(box_points, None, voxels=tets)


def compute_convex_hulls(mesh):
    comps = pymesh.separate_mesh(mesh)
    return [pymesh.convex_hull(comp) for comp in comps]


def concat_meshes(meshes):
    vertices = []
    voxels = []
    num_vertices = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        voxels.append(mesh.voxels + num_vertices)
        num_vertices += mesh.vertices.shape[0]
    return pymesh.form_mesh(
        numpy.vstack(vertices), None, voxels=numpy.vstack(voxels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=pathlib.Path)
    parser.add_argument('-o,--outdir', dest="outdir",
                        default=pathlib.Path("."), type=pathlib.Path)
    args = parser.parse_args()

    print(f"loading {args.mesh.resolve()}")
    mesh = pymesh.load_mesh(str(args.mesh.resolve()))

    print("separating mesh")
    components = pymesh.separate_mesh(mesh)

    print("generating proxies")
    proxies = []
    # proxies = compute_convex_hulls(component)
    for component in tqdm(components):
        proxies.append(OBB(component))

    print("computing weights")
    # Compute weights
    block_weights = []
    comps = pymesh.separate_mesh(mesh)
    prev_v = 0
    for i, (comp, proxy) in enumerate(tzip(comps, proxies)):
        block_weights.append(compute_barycentric_weights(
            comp.vertices, proxy.vertices, proxy.voxels))

        error = numpy.linalg.norm(
            block_weights[-1] @ proxy.vertices - comp.vertices, axis=1)
        # print(i, error.max())
        assert(error.max() < 1e-10)

        prev_v += comp.vertices.shape[0]

    weights = scipy.sparse.block_diag(block_weights)
    weights.eliminate_zeros()

    fem_mesh = concat_meshes(proxies)

    args.outdir.mkdir(exist_ok=True, parents=True)
    # pymesh.save_mesh(str(args.outdir / f"fem.obj"), fem_mesh)
    pymesh.save_mesh(
        str(args.outdir / f"{args.mesh.stem}_proxies.msh"), fem_mesh)
    # pymesh.save_mesh(str(args.outdir / f"contact.obj"),
    #                  pymesh.form_mesh(weights @ fem_mesh.vertices, mesh.faces))
    save_weights(args.outdir / f"{args.mesh.stem}_proxy_weights.hdf5",
                 weights, fem_mesh.vertices.shape[0])


if __name__ == "__main__":
    main()
