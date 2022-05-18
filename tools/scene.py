"""Combines multiple weights and collision meshes into one."""

import argparse
import pathlib
import json

import numpy
import scipy.sparse
import meshio

from weights.utils import load_weights, save_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=pathlib.Path)
    return parser.parse_args()


def read_mesh(filename):
    print(f"Loading {filename}")
    mesh = meshio.read(filename)
    V = mesh.points
    F = []
    for cells in mesh.cells:
        assert(cells.type == "triangle")
        F.append(cells.data.astype(int))
    F = numpy.vstack(F)
    return V, F


def resolve_path(path, root):
    path = pathlib.Path(path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def main():
    args = parse_args()

    with open(args.json) as f:
        scene = json.load(f)

    root = args.json.parent

    V, F = [], []
    weights = []
    offset = 0
    for mesh in scene["meshes"]:
        if not mesh.get("enabled", True):
            continue
        lV, lF = read_mesh(resolve_path(mesh["collision_proxy"]["mesh"], root))
        V.append(lV)
        F.append(lF + offset)
        offset += lV.shape[0]

        weights.append(load_weights(resolve_path(
            mesh["collision_proxy"]["linear_map"], root)))
    V = numpy.vstack(V)
    F = numpy.vstack(F)

    mesh_path = resolve_path(scene["collision_mesh"]["mesh"], root)
    mesh_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving scene mesh to {mesh_path}")
    meshio.write(mesh_path, meshio.Mesh(V, [("triangle", F)]))

    weights_path = resolve_path(scene["collision_mesh"]["linear_map"], root)
    weights_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving scene weights to {weights_path}")
    W = scipy.sparse.block_diag(weights)
    assert(W.shape[0] == V.shape[0])
    save_weights(weights_path, W, W.shape[1])


if __name__ == "__main__":
    main()
