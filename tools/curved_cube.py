import pathlib
import meshio
import numpy

import igl

from mesh.utils import attach_higher_order_nodes, faces
from weights.higher_order_3D import build_phi_3D

cube = meshio.read(pathlib.Path(
    __file__).parents[1] / "meshes" / "cube" / "cube.msh")


V = cube.points / numpy.linalg.norm(cube.points, axis=1)[:, None]
T = cube.cells[0].data

E = igl.edges(T)
F = faces(T)

BF

V, F = attach_higher_order_nodes(V, E, F, T, 3)

build_phi_3D(V.shape[0], E.shape[0], V, )

norm_inf = numpy.linalg.norm(V, axis=1, ord=numpy.Inf)
norm2 = numpy.linalg.norm(V, axis=1)
breakpoint()
V[norm_inf > 0.9] /= norm2[norm_inf > 0.9, None]

curved_cube = meshio.Mesh(V, [("tetra20", F)])
curved_cube.write("cuved_cube.msh", file_format="gmsh")
