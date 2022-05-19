import numpy

hat_phis_1D = {
    1: [lambda x: 1 - x, lambda x: x],
    2: [
        lambda x: 2 * (x - 0.5) * (x - 1),
        lambda x: 2 * (x - 0) * (x - 0.5),
        lambda x: -4 * (x - 0.5)**2 + 1
    ],
    3: [
        lambda x: -9 / 2 * (x - 1 / 3) * (x - 2 / 3) * (x - 1),
        lambda x: 9 / 2 * (x - 0) * (x - 1 / 3) * (x - 2 / 3),
        lambda x: 27 / 2 * (x - 0) * (x - 2 / 3) * (x - 1),
        lambda x: -27 / 2 * (x - 0) * (x - 1 / 3) * (x - 1),
    ]
}

hat_phis_2D = {
    1: [
        lambda x, y: -x - y + 1,
        lambda x, y: x,
        lambda x, y: y
    ],
    2: [
        lambda x, y: (x + y - 1) * (2 * x + 2 * y - 1),
        lambda x, y: x * (2 * x - 1),
        lambda x, y: y * (2 * y - 1),
        lambda x, y: -4 * x * (x + y - 1),
        lambda x, y: 4 * x * y,
        lambda x, y: -4 * y * (x + y - 1),
    ],
    3: [
        lambda x, y: (-27.0 / 2.0 * x**2 * y + 9 * x**2 - 27.0 / 2.0 * y**2 * x + 9 * y**2 -
                      9.0 / 2.0 * x**3 + 18 * x * y - 11.0 / 2.0 * x - 9.0 / 2.0 * y**3 - 11.0 / 2.0 * y + 1),
        lambda x, y: (1.0 / 2.0) * x * (9 * x**2 - 9 * x + 2),
        lambda x, y: (1.0 / 2.0) * y * (9 * y**2 - 9 * y + 2),
        lambda x, y: (9.0 / 2.0) * x * (x + y - 1) * (3 * x + 3 * y - 2),
        lambda x, y: -9.0 / 2.0 * x * (3 * x**2 + 3 * x * y - 4 * x - y + 1),
        lambda x, y: (9.0 / 2.0) * x * y * (3 * x - 1),
        lambda x, y: (9.0 / 2.0) * x * y * (3 * y - 1),
        lambda x, y: (9.0 / 2.0) * y * (x + y - 1) * (3 * x + 3 * y - 2),
        lambda x, y: -9.0 / 2.0 * y * (3 * x * y - x + 3 * y**2 - 4 * y + 1),
        lambda x, y: -27 * x * y * (x + y - 1),
    ],
    4: [
        lambda x, y: (
            helper_0 := pow(x, 2),
            helper_1 := pow(x, 3),
            helper_2 := pow(y, 2),
            helper_3 := pow(y, 3),
            64*helper_0*helper_2 - 80*helper_0*y + (70.0/3.0)*helper_0
            + (128.0/3.0)*helper_1*y - 80.0/3.0*helper_1 - 80*helper_2*x
            + (70.0/3.0)*helper_2 + (128.0/3.0)*helper_3*x - 80.0/3.0*helper_3
            + (32.0/3.0)*pow(x, 4) + (140.0/3.0)*x*y - 25.0/3.0*x
            + (32.0/3.0)*pow(y, 4) - 25.0/3.0*y + 1
        )[-1],
        lambda x, y: (1.0/3.0)*x*(32*pow(x, 3) - 48*pow(x, 2) + 22*x - 3),
        lambda x, y: (1.0/3.0)*y*(32*pow(y, 3) - 48*pow(y, 2) + 22*y - 3),
        lambda x, y: (
            helper_0 := pow(x, 2),
            helper_1 := pow(y, 2),
            - 16.0/3.0*x*(24*helper_0*y - 18*helper_0 + 24*helper_1*x
                          - 18*helper_1 + 8*pow(x, 3) - 36*x*y + 13*x + 8*pow(y, 3) + 13*y - 3)
        )[-1],
        lambda x, y: (
            helper_0 := 32*pow(x, 2),
            helper_1 := pow(y, 2),
            4*x*(helper_0*y - helper_0 + 16*helper_1*x - 4 *
                 helper_1 + 16*pow(x, 3) - 36*x*y + 19*x + 7*y - 3)
        )[-1],
        lambda x, y: (
            helper_0 := pow(x, 2),
            -16.0/3.0*x * (8*helper_0*y - 14*helper_0 + 8 *
                           pow(x, 3) - 6*x*y + 7*x + y - 1),
        )[-1],
        lambda x, y: (16.0/3.0)*x*y*(8*pow(x, 2) - 6*x + 1),
        lambda x, y: (
            helper_0 := 4*x,
            helper_0*y * (-helper_0 + 16*x*y - 4*y + 1),
        )[-1],
        lambda x, y: (16.0/3.0)*x*y*(8*pow(y, 2) - 6*y + 1),
        lambda x, y: (
            helper_0 := pow(y, 2),
            -16.0/3.0*y * (8*helper_0*x - 14*helper_0 - 6 *
                           x*y + x + 8*pow(y, 3) + 7*y - 1)
        )[-1],
        lambda x, y: (
            helper_0 := pow(x, 2),
            helper_1 := 32*pow(y, 2),
            4*y*(16*helper_0*y - 4*helper_0 + helper_1 *
                 x - helper_1 - 36*x*y + 7*x + 16*pow(y, 3) + 19*y - 3)
        )[-1],
        lambda x, y: (
            helper_0 := pow(x, 2),
            helper_1 := pow(y, 2),
            -16.0/3.0*y*(24*helper_0*y - 18*helper_0 + 24*helper_1*x -
                         18*helper_1 + 8*pow(x, 3) - 36*x*y + 13*x + 8*pow(y, 3) + 13*y - 3)
        )[-1],
        lambda x, y: 32*x*y*(x + y - 1)*(4*x + 4*y - 3),
        lambda x, y: -32*x*y*(4*y - 1)*(x + y - 1),
        lambda x, y: -32*x*y*(4*x - 1)*(x + y - 1),
    ]
}

edges_2D_order = numpy.array([[0, 1], [1, 2], [2, 0]])

hat_phis_3D = {
    1: [
        lambda x, y, z: -x - y - z + 1,
        lambda x, y, z: x,
        lambda x, y, z: y,
        lambda x, y, z: z
    ],
    2: [
        lambda x, y, z: (x + y + z - 1)*(2*x + 2*y + 2*z - 1),
        lambda x, y, z: x*(2*x - 1),
        lambda x, y, z: y*(2*y - 1),
        lambda x, y, z: z*(2*z - 1),
        lambda x, y, z: -4*x*(x + y + z - 1),
        lambda x, y, z: 4*x*y,
        lambda x, y, z: -4*y*(x + y + z - 1),
        lambda x, y, z: -4*z*(x + y + z - 1),
        lambda x, y, z: 4*x*z,
        lambda x, y, z: 4*y*z,
    ],
    3: [
        lambda x, y, z: (
            helper_0 := 18*x,
            helper_1 := y*z,
            helper_2 := pow(x, 2),
            helper_3 := pow(y, 2),
            helper_4 := pow(z, 2),
            helper_5 := (27.0/2.0)*x,
            helper_6 := (27.0/2.0)*y,
            helper_7 := (27.0/2.0)*z,
            helper_0*y + helper_0*z - 27*helper_1*x + 18*helper_1 - helper_2*helper_6
            - helper_2*helper_7 + 9*helper_2 - helper_3*helper_5 - helper_3*helper_7
            + 9*helper_3 - helper_4*helper_5 - helper_4*helper_6 + 9*helper_4
            - 9.0/2.0*pow(x, 3) - 11.0/2.0*x - 9.0/2.0*pow(y, 3) - 11.0/2.0*y
            - 9.0/2.0*pow(z, 3) - 11.0/2.0*z + 1)[-1],
        lambda x, y, z: (1.0/2.0)*x*(9*pow(x, 2) - 9*x + 2),
        lambda x, y, z: (1.0/2.0)*y*(9*pow(y, 2) - 9*y + 2),
        lambda x, y, z: (1.0/2.0)*z*(9*pow(z, 2) - 9*z + 2),
        lambda x, y, z: (9.0/2.0)*x*(x + y + z - 1)*(3*x + 3*y + 3*z - 2),
        lambda x, y, z: (helper_0 := 3*x, -9.0/2.0*x*(helper_0 *
                                                      y + helper_0*z + 3*pow(x, 2) - 4*x - y - z + 1))[-1],
        lambda x, y, z: (9.0/2.0)*x*y*(3*x - 1),
        lambda x, y, z: (9.0/2.0)*x*y*(3*y - 1),
        lambda x, y, z: (helper_0 := 3*y, -9.0/2.0*y*(helper_0 *
                                                      x + helper_0*z - x + 3*pow(y, 2) - 4*y - z + 1))[-1],
        lambda x, y, z: (9.0/2.0)*y*(x + y + z - 1)*(3*x + 3*y + 3*z - 2),
        lambda x, y, z: (9.0/2.0)*z*(x + y + z - 1)*(3*x + 3*y + 3*z - 2),
        lambda x, y, z: (helper_0 := 3*z, -9.0/2.0*z*(helper_0 *
                                                      x + helper_0*y - x - y + 3*pow(z, 2) - 4*z + 1))[-1],
        lambda x, y, z: (9.0/2.0)*x*z*(3*x - 1),
        lambda x, y, z: (9.0/2.0)*x*z*(3*z - 1),
        lambda x, y, z: (9.0/2.0)*y*z*(3*y - 1),
        lambda x, y, z: (9.0/2.0)*y*z*(3*z - 1),
        lambda x, y, z: -27*x*y*(x + y + z - 1),
        lambda x, y, z: -27*x*z*(x + y + z - 1),
        lambda x, y, z: 27*x*y*z,
        lambda x, y, z: -27*y*z*(x + y + z - 1),
    ],
    4: [
        lambda x, y, z: (
            helper_0 := x + y + z - 1,
            helper_1 := x*y,
            helper_2 := pow(y, 2),
            helper_3 := 9*x,
            helper_4 := pow(z, 2),
            helper_5 := pow(x, 2),
            helper_6 := 9*y,
            helper_7 := 9*z,
            helper_8 := 26*helper_0,
            helper_9 := helper_8*z,
            helper_10 := 13*pow(helper_0, 2),
            helper_11 := 13*helper_0,
            (1.0/3.0)*helper_0*(
                3*pow(helper_0, 3) + helper_1*helper_8 + 18*helper_1*z
                + helper_10*x + helper_10*y + helper_10*z + helper_11*helper_2
                + helper_11*helper_4 + helper_11*helper_5 + helper_2*helper_3
                + helper_2*helper_7 + helper_3*helper_4 + helper_4*helper_6
                + helper_5*helper_6 + helper_5*helper_7 + helper_9*x
                + helper_9*y + 3*pow(x, 3) + 3*pow(y, 3) + 3*pow(z, 3)),
        )[-1],
        lambda x, y, z: (1.0/3.0)*x*(32*pow(x, 3) - 48*pow(x, 2) + 22*x - 3),
        lambda x, y, z: (1.0/3.0)*y*(32*pow(y, 3) - 48*pow(y, 2) + 22*y - 3),
        lambda x, y, z: (1.0/3.0)*z*(32*pow(z, 3) - 48*pow(z, 2) + 22*z - 3),
        lambda x, y, z:(
            helper_0 := 36*x,
            helper_1 := y*z,
            helper_2 := pow(x, 2),
            helper_3 := pow(y, 2),
            helper_4 := pow(z, 2),
            helper_5 := 24*x,
            helper_6 := 24*y,
            helper_7 := 24*z,
            -16.0/3.0*x*(
                -helper_0*y - helper_0*z + 48*helper_1*x - 36*helper_1
                + helper_2*helper_6 + helper_2*helper_7 - 18*helper_2
                + helper_3*helper_5 + helper_3*helper_7 - 18*helper_3
                + helper_4*helper_5 + helper_4*helper_6 - 18*helper_4
                + 8*pow(x, 3) + 13*x + 8*pow(y, 3) + 13*y + 8*pow(z, 3)
                + 13*z - 3)
        )[-1],
        lambda x, y, z: (
            helper_0 := 2*y,
            helper_1 := 2*z,
            helper_2 := x + y + z - 1,
            helper_3 := helper_2*x,
            4*helper_3*(
                -helper_0*helper_2 + helper_0*x - helper_0*z - helper_1*helper_2
                + helper_1*x + 3*pow(helper_2, 2) + 10*helper_3 + 3*pow(x, 2)
                - pow(y, 2) - pow(z, 2))
        )[-1],
        lambda x, y, z:(
            helper_0 := 6*x,
            helper_1 := pow(x, 2),
            helper_2 := 8*helper_1,
            - 16.0/3.0*x*(-helper_0*y - helper_0*z - 14*helper_1 +
                          helper_2*y + helper_2*z + 8*pow(x, 3) + 7*x + y + z - 1)
        )[-1],
        lambda x, y, z: (16.0/3.0)*x*y*(8*pow(x, 2) - 6*x + 1),
        lambda x, y, z: (
            helper_0 := 4*x, helper_0*y*(-helper_0 + 16*x*y - 4*y + 1)
        )[-1],
        lambda x, y, z: (16.0/3.0)*x*y*(8*pow(y, 2) - 6*y + 1),
        lambda x, y, z:(
            helper_0 := 6*y,
            helper_1 := pow(y, 2),
            helper_2 := 8*helper_1,
            -16.0/3.0*y*(-helper_0*x - helper_0*z - 14*helper_1 +
                         helper_2*x + helper_2*z + x + 8*pow(y, 3) + 7*y + z - 1)
        )[-1],
        lambda x, y, z:(
            helper_0 := 2*y,
            helper_1 := 2*x,
            helper_2 := x + y + z - 1,
            helper_3 := helper_2*y,
            - 4*helper_3*(
                -helper_0*x - helper_0*z + helper_1*helper_2 + helper_1*z
                - 3*pow(helper_2, 2) + 2*helper_2*z - 10*helper_3 + pow(x, 2)
                - 3*pow(y, 2) + pow(z, 2))
        )[-1],
        lambda x, y, z:(
            helper_0 := 36*x,
            helper_1 := y*z,
            helper_2 := pow(x, 2),
            helper_3 := pow(y, 2),
            helper_4 := pow(z, 2),
            helper_5 := 24*x,
            helper_6 := 24*y,
            helper_7 := 24*z,
            -16.0/3.0*y*(
                -helper_0*y - helper_0*z + 48*helper_1*x - 36*helper_1
                + helper_2*helper_6 + helper_2*helper_7 - 18*helper_2
                + helper_3*helper_5 + helper_3*helper_7 - 18*helper_3
                + helper_4*helper_5 + helper_4*helper_6 - 18*helper_4
                + 8*pow(x, 3) + 13*x + 8*pow(y, 3) + 13*y + 8*pow(z, 3)
                + 13*z - 3)
        )[-1],
        lambda x, y, z:(
            helper_0 := 36*x,
            helper_1 := y*z,
            helper_2 := pow(x, 2),
            helper_3 := pow(y, 2),
            helper_4 := pow(z, 2),
            helper_5 := 24*x,
            helper_6 := 24*y,
            helper_7 := 24*z,
            -16.0/3.0*z*(
                -helper_0*y - helper_0*z + 48*helper_1*x - 36*helper_1
                + helper_2*helper_6 + helper_2*helper_7 - 18*helper_2
                + helper_3*helper_5 + helper_3*helper_7 - 18*helper_3
                + helper_4*helper_5 + helper_4*helper_6 - 18*helper_4
                + 8*pow(x, 3) + 13*x + 8*pow(y, 3) + 13*y + 8*pow(z, 3)
                + 13*z - 3)
        )[-1],
        lambda x, y, z:(
            helper_0 := 2*x,
            helper_1 := 2*z,
            helper_2 := x + y + z - 1,
            helper_3 := helper_2*z,
            -4*helper_3*(helper_0*helper_2 + helper_0*y - helper_1*x - helper_1*y - 3*pow(
                helper_2, 2) + 2*helper_2*y - 10*helper_3 + pow(x, 2) + pow(y, 2) - 3*pow(z, 2))
        )[-1],
        lambda x, y, z: (
            helper_0 := 6*z,
            helper_1 := pow(z, 2),
            helper_2 := 8*helper_1,
            -16.0/3.0*z*(-helper_0*x - helper_0*y - 14*helper_1 +
                         helper_2*x + helper_2*y + x + y + 8*pow(z, 3) + 7*z - 1)
        )[-1],
        lambda x, y, z: (16.0/3.0)*x*z*(8*pow(x, 2) - 6*x + 1),
        lambda x, y, z:(
            helper_0 := 4*x, helper_0*z*(-helper_0 + 16*x*z - 4*z + 1))[-1],
        lambda x, y, z: (16.0/3.0)*x*z*(8*pow(z, 2) - 6*z + 1),
        lambda x, y, z: (16.0/3.0)*y*z*(8*pow(y, 2) - 6*y + 1),
        lambda x, y, z:(
            helper_0 := 4*y,
            helper_0*z*(-helper_0 + 16*y*z - 4*z + 1)
        )[-1],
        lambda x, y, z: (16.0/3.0)*y*z*(8*pow(z, 2) - 6*z + 1),
        lambda x, y, z: 32*x*y*(x + y + z - 1)*(4*x + 4*y + 4*z - 3),
        lambda x, y, z: -32*x*y*(4*y - 1)*(x + y + z - 1),
        lambda x, y, z: -32*x*y*(4*x - 1)*(x + y + z - 1),
        lambda x, y, z: 32*x*z*(x + y + z - 1)*(4*x + 4*y + 4*z - 3),
        lambda x, y, z: -32*x*z*(4*z - 1)*(x + y + z - 1),
        lambda x, y, z: -32*x*z*(4*x - 1)*(x + y + z - 1),
        lambda x, y, z: 32*x*y*z*(4*x - 1),
        lambda x, y, z: 32*x*y*z*(4*z - 1),
        lambda x, y, z: 32*x*y*z*(4*y - 1),
        lambda x, y, z: -32*y*z*(4*y - 1)*(x + y + z - 1),
        lambda x, y, z: -32*y*z*(4*z - 1)*(x + y + z - 1),
        lambda x, y, z: 32*y*z*(x + y + z - 1)*(4*x + 4*y + 4*z - 3),
        lambda x, y, z: -256*x*y*z*(x + y + z - 1),
    ]
}

edges_3D_order = numpy.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
faces_3D_order = numpy.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [3, 2, 0]])
face_node_3D_order = {
    3: numpy.zeros(4, dtype=int),
    4: numpy.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [2, 0, 1]])
}


nodes_per_edge = {1: 0, 2: 1, 3: 2, 4: 3}
nodes_per_face = {1: 0, 2: 0, 3: 1, 4: 3}
nodes_per_cell = {1: 0, 2: 0, 3: 0, 4: 1}
nodes_per_element = {1: 4, 2: 10, 3: 20, 4: 35}

nodes_count_to_order = {4: 1, 10: 2, 20: 3, 35: 4}


def edge_nodes(v1, v2, order):
    if order == 1:
        return []
    if order == 2:
        return [(v1 + v2) / 2]
    elif order == 3:
        return [v1 + (v2 - v1) / 3, v1 + 2 * (v2 - v1) / 3]
    elif order == 4:
        return [
            v1 + (v2 - v1) * 0.25,
            v1 + (v2 - v1) * 0.5,
            v1 + (v2 - v1) * 0.75
        ]
    else:
        raise NotImplementedError(f"P{order} not implemented!")


def face_nodes(v1, v2, v3, order):
    if order <= 2:
        return []
    elif order == 3:
        return numpy.array([(v1 + v2 + v3) / 3])
    elif order == 4:
        return numpy.array([
            0.5 * v1 + 0.25 * v2 + 0.25 * v3,
            0.25 * v1 + 0.25 * v2 + 0.5 * v3,
            0.25 * v1 + 0.5 * v2 + 0.25 * v3,
        ])
    else:
        raise NotImplementedError(f"P{order} not implemented!")


def cell_nodes(v1, v2, v3, v4, order):
    if order <= 3:
        return []
    elif order == 4:
        return [(v1 + v2 + v3 + v4) / 4]
    else:
        raise NotImplementedError(f"P{order} not implemented!")


gmsh_to_basis_order = {
    1: [0, 1, 2, 3],
    2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    3: [0, 1, 2, 3,  # vertices
        4, 5,    6, 7,    8, 9,    11, 10,    15, 14,    13, 12,  # edges
        16, 17, 19, 18],  # faces
    4: [0, 1, 2, 3,  # vertices
        4, 5, 6,  7, 8, 9,  10, 11, 12,  15, 14, 13,  21, 20, 19,  18, 17, 16,  # edges
        22, 23, 24,  25, 27, 26,  32, 31, 33,  30, 29, 28,   # faces
        34],  # cell
}

basis_order_to_gmsh = {
    1: [0, 1, 2, 3],
    2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10, 15, 14, 13, 12, 16, 17, 19, 18],
    4: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 14, 13, 21, 20, 19, 18,
        17, 16, 22, 23, 24, 25, 27, 26, 33, 32, 31, 29, 28, 30, 34],
}


if __name__ == "__main__":
    import itertools

    order = 2

    V = numpy.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=float)

    E = edges_3D_order
    EV = []
    for e in E:
        EV.append(edge_nodes(*V[e], order))
    EV = numpy.vstack(EV)

    F = faces_3D_order
    if order > 2:
        FV = []
        for fi, f in enumerate(F):
            FV.append(face_nodes(*V[f], order)[face_node_3D_order[order][fi]])
        FV = numpy.array(FV)

    CV = numpy.array(cell_nodes(*V, order))

    if order == 2:
        N = numpy.vstack([V, EV])
    elif order == 3:
        N = numpy.vstack([V, EV, FV])
    elif order == 4:
        N = numpy.vstack([V, EV, FV, CV])
        assert(N.shape[0] == 35)
    print([hat_phis_3D[order][i](*n) for i, n in enumerate(N)])
    # if all([abs(hat_phis_3D[order][i](*n) - 1) < 1e-12 for i, n in enumerate(N)]):
    #     print(F)
