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
    ]
}

edges_3D_order = numpy.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
# The order of the vertices in the face is not verified (need P4)
faces_3D_order = numpy.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])

if __name__ == "__main__":
    import itertools

    V = numpy.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=float)

    E = edges_3D_order
    EV = numpy.empty((12, 3))
    EV[::2] = ((V[E[:, 1]] - V[E[:, 0]]) * 1/3 + V[E[:, 0]])
    EV[1::2] = ((V[E[:, 1]] - V[E[:, 0]]) * 2/3 + V[E[:, 0]])

    F = faces_3D_order
    FV = V[F].sum(axis=1) / 3

    # for order in itertools.permutations(numpy.arange(4)):
    #     N = numpy.vstack([V, EV, FV[numpy.array(order)]])
    #     if all([abs(hat_phis_3D[3][i](*n) - 1) < 1e-12 for i, n in enumerate(N)]):
    #         print(F[numpy.array(order)])
    #         # break
    N = numpy.vstack([V, EV, FV])
    print([hat_phis_3D[3][i](*n) for i, n in enumerate(N)])
