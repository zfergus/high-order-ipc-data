import numpy as np
import igl
import meshio
import h5py
import argparse
import pathlib
import scipy.sparse


def save_weights(path, W):
    h5f = h5py.File(path, 'w')

    if scipy.sparse.issparse(W):
        # Saving as sparse matrix
        W_coo = W.tocoo()
        g = h5f.create_group('weight_triplets')
        g.create_dataset('values', data=W_coo.data)
        g.create_dataset('rows', data=W_coo.row)
        g.create_dataset('cols', data=W_coo.col)
        g.attrs['shape'] = W_coo.shape
    else:
        h5f.create_dataset('weights', data=W)

    h5f.close()


def write_obj(filename, V, E=None, F=None):
    with open(filename, 'w') as f:
        for v in V:
            f.write("v {:.16f} {:.16f} {:.16f}\n".format(*v))
        if E is not None:
            for e in E:
                f.write("l {:d} {:d}\n".format(*(e + 1)))
        if F is not None:
            for face in F:
                f.write("f {:d} {:d} {:d}\n".format(*(face + 1)))


def hat_phi_1_0(x):
    return 1 - x


def hat_phi_1_1(x):
    return x


def hat_phi_2_0(x):
    return 2 * (x - 0.5) * (x - 1)


def hat_phi_2_1(x):
    return 2 * (x - 0) * (x - 0.5)


def hat_phi_2_2(x):
    return -4 * (x - 0.5)**2 + 1


def hat_phi_3_0(x):
    return -9 / 2 * (x - 1 / 3) * (x - 2 / 3) * (x - 1)


def hat_phi_3_1(x):
    return 9 / 2 * (x - 0) * (x - 1 / 3) * (x - 2 / 3)


def hat_phi_3_2(x):
    return 27 / 2 * (x - 0) * (x - 2 / 3) * (x - 1)


def hat_phi_3_3(x):
    return -27 / 2 * (x - 0) * (x - 1 / 3) * (x - 1)


def hat_phis(order):
    if order == 1:
        return [hat_phi_1_0, hat_phi_1_1]
    elif order == 2:
        return [hat_phi_2_0, hat_phi_2_1, hat_phi_2_2]
    elif order == 3:
        return [hat_phi_3_0, hat_phi_3_1, hat_phi_3_2, hat_phi_3_3]
    else:
        raise NotImplementedError()


def get_highorder_nodes(V, F, order):
    E = igl.edges(F)
    V_new = V.tolist()
    for e in E:
        v1, v2 = V[e]
        if order == 2:
            V_new.append((v1 + v2) / 2)
        elif order == 3:
            V_new.append(v1 + (v2 - v1) / 3)
            V_new.append(v1 + 2 * (v2 - v1) / 3)
        else:
            raise NotImplementedError(f"P{order} not implemented!")
    for f in F:
        v1, v2, v3 = V[f]
        if order == 2:
            break
        elif order == 3:
            V_new.append((v1 + v2 + v3) / 3)
        else:
            raise NotImplementedError(f"P{order} not implemented!")
    return np.array(V_new)


def get_phi_2d(num_nodes, n_vertex_nodes, E, E_boundary_to_E_full, order, div_per_edge):
    assert(div_per_edge > 2)
    alpha = np.linspace(0, 1, div_per_edge)
    alpha = alpha[1:-1]

    n_el = E.shape[0]
    num_boundary_vertices = len(set(E.flatten()))
    num_coll_nodes = num_boundary_vertices + n_el * (div_per_edge - 2)
    n_edge_nodes = order - 1

    E_col = np.empty((n_el * (div_per_edge - 1), 2), dtype=int)

    V_edge_to_V_collision = {}
    for e in E:
        for i in e:
            if i not in V_edge_to_V_collision:
                V_edge_to_V_collision[i] = len(V_edge_to_V_collision)
    assert(len(V_edge_to_V_collision) == num_boundary_vertices)

    phi = np.zeros((num_coll_nodes, num_nodes))
    for fem_i, coll_i in V_edge_to_V_collision.items():
        phi[coll_i, fem_i] = 1

    start_vi = num_boundary_vertices
    delta_vi = alpha.size
    start_ei = 0
    delta_ei = (div_per_edge - 1)
    for ei, e in enumerate(E):
        v_indices = np.append(
            e, np.array(range(n_edge_nodes)) + n_vertex_nodes + E_boundary_to_E_full[ei] * n_edge_nodes)

        for i in range(order + 1):
            val = hat_phis(order)[i](alpha)
            ind = v_indices[i]
            phi[start_vi:(start_vi + delta_vi), ind] = val

        E_col[start_ei] = [V_edge_to_V_collision[e[0]], start_vi]
        start_ei += 1
        for i in range(delta_ei - 2):
            E_col[start_ei + i] = [start_vi + i, start_vi + i + 1]
        start_ei += delta_ei - 2
        E_col[start_ei] = [start_vi + delta_vi - 1, V_edge_to_V_collision[e[1]]]
        start_ei += 1

        start_vi += delta_vi

    return phi, E_col


def find(e, E):
    for i, ei in enumerate(E):
        if (e == ei).all() or (e == ei[::-1]).all():
            return i
    raise Exception()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=pathlib.Path)
    parser.add_argument('-o,--order', dest="order", type=int,
                        default="2", choices=[2, 3])
    parser.add_argument('-m', dest="div_per_edge", type=int,
                        default=10)

    args = parser.parse_args()

    # Triangular mesh
    mesh = meshio.read(args.mesh)
    assert(mesh.cells[0].type == "triangle")
    F = mesh.cells[0].data
    V = mesh.points
    n_vertex_nodes = V.shape[0]

    E_full = igl.edges(F)
    E_boundary = igl.boundary_facets(F)
    boundary_to_full = [find(e, E_full) for e in E_boundary]

    # insert higher order indices at end
    V = get_highorder_nodes(V, F, args.order)

    # get phi matrix
    phi, E_col = get_phi_2d(V.shape[0], n_vertex_nodes, E_boundary,
                            boundary_to_full, args.order, args.div_per_edge)

    # center = np.zeros(3)
    # radius = 0.5
    # for i in range(E_boundary.shape[0]):
    #     for j in range(args.order - 1):
    #         vi = n_vertex_nodes + (args.order - 1) * boundary_to_full[i] + j
    #         print(vi)
    #         point = V[vi]
    #         center_to_point = point - center
    #         center_to_point /= np.linalg.norm(center_to_point)
    #         V[vi] = center + center_to_point * radius

    save_weights("phi.hdf5", scipy.sparse.csc_matrix(phi))
    write_obj("fem_mesh.obj", V, E_full)

    # compute collision matrix
    V_col = phi @ V

    # test code to visualise
    write_obj("coll_mesh.obj", V_col, E_col)


if __name__ == '__main__':
    main()  # input is the order
