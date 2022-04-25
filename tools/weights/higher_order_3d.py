from time import ctime
import numpy as np
import igl
import meshio
import matplotlib.pyplot as plt
import h5py
import argparse
import pathlib
import itertools
import scipy
from numba import njit, prange

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

def hat_phi_1_0(x, y):
    return -x-y+1
def hat_phi_1_1(x, y):
    return x
def hat_phi_1_2(x, y):
    return y

def hat_phi_2_0(x, y):
    return (x + y - 1)*(2*x + 2*y - 1)
def hat_phi_2_1(x, y):
    return x*(2*x - 1)
def hat_phi_2_2(x, y):
    return y*(2*y - 1)
def hat_phi_2_3(x, y):
    return -4*x*(x + y - 1)
def hat_phi_2_4(x, y):
    return 4*x*y
def hat_phi_2_5(x, y):
    return -4*y*(x + y - 1)

def hat_phi_3_0(x, y):
    return -27.0/2.0*x**2*y + 9*x**2 - 27.0/2.0*y**2*x + 9*y**2 - 9.0/2.0*x**3 + 18*x*y - 11.0/2.0*x - 9.0/2.0*y**3 - 11.0/2.0*y + 1
def hat_phi_3_1(x, y):
    return (1.0/2.0)*x*(9*x**2 - 9*x + 2)
def hat_phi_3_2(x, y):
    return (1.0/2.0)*y*(9*y**2 - 9*y + 2)
def hat_phi_3_3(x, y):
    return (9.0/2.0)*x*(x + y - 1)*(3*x + 3*y - 2)
def hat_phi_3_4(x, y):
    return -9.0/2.0*x*(3*x**2 + 3*x*y - 4*x - y + 1)
def hat_phi_3_5(x, y):
    return (9.0/2.0)*x*y*(3*x - 1)
def hat_phi_3_6(x, y):
    return (9.0/2.0)*x*y*(3*y - 1)
def hat_phi_3_7(x, y):
    return -9.0/2.0*y*(3*x*y - x + 3*y**2 - 4*y + 1)
def hat_phi_3_8(x, y):
    return (9.0/2.0)*y*(x + y - 1)*(3*x + 3*y - 2)
def hat_phi_3_9(x, y):
    return -27*x*y*(x + y - 1)

def hat_phis(order):
    if order == 1:
        return [hat_phi_1_0, hat_phi_1_1, hat_phi_1_2]
    elif order == 2:
        return [hat_phi_2_0, hat_phi_2_1, hat_phi_2_2, hat_phi_2_3, hat_phi_2_4, hat_phi_2_5]
    elif order == 3:
        return [hat_phi_3_0, hat_phi_3_1, hat_phi_3_2, hat_phi_3_3, hat_phi_3_4, hat_phi_3_5, hat_phi_3_6, hat_phi_3_7, hat_phi_3_8, hat_phi_3_9]

@njit
def findE(e, E):
    for i in prange(E.shape[0]):
    # for i, ei in enumerate(E):
        if (e == E[i]).all() or (e == E[i][::-1]).all():
            return i
    raise Exception()

def get_highorder_nodes(V, F, E, order):
    V_new = V.tolist()

    counter = len(V_new)
    edge_to_edge_node = {}
    face_to_face_node = {}
    if order == 2:
        for i in range(F.shape[0]):
            v1 = V[F[i][0]]
            v2 = V[F[i][1]]
            v3 = V[F[i][2]]

            e1 = findE(np.array([F[i][0], F[i][1]]), E)
            e2 = findE(np.array([F[i][1], F[i][2]]), E)
            e3 = findE(np.array([F[i][0], F[i][2]]), E)

            if e1 not in edge_to_edge_node.keys():
                V_new.append((v1+v2)/2)
                edge_to_edge_node[e1] = counter
                counter += 1

            if e2 not in edge_to_edge_node.keys():
                V_new.append((v2+v3)/2)
                edge_to_edge_node[e2] = counter
                counter += 1

            if e3 not in edge_to_edge_node.keys():
                V_new.append((v1+v3)/2)
                edge_to_edge_node[e3] = counter
                counter += 1

    elif order == 3:
        for i in range(F.shape[0]):
            v1 = V[F[i][0]]
            v2 = V[F[i][1]]
            v3 = V[F[i][2]]

            e1 = findE(np.array([F[i][0], F[i][1]]), E)
            e2 = findE(np.array([F[i][1], F[i][2]]), E)
            e3 = findE(np.array([F[i][0], F[i][2]]), E)

            if e1 not in edge_to_edge_node.keys():
                V_new.append(v1 + (v2-v1)/3)
                V_new.append(v1 + 2*(v2-v1)/3)
                edge_to_edge_node[e1] = counter
                counter += 2

            if e2 not in edge_to_edge_node.keys():
                V_new.append(v2 + (v3-v2)/3)
                V_new.append(v2 + 2*(v3-v2)/3)
                edge_to_edge_node[e2] = counter
                counter += 2

            if e3 not in edge_to_edge_node.keys():
                V_new.append(v3 + (v1-v3)/3)
                V_new.append(v3 + 2*(v1-v3)/3)
                edge_to_edge_node[e3] = counter
                counter += 2

            V_new.append((v1+v2+v3)/3)
            face_to_face_node[i] = counter
            counter += 1
            
    return np.array(V_new), edge_to_edge_node, face_to_face_node

def num_ho_nodes(order):
    if order == 1:
        return 3
    elif order == 2:
        return 6
    elif order == 3:
        return 10

def regular_2d_grid(n):
    V = []
    F = []
    delta = 1. / (n - 1.)
    map = -1*np.ones(n*n, dtype=int)

    index = 0
    for i in range(n):
        for j in range(n):
            if i+j >= n:
                continue

            map[i + j * n] = index
            V.append([i * delta, j * delta])
            index += 1

    for i in range(n-1):
        for j in range(n-1):
            tmp = np.array([map[i + j * n], map[i + 1 + j * n], map[i + (j + 1) * n]], dtype=int)
            if np.all(tmp >= 0):
                F.append(tmp)

            tmp = np.array([map[i + 1 + j * n], map[i + 1 + (j + 1) * n], map[i + (j + 1) * n]], dtype=int)
            if np.all(tmp >= 0):
                F.append(tmp)
	
    return np.array(V), np.array(F)

def get_phi_3d(num_nodes, n_vertex_nodes, F, E, edge_to_edge_node, face_to_face_node, order, div_per_edge):
    assert(div_per_edge > 2)

    V_grid, F_grid = regular_2d_grid(div_per_edge)

    alpha = V_grid[:, 0]
    beta =  V_grid[:, 1]

    num_ho = num_ho_nodes(order)
    n_el = F.shape[0]

    phi = np.zeros(((alpha.shape[0])*n_el, num_nodes))

    ct = 0
    F_col = F_grid
    for f in range(n_el):
        e1 = findE(np.array([F[f][0], F[f][1]]), E)
        e2 = findE(np.array([F[f][1], F[f][2]]), E)
        e3 = findE(np.array([F[f][0], F[f][2]]), E)
        if order == 2:
            v_indices = [F[f][0], F[f][1], F[f][2]] + [edge_to_edge_node[e1], edge_to_edge_node[e2], edge_to_edge_node[e3]]

        elif order == 3:
            v_indices = [F[f][0], F[f][1], F[f][2]] + [edge_to_edge_node[e1], edge_to_edge_node[e1]+1, edge_to_edge_node[e2], edge_to_edge_node[e2]+1, edge_to_edge_node[e3], edge_to_edge_node[e3]+1, face_to_face_node[f]]
        
        for i in range(num_ho):
            val = hat_phis(order)[i](alpha, beta)
            ind = v_indices[i]

            phi[f*alpha.shape[0]:(f+1)*alpha.shape[0], ind] = val

        if f > 0:
            ct += V_grid.shape[0]
            F_col = np.append(F_col, F_grid+ct, axis=0)

        print(F_col)

    return np.array(phi), F_col

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=pathlib.Path)
    parser.add_argument('-o,--order', dest="order",
                        default="2", choices=["2", "3"])
    parser.add_argument('-m', dest="div_per_edge",
                        default="10")

    args = parser.parse_args()

    # Triangular mesh
    mesh = meshio.read(args.mesh)
    assert(mesh.cells[0].type == "tetra")
    T = np.array(mesh.cells[0].data)
    V = np.array(mesh.points)
    n_vertex_nodes = V.shape[0]

    order = int(args.order)
    div_per_edge = int(args.div_per_edge)

    F_boundary_global = igl.boundary_facets(T)

    # get vertices and according face indices just for the boundary
    V_boundary, F_boundary, _, _ = igl.remove_unreferenced(V, F_boundary_global)

    F_boundary_edges = igl.edges(F_boundary)

    # insert higher order indices at end
    V_fem, edge_to_edge_node, face_to_face_node = get_highorder_nodes(V_boundary, F_boundary, F_boundary_edges, order)

    # get phi matrix
    phi, F_col = get_phi_3d(V_fem.shape[0], n_vertex_nodes, F_boundary, F_boundary_edges, edge_to_edge_node, face_to_face_node, order, div_per_edge)
    phi, _, _, F_col = igl.remove_duplicate_vertices(phi, F_col, 1e-7) # Removing duplicate rows (kind of hacky)

    save_weights("phi.hdf5", phi)

    # compute collision matrix
    V_col = phi @ V_fem

    write_obj("fem_mesh.obj", V_fem, F=F_boundary)
    write_obj("coll_mesh.obj", V_col, F=F_col)

if __name__ == '__main__':
    main()
