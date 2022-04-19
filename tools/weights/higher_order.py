import numpy as np
import igl
import meshio
import matplotlib.pyplot as plt
import h5py
import argparse
import pathlib
import scipy

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

def write_obj(filename, V, E):
    with open(filename, 'w') as f:
        for i in range(V.shape[0]):
            s = 'v {:.16f} {:.16f} {:.16f}\n'.format(V[i, 0], V[i, 1], V[i, 2])
            f.write(s)
        
        f.write('l')
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                s = ' {} '.format(E[i,j])
                f.write(s)
        f.write('\n') 

def hat_phi_1_0(x):
    return 1-x
def hat_phi_1_1(x):
    return x
def hat_phi_2_0(x):
    return 2*(x-0.5)*(x-1)
def hat_phi_2_1(x):
    return 2*(x-0)*(x-0.5)
def hat_phi_2_2(x):
    return -4*(x-0.5)**2+1
def hat_phi_3_0(x):
    return -9/2*(x-1/3)*(x-2/3)*(x-1)
def hat_phi_3_1(x):
    return 9/2*(x-0)*(x-1/3)*(x-2/3)
def hat_phi_3_2(x):
    return 27/2*(x-0)*(x-2/3)*(x-1)
def hat_phi_3_3(x):
    return -27/2*(x-0)*(x-1/3)*(x-1)

def hat_phis(order):
    if order == 1:
        return [hat_phi_1_0, hat_phi_1_1]
    elif order == 2:
        return [hat_phi_2_0, hat_phi_2_1, hat_phi_2_2]
    elif order == 3:
        return [hat_phi_3_0, hat_phi_3_1, hat_phi_3_2, hat_phi_3_3]

def get_highorder_nodes(V, E, order):
    V_new = V.tolist()

    if order == 2:
        for i in range(E.shape[0]):
            v1 = V[E[i][0]]
            v2 = V[E[i][1]]

            V_new.append((v1+v2)/2)

    elif order == 3:
        for i in range(E.shape[0]):
            v1 = V[E[i][0]]
            v2 = V[E[i][1]]

            V_new.append(v1 + (v2-v1)/3)
            V_new.append(v1 + 2*(v2-v1)/3)

    return np.array(V_new)

def get_local_elements(E, V_orig):
    V = []
    d = {}
    counter = 0
    E_local = []
    for i in range(E.shape[0]):
        temp = []
        for j in range(2):
            if E[i,j] in d.keys():
                temp.append(d[E[i,j]])
            else:
                temp.append(counter)
                V.append(V_orig[E[i,j]])
                d[E[i,j]] = counter
                counter += 1
        E_local.append(temp)
    
    return np.array(V), np.array(E_local)
    

def get_phi_2d(E, order, div_per_edge): # Default to dividing each edge in two parts
    assert 0 < order < 4

    n_el = E.shape[0]
    alpha = np.linspace(0, 1, div_per_edge)
    # alpha = alpha[1:-1]

    num_nodes = order*n_el
    phi = np.zeros(((alpha.shape[0])*n_el, num_nodes))
    # phi = []

    for e in range(n_el):
        v_indices = [E[e][0], E[e][1]] + list(range(n_el + e*(order-1), n_el + e*(order-1) + order-1))
        for i in range(order + 1):
            val = hat_phis(order)[i](alpha)
            ind = v_indices[i]

            phi[e*alpha.shape[0]:(e+1)*alpha.shape[0], ind] = val

    return np.array(phi)

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
    assert(mesh.cells[0].type == "triangle")
    F = np.array(mesh.cells[0].data)
    V_orig = np.array(mesh.points)

    order = int(args.order)
    div_per_edge = int(args.div_per_edge)

    E = igl.boundary_facets(F)

    # get vertices and according edge indices just for the boundary
    V, E_local = get_local_elements(E, V_orig) 

    # insert higher order indices at end
    V = get_highorder_nodes(V, E_local, order)

    # get phi matrix
    phi = get_phi_2d(E_local, order, div_per_edge)

    save_weights("phi.hdf5", phi)
    write_obj("mesh.obj", V, E_local)

    # # compute collision matrix
    # V_col = phi @ V

    # # test code to visualise
    # plt.scatter(V[:,0], V[:,1])
    # plt.scatter(V_col[:,0], V_col[:,1])
    # plt.show()

if __name__ == '__main__':
    main() # input is the order
