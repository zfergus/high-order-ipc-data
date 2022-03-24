import numpy as np
import meshio
import h5py
import argparse
import pathlib
import scipy
import scipy.sparse

from tqdm import tqdm


def compute_bary(p, v):
    """
    p: point of dense mesh; vector (3)
    v: vertices of single tet; 2d matrix (4, 3)
    returns the bary coords wrt v
    """
    v_ap = p - v[0]
    v_bp = p - v[1]

    v_ab = v[1] - v[0]
    v_ac = v[2] - v[0]
    v_ad = v[3] - v[0]

    v_bc = v[2] - v[1]
    v_bd = v[3] - v[1]

    V_a = 1 / 6 * np.dot(v_bp, np.cross(v_bd, v_bc))
    V_b = 1 / 6 * np.dot(v_ap, np.cross(v_ac, v_ad))
    V_c = 1 / 6 * np.dot(v_ap, np.cross(v_ad, v_ab))
    V_d = 1 / 6 * np.dot(v_ap, np.cross(v_ab, v_ac))
    V = 1 / 6 * np.dot(v_ab, np.cross(v_ac, v_ad))

    bary = np.zeros(4)
    bary[0] = V_a / np.abs(V)
    bary[1] = V_b / np.abs(V)
    bary[2] = V_c / np.abs(V)
    bary[3] = V_d / np.abs(V)

    assert(abs(bary.sum() - 1) < 1e-10)

    return bary


def compute_weights(P, V, F, quiet=True):
    """
    p: points of dense mesh; 2d matrix (|V_dense|, 3)
    V: vertices of tet mesh; 2d matrix (|V_tet|, 3)
    F: faces of tet mesh; 2d matrix (|F|, 4)
    returns mapping matrix
    """
    M = np.zeros((P.shape[0], V.shape[0]))
    if quiet:
        def tqdm(x): return x
    for idx1, p in enumerate(tqdm(P)):
        flag = 0
        for idx2, f in enumerate(F):
            tet_vert = []
            for i in range(4):
                tet_vert.append(V[f[i]])
            tet_vert = np.array(tet_vert)

            bary = compute_bary(p, tet_vert)
            if np.all(bary >= 0):
                flag = 1
                for i in range(4):
                    M[idx1][f[i]] = bary[i]
                break

        # If the vertex is not present inside any tet
        # Finds the tet such that min number of bary coords are negative
        # Possible ISSUE: Doesn't handle the case of tie i.e if for two tets equal number of bary coords are negative elements
        tot_count = 0  # Sets the number of negative bary coords that we require
        while(flag == 0):
            tot_count += 1  # In first itration, tot_count = 1
            for idx2, f in enumerate(F):
                tet_vert = []
                for i in range(4):
                    tet_vert.append(V[f[i]])
                tet_vert = np.array(tet_vert)

                bary = compute_bary(p, tet_vert)
                count = 0
                for i in range(bary.shape[0]):  # counts negative terms
                    if bary[i] < 0:
                        count += 1
                if count == tot_count:
                    flag = 1
                    for i in range(4):
                        M[idx1][f[i]] = bary[i]
                    break

    return M


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


def load_weights(path):
    with h5py.File(path, 'r') as h5f:
        if "weight_triplets" in h5f:
            g = f['weight_triplets']
            return scipy.sparse.coo_matrix(
                (g['values'][:], (g['rows'][:], g['cols'][:])), g.attrs['shape']
            ).tocsc()
        else:
            assert("weights" in h5f)
            h5f.create_dataset('weights', data=W)
            h5f = h5py.File(
                f'{args.coarse_mesh.stem}-to-{args.dense_mesh.stem}.hdf5', 'r')
            return h5f['weights']


def test(v_tet, f_tet, f_tri, weights):
    from scipy.spatial.transform import Rotation as R

    T = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    T = R.from_rotvec(np.pi * np.random.random(3)).as_matrix() @ T

    deformed_v_tet = v_tet @ T.T
    mesh = meshio.Mesh(deformed_v_tet, [("tetra", f_tet)])
    mesh.write("deformed_coarse.msh")

    deformed_v_tri = weights @ deformed_v_tet
    mesh = meshio.Mesh(deformed_v_tri, [("triangle", f_tri)])
    mesh.write("deformed_dense.obj")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dense_mesh', type=pathlib.Path)
    parser.add_argument('coarse_mesh', type=pathlib.Path)

    args = parser.parse_args()

    # Triangular mesh
    mesh = meshio.read(args.dense_mesh)
    f_tri = np.array(mesh.cells[0].data)
    v_tri = np.array(mesh.points)

    # Tetrahedral Mesh
    mesh = meshio.read(args.coarse_mesh)
    f_tet = np.array(mesh.cells[0].data)
    v_tet = np.array(mesh.points)

    hdf5_path = f'{args.coarse_mesh.stem}-to-{args.dense_mesh.stem}.hdf5'

    if True:
        M = get_matrix(v_tri, v_tet, f_tet, quiet=False)
        M_csc = scipy.sparse.csc_matrix(M)
        save_weights(hdf5_path, M_csc)
    else:
        M = load_weights(hdf5_path)

    # Checks error of mapping
    print("Error:", np.linalg.norm(M @ v_tet - v_tri, np.inf))


if __name__ == "__main__":
    main()
