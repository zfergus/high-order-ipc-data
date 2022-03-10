import numpy as np
import meshio
import h5py
import argparse
import pathlib

from scipy.spatial.transform import Rotation as R
from scipy.sparse import csc_matrix

from tqdm import tqdm


def get_bary(p, v):
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

    return bary


def get_matrix(P, V, F):
    """
    p: points of dense mesh; 2d matrix (|V_dense|, 3)
    V: vertices of tet mesh; 2d matrix (|V_tet|, 3)
    F: faces of tet mesh; 2d matrix (|F|, 4)
    returns mapping matrix
    """
    M = np.zeros((P.shape[0], V.shape[0]))
    for idx1, p in enumerate(tqdm(P)):
        flag = 0
        for idx2, f in enumerate(F):
            tet_vert = []
            for i in range(4):
                tet_vert.append(V[f[i]])
            tet_vert = np.array(tet_vert)

            bary = get_bary(p, tet_vert)
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

                bary = get_bary(p, tet_vert)
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

    if True:
        M = get_matrix(v_tri, v_tet, f_tet)
        M_csc = csc_matrix(M)
        print(M)

        h5f = h5py.File(
            f'{args.coarse_mesh.stem}-to-{args.dense_mesh.stem}.hdf5', 'w')
        # h5f.create_dataset('bary_matrix', data=M)
        # h5f.close()

        # Saving as sparse matrix
        g = h5f.create_group('Mcsc')
        g.create_dataset('data',data=M_csc.data)
        g.create_dataset('indptr',data=M_csc.indptr)
        g.create_dataset('indices',data=M_csc.indices)
        g.attrs['shape'] = M_csc.shape
        h5f.close()

    else:
        h5f = h5py.File(
            f'{args.coarse_mesh.stem}-to-{args.dense_mesh.stem}.hdf5', 'r')
        M = h5f['bary_matrix']

    # T = np.array([
    #     [1, 1, 0],
    #     [0, 1, 0],
    #     [0, 0, 1],
    # ])
    # T = R.from_rotvec(np.pi * np.random.random(3)).as_matrix() @ T
    #
    # deformed_v_tet = v_tet @ T.T
    # mesh = meshio.Mesh(deformed_v_tet, [("tetra", f_tet)])
    # mesh.write("deformed_coarse.msh")
    #
    # deformed_v_tri = M @ deformed_v_tet
    # mesh = meshio.Mesh(deformed_v_tri, [("triangle", f_tri)])
    # mesh.write("deformed_dense.obj")

    ############### To read from sparse h5py file ###################
    f = h5py.File(
    	f'{args.coarse_mesh.stem}-to-{args.dense_mesh.stem}.hdf5','r')

    g2 = f['Mcsc']

    M1 = csc_matrix((g2['data'][:],g2['indices'][:],
    	g2['indptr'][:]), g2.attrs['shape'])

    M = np.array(M1.todense())
    ##################################################################

    print("Error:", np.linalg.norm(M @ v_tet - v_tri, np.inf)) # Checks error of mapping



if __name__ == "__main__":
    main()
