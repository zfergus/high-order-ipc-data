import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix, vstack, csc_matrix, coo_matrix
import scipy
import sys
import meshio
import igl
import h5py
import argparse
import pathlib

EPSILON = 1e-10
EPSILON_FACE = 1e-10

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
    meshio.write("deformed_coarse.msh", mesh, file_format="gmsh22")

    deformed_v_tri = weights @ deformed_v_tet
    mesh = meshio.Mesh(deformed_v_tri, [("triangle", f_tri)])
    mesh.write("deformed_dense.obj")

def get_imp1(i, complex_dim):
	ip1 = 0
	im1 = 0
	if i+1>=complex_dim:
		ip1 = 0
	else:
		ip1 = i+1
	if i>0:
		im1 = i-1
	else:
		im1 = complex_dim - 1

	return ip1, im1

def MorphPoint(Vertices, Values, point, Faces, nVertices, nFaces, dim=3, complex_dim=3):
	totalF = np.zeros(dim)
	totalW = 0.0

	# weights = np.zeros(nVertices)
	weights = lil_matrix((nVertices, 1))

	for i in range(nVertices):
		d = np.linalg.norm(Vertices[i] - point)
		if d < EPSILON:
			# weights[i] = 1
			weights[(i,0)] = 1
			return Values[i], weights

	for j in range(nFaces):
		vert = np.zeros((complex_dim, dim))
		val = np.zeros((complex_dim, dim))

		for k in range(complex_dim):
			vert[k] = Vertices[Faces[j][k]]
			val[k] = Values[Faces[j][k]]

		u = np.zeros((complex_dim, dim))
		d = np.zeros(complex_dim)

		for i in range(complex_dim):
			d[i] = np.linalg.norm(vert[i] - point)
			u[i] = (vert[i] - point)/d[i]

		l = np.zeros(complex_dim)
		theta = np.zeros(complex_dim)
		c = np.zeros(complex_dim)
		s = np.zeros(complex_dim)
		w = np.zeros(complex_dim)

		for i in range(complex_dim):
			ip1, im1 = get_imp1(i, complex_dim)

			l[i] = np.linalg.norm(u[ip1]-u[im1])

			if abs(l[i] - 2) < EPSILON:
				l[i] = 2
			elif abs(l[i] + 2) < EPSILON:
				l[i] = -2

			theta[i] = 2*np.arcsin(l[i]/2.)

		h = np.sum(theta)/2.

		if np.pi - h < EPSILON_FACE:
			# weights = np.zeros(nVertices)
			weights = lil_matrix((nVertices, 1))
			for i in range(complex_dim):
				ip1, im1 = get_imp1(i, complex_dim)

				w[i] = np.sin(theta[i])*d[ip1]*d[im1]
				# weights[Faces[j][i]] = w[i]
				weights[(Faces[j][i], 0)] = w[i]
				totalF += w[i]*val[i]

			totalF /= np.sum(w)
			return totalF, weights

		flag = 0
		for i in range(complex_dim):
			ip1, im1 = get_imp1(i, complex_dim)

			c[i] = (2*np.sin(h)*np.sin(h-theta[i]))/(np.sin(theta[ip1])*np.sin(theta[im1])) - 1
			s[i] = np.sign(np.linalg.det(u)) * np.sqrt(1-c[i]**2)

			if s[i]!=s[i] or abs(s[i]) < EPSILON:
				flag = 1

		if flag == 1:
			continue

		for i in range(complex_dim):
			ip1, im1 = get_imp1(i, complex_dim)

			w[i] = (theta[i] - c[ip1]*theta[im1] - c[im1]*theta[ip1])/(d[i]*np.sin(theta[ip1])*s[im1])
			# weights[Faces[j][i]] += w[i]
			weights[(Faces[j][i], 0)] += w[i]
			totalF += w[i]*val[i]
			totalW += w[i]

	if totalW == 0:
		print("Error")
		return totalF, weights

	return totalF/totalW, weights

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

	surface_f_tet = igl.boundary_facets(f_tet) # get surface
	surface_f_tri = igl.boundary_facets(f_tri) # get surface

	
	# Handling sparse matrices
	rows = np.array([])
	cols = np.array([])
	data = np.array([])

	M = csc_matrix((v_tri.shape[0], v_tet.shape[0]))
	for i, x in enumerate(v_tri):
		_, w = MorphPoint(v_tet, v_tet, x, surface_f_tet, v_tri.shape[0], surface_f_tet.shape[0])
		w = coo_matrix(w)

		rows = np.append(rows, i*np.ones(w.row.shape[0]))
		cols = np.append(cols, w.row)
		data = np.append(data, w.data/np.sum(w.data))

	M = csc_matrix((data, (rows, cols)), shape=(v_tri.shape[0], v_tet.shape[0])) # Final weight matrix

	print("Error:", np.linalg.norm(M@v_tet-v_tri,np.inf))

	if True:
		save_weights(hdf5_path, M)
	else:
		M = load_weights(hdf5_path)

	# test(v_tet, f_tet, f_tri, M)

if __name__ == '__main__':
	main()