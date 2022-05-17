import numpy as np
import torch
from tqdm import tqdm

from weights.bases import hat_phis_3D, gmsh_to_basis_order


def p_eval(order, local_index, uv):
    return hat_phis_3D[order][local_index](uv[-3:])


def add_tet(tmp, V):
    if tmp[0] >= 0 and tmp[1] >= 0 and tmp[2] >= 0 and tmp[3] >= 0:
        e0 = V[tmp[1], :] - V[tmp[0], :]
        e1 = V[tmp[2], :] - V[tmp[0], :]
        e2 = V[tmp[3], :] - V[tmp[0], :]

        vol = np.dot(np.cross(e0, e1), e2)
        if vol < 0:
            T = np.array([tmp[0], tmp[1], tmp[2], tmp[3]])
        else:
            T = np.array([tmp[0], tmp[1], tmp[3], tmp[2]])

        return T
    else:
        return None


def sample_tet(nn):
    n = nn
    delta = 1. / (n - 1.)

    T = np.zeros(((n - 1) * (n - 1) * (n - 1) * 6, 4))
    V = np.zeros((n * n * n, 3))
    mmap = -np.ones(n * n * n, dtype=int)

    index = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i + j + k >= n:
                    continue
                mmap[(i + j * n) * n + k] = index
                V[index, :] = [i * delta, j * delta, k * delta]
                index += 1

    V = V[:index, :]

    index = 0
    for i in range(n-1):
        for j in range(n-1):
            for k in range(n-1):
                indices = [(i + j * n) * n + k,
                           (i + 1 + j * n) * n + k,
                           (i + 1 + (j + 1) * n) * n + k,
                           (i + (j + 1) * n) * n + k,
                           (i + j * n) * n + k + 1,
                           (i + 1 + j * n) * n + k + 1,
                           (i + 1 + (j + 1) * n) * n + k + 1,
                           (i + (j + 1) * n) * n + k + 1]

                tmp = [mmap[indices[1 - 1]], mmap[indices[2 - 1]],
                       mmap[indices[4 - 1]], mmap[indices[5 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

                tmp = [mmap[indices[6 - 1]], mmap[indices[3 - 1]],
                       mmap[indices[7 - 1]], mmap[indices[8 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

                tmp = [mmap[indices[5 - 1]], mmap[indices[2 - 1]],
                       mmap[indices[6 - 1]], mmap[indices[4 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

                tmp = [mmap[indices[5 - 1]], mmap[indices[4 - 1]],
                       mmap[indices[8 - 1]], mmap[indices[6 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

                tmp = [mmap[indices[4 - 1]], mmap[indices[2 - 1]],
                       mmap[indices[6 - 1]], mmap[indices[3 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

                tmp = [mmap[indices[3 - 1]], mmap[indices[4 - 1]],
                       mmap[indices[8 - 1]], mmap[indices[6 - 1]]]
                tmp = add_tet(tmp, V)
                if tmp is not None:
                    T[index, :] = tmp
                    index += 1

    T = T[:index, :]

    return V, T


def gmapping(order, uv, pts):
    if type(pts) is np.ndarray or type(pts) is np.array:
        res = np.zeros_like(uv)
    else:
        res = torch.zeros_like(uv)

    rr = gmsh_to_basis_order[order]

    for i in range(pts.shape[0]):
        bb = p_eval(order, i, uv)
        if len(bb.shape) == 0:
            res += pts[rr[i], :] * bb
        else:
            for d in range(3):
                res[:, d] += pts[rr[i], d] * bb

    return res


def gmapping_energy(order, uv, pts, target):
    img_p = gmapping(order, uv, pts)

    if type(pts) is np.array or type(pts) is np.ndarray:
        return np.linalg.norm(img_p - target)**2
    else:
        return torch.norm(img_p - target)**2


def invert_gmapping(order, uv0, pts, target, eps=1e-10):
    tuv0 = torch.tensor(uv0, requires_grad=True, dtype=torch.float64)
    tpts = torch.tensor(pts, dtype=torch.float64)
    tt = torch.tensor(target, dtype=torch.float64)

    optimizer = torch.torch.optim.LBFGS([tuv0],
                                        lr=0.1,
                                        history_size=20,
                                        max_iter=10,
                                        line_search_fn="strong_wolfe")

    def f(uv): return gmapping_energy(order, uv, tpts, tt)
    vv = []

    for i in tqdm(range(1000)):
        optimizer.zero_grad()
        objective = f(tuv0)
        objective.backward()
        optimizer.step(lambda: f(tuv0))
        vv.append(objective.item())

        if objective.item() < eps:
            break

    return tuv0.detach().numpy(), vv
