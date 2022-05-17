import numpy as np
import torch
from tqdm import tqdm


def pow(x, n):
    if type(x) is np.array or type(x) is np.ndarray or type(x) is np.float64:
        return np.power(x, n)
    else:
        return torch.pow(x, n)


def remapping(order):
    if order == 3:
        return [0, 1, 2, 3,   4, 5,  6, 7,   8, 9,    11, 10,  15, 14,  13, 12,    16, 17, 19, 18]
    elif order == 4:
        return [0, 1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12,  15, 14, 13,  21, 20, 19,  18, 17, 16,      22, 23, 24,  25, 27, 26,  32, 31, 33, 30, 29, 28,   34]


def p3(local_index, uv):
    if len(uv.shape) == 2:
        x = uv[:, 0]
        y = uv[:, 1]
        z = uv[:, 2]
    else:
        x = uv[0]
        y = uv[1]
        z = uv[2]

    if local_index == 0:
        helper_0 = 18*x
        helper_1 = y*z
        helper_2 = pow(x, 2)
        helper_3 = pow(y, 2)
        helper_4 = pow(z, 2)
        helper_5 = (27.0/2.0)*x
        helper_6 = (27.0/2.0)*y
        helper_7 = (27.0/2.0)*z
        result_0 = helper_0*y + helper_0*z - 27*helper_1*x + 18*helper_1 - helper_2*helper_6 - helper_2*helper_7 + 9*helper_2 - helper_3*helper_5 - helper_3*helper_7 + 9 * \
            helper_3 - helper_4*helper_5 - helper_4*helper_6 + 9*helper_4 - 9.0/2.0 * \
            pow(x, 3) - 11.0/2.0*x - 9.0/2.0*pow(y, 3) - \
            11.0/2.0*y - 9.0/2.0*pow(z, 3) - 11.0/2.0*z + 1
    elif local_index == 1:
        result_0 = (1.0/2.0)*x*(9*pow(x, 2) - 9*x + 2)
    elif local_index == 2:
        result_0 = (1.0/2.0)*y*(9*pow(y, 2) - 9*y + 2)
    elif local_index == 3:
        result_0 = (1.0/2.0)*z*(9*pow(z, 2) - 9*z + 2)
    elif local_index == 4:
        result_0 = (9.0/2.0)*x*(x + y + z - 1)*(3*x + 3*y + 3*z - 2)
    elif local_index == 5:
        helper_0 = 3*x
        result_0 = -9.0/2.0*x*(helper_0*y + helper_0 *
                               z + 3*(x**2) - 4*x - y - z + 1)
    elif local_index == 6:
        result_0 = (9.0/2.0)*x*y*(3*x - 1)
    elif local_index == 7:
        result_0 = (9.0/2.0)*x*y*(3*y - 1)
    elif local_index == 8:
        helper_0 = 3*y
        result_0 = -9.0/2.0*y*(helper_0*x + helper_0 *
                               z - x + 3*pow(y, 2) - 4*y - z + 1)
    elif local_index == 9:
        result_0 = (9.0/2.0)*y*(x + y + z - 1)*(3*x + 3*y + 3*z - 2)
    elif local_index == 10:
        result_0 = (9.0/2.0)*z*(x + y + z - 1)*(3*x + 3*y + 3*z - 2)
    elif local_index == 11:
        helper_0 = 3*z
        result_0 = -9.0/2.0*z*(helper_0*x + helper_0 *
                               y - x - y + 3*pow(z, 2) - 4*z + 1)
    elif local_index == 12:
        result_0 = (9.0/2.0)*x*z*(3*x - 1)
    elif local_index == 13:
        result_0 = (9.0/2.0)*x*z*(3*z - 1)
    elif local_index == 14:
        result_0 = (9.0/2.0)*y*z*(3*y - 1)
    elif local_index == 15:
        result_0 = (9.0/2.0)*y*z*(3*z - 1)
    elif local_index == 16:
        result_0 = -27*x*y*(x + y + z - 1)
    elif local_index == 17:
        result_0 = -27*x*z*(x + y + z - 1)
    elif local_index == 18:
        result_0 = 27*x*y*z
    elif local_index == 19:
        result_0 = -27*y*z*(x + y + z - 1)
    else:
        assert(False)

    return result_0


def p4(local_index, uv):
    if len(uv.shape) == 2:
        x = uv[:, 0]
        y = uv[:, 1]
        z = uv[:, 2]
    else:
        x = uv[0]
        y = uv[1]
        z = uv[2]

    if local_index == 0:
        helper_0 = x + y + z - 1
        helper_1 = x*y
        helper_2 = pow(y, 2)
        helper_3 = 9*x
        helper_4 = pow(z, 2)
        helper_5 = pow(x, 2)
        helper_6 = 9*y
        helper_7 = 9*z
        helper_8 = 26*helper_0
        helper_9 = helper_8*z
        helper_10 = 13*pow(helper_0, 2)
        helper_11 = 13*helper_0
        result_0 = (1.0/3.0)*helper_0*(3*pow(helper_0, 3) + helper_1*helper_8 + 18*helper_1*z + helper_10*x + helper_10*y + helper_10*z + helper_11*helper_2 + helper_11*helper_4 + helper_11*helper_5 +
                                       helper_2*helper_3 + helper_2*helper_7 + helper_3*helper_4 + helper_4*helper_6 + helper_5*helper_6 + helper_5*helper_7 + helper_9*x + helper_9*y + 3*pow(x, 3) + 3*pow(y, 3) + 3*pow(z, 3))
    elif local_index == 1:
        result_0 = (1.0/3.0)*x*(32*pow(x, 3) - 48*pow(x, 2) + 22*x - 3)
    elif local_index == 2:
        result_0 = (1.0/3.0)*y*(32*pow(y, 3) - 48*pow(y, 2) + 22*y - 3)
    elif local_index == 3:
        result_0 = (1.0/3.0)*z*(32*pow(z, 3) - 48*pow(z, 2) + 22*z - 3)
    elif local_index == 4:
        helper_0 = 36*x
        helper_1 = y*z
        helper_2 = pow(x, 2)
        helper_3 = pow(y, 2)
        helper_4 = pow(z, 2)
        helper_5 = 24*x
        helper_6 = 24*y
        helper_7 = 24*z
        result_0 = -16.0/3.0*x*(-helper_0*y - helper_0*z + 48*helper_1*x - 36*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 18*helper_2 + helper_3*helper_5 +
                                helper_3*helper_7 - 18*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 18*helper_4 + 8*pow(x, 3) + 13*x + 8*pow(y, 3) + 13*y + 8*pow(z, 3) + 13*z - 3)
    elif local_index == 5:
        helper_0 = 2*y
        helper_1 = 2*z
        helper_2 = x + y + z - 1
        helper_3 = helper_2*x
        result_0 = 4*helper_3*(-helper_0*helper_2 + helper_0*x - helper_0*z - helper_1*helper_2 +
                               helper_1*x + 3*pow(helper_2, 2) + 10*helper_3 + 3*pow(x, 2) - pow(y, 2) - pow(z, 2))
    elif local_index == 6:
        helper_0 = 6*x
        helper_1 = pow(x, 2)
        helper_2 = 8*helper_1
        result_0 = -16.0/3.0*x*(-helper_0*y - helper_0*z - 14*helper_1 +
                                helper_2*y + helper_2*z + 8*pow(x, 3) + 7*x + y + z - 1)
    elif local_index == 7:
        result_0 = (16.0/3.0)*x*y*(8*pow(x, 2) - 6*x + 1)
    elif local_index == 8:
        helper_0 = 4*x
        result_0 = helper_0*y*(-helper_0 + 16*x*y - 4*y + 1)
    elif local_index == 9:
        result_0 = (16.0/3.0)*x*y*(8*pow(y, 2) - 6*y + 1)
    elif local_index == 10:
        helper_0 = 6*y
        helper_1 = pow(y, 2)
        helper_2 = 8*helper_1
        result_0 = -16.0/3.0*y*(-helper_0*x - helper_0*z - 14*helper_1 +
                                helper_2*x + helper_2*z + x + 8*pow(y, 3) + 7*y + z - 1)
    elif local_index == 11:
        helper_0 = 2*y
        helper_1 = 2*x
        helper_2 = x + y + z - 1
        helper_3 = helper_2*y
        result_0 = -4*helper_3*(-helper_0*x - helper_0*z + helper_1*helper_2 + helper_1*z - 3*pow(
            helper_2, 2) + 2*helper_2*z - 10*helper_3 + pow(x, 2) - 3*pow(y, 2) + pow(z, 2))
    elif local_index == 12:
        helper_0 = 36*x
        helper_1 = y*z
        helper_2 = pow(x, 2)
        helper_3 = pow(y, 2)
        helper_4 = pow(z, 2)
        helper_5 = 24*x
        helper_6 = 24*y
        helper_7 = 24*z
        result_0 = -16.0/3.0*y*(-helper_0*y - helper_0*z + 48*helper_1*x - 36*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 18*helper_2 + helper_3*helper_5 +
                                helper_3*helper_7 - 18*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 18*helper_4 + 8*pow(x, 3) + 13*x + 8*pow(y, 3) + 13*y + 8*pow(z, 3) + 13*z - 3)
    elif local_index == 13:
        helper_0 = 36*x
        helper_1 = y*z
        helper_2 = pow(x, 2)
        helper_3 = pow(y, 2)
        helper_4 = pow(z, 2)
        helper_5 = 24*x
        helper_6 = 24*y
        helper_7 = 24*z
        result_0 = -16.0/3.0*z*(-helper_0*y - helper_0*z + 48*helper_1*x - 36*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 18*helper_2 + helper_3*helper_5 +
                                helper_3*helper_7 - 18*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 18*helper_4 + 8*pow(x, 3) + 13*x + 8*pow(y, 3) + 13*y + 8*pow(z, 3) + 13*z - 3)
    elif local_index == 14:
        helper_0 = 2*x
        helper_1 = 2*z
        helper_2 = x + y + z - 1
        helper_3 = helper_2*z
        result_0 = -4*helper_3*(helper_0*helper_2 + helper_0*y - helper_1*x - helper_1*y - 3*pow(
            helper_2, 2) + 2*helper_2*y - 10*helper_3 + pow(x, 2) + pow(y, 2) - 3*pow(z, 2))
    elif local_index == 15:
        helper_0 = 6*z
        helper_1 = pow(z, 2)
        helper_2 = 8*helper_1
        result_0 = -16.0/3.0*z*(-helper_0*x - helper_0*y - 14*helper_1 +
                                helper_2*x + helper_2*y + x + y + 8*pow(z, 3) + 7*z - 1)
    elif local_index == 16:
        result_0 = (16.0/3.0)*x*z*(8*pow(x, 2) - 6*x + 1)
    elif local_index == 17:
        helper_0 = 4*x
        result_0 = helper_0*z*(-helper_0 + 16*x*z - 4*z + 1)
    elif local_index == 18:
        result_0 = (16.0/3.0)*x*z*(8*pow(z, 2) - 6*z + 1)
    elif local_index == 19:
        result_0 = (16.0/3.0)*y*z*(8*pow(y, 2) - 6*y + 1)
    elif local_index == 20:
        helper_0 = 4*y
        result_0 = helper_0*z*(-helper_0 + 16*y*z - 4*z + 1)
    elif local_index == 21:
        result_0 = (16.0/3.0)*y*z*(8*pow(z, 2) - 6*z + 1)
    elif local_index == 22:
        result_0 = 32*x*y*(x + y + z - 1)*(4*x + 4*y + 4*z - 3)
    elif local_index == 23:
        result_0 = -32*x*y*(4*y - 1)*(x + y + z - 1)
    elif local_index == 24:
        result_0 = -32*x*y*(4*x - 1)*(x + y + z - 1)
    elif local_index == 25:
        result_0 = 32*x*z*(x + y + z - 1)*(4*x + 4*y + 4*z - 3)
    elif local_index == 26:
        result_0 = -32*x*z*(4*z - 1)*(x + y + z - 1)
    elif local_index == 27:
        result_0 = -32*x*z*(4*x - 1)*(x + y + z - 1)
    elif local_index == 28:
        result_0 = 32*x*y*z*(4*x - 1)
    elif local_index == 29:
        result_0 = 32*x*y*z*(4*z - 1)
    elif local_index == 30:
        result_0 = 32*x*y*z*(4*y - 1)
    elif local_index == 31:
        result_0 = -32*y*z*(4*y - 1)*(x + y + z - 1)
    elif local_index == 32:
        result_0 = -32*y*z*(4*z - 1)*(x + y + z - 1)
    elif local_index == 33:
        result_0 = 32*y*z*(x + y + z - 1)*(4*x + 4*y + 4*z - 3)
    elif local_index == 34:
        result_0 = -256*x*y*z*(x + y + z - 1)
    else:
        assert(False)

    return result_0


def p_eval(order, local_index, uv):
    if order == 3:
        return p3(local_index, uv)
    elif order == 4:
        return p4(local_index, uv)
    else:
        assert(False)
        return None


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

    rr = remapping(order)

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


def inver_gmapping(order, uv0, pts, target, eps=1e-10):
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
