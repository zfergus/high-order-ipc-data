import numpy as np
import torch
from tqdm import tqdm

from weights.bases import hat_phis_3D
from weights.utils import labeled_tqdm


def zero_like(x, rtype=None):
    if rtype is None:
        rtype = type(x)
    if rtype is np.ndarray or rtype is np.array:
        return np.zeros_like(x)
    else:
        return torch.zeros_like(x)


def norm_sq(x):
    assert(len(x.shape) == 1)
    return x.dot(x)


def hstack(xs, rtype):
    if rtype is np.ndarray or rtype is np.array:
        return np.hstack(xs)
    else:
        return torch.hstack(xs)


def gmapping(order, uvw, pts):
    if len(uvw.shape) == 1:
        u, v, w = uvw[0], uvw[1], uvw[2]
        Phi = hstack([phi(u, v, w) for phi in hat_phis_3D[order]], type(uvw))
    else:
        u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
        Phi = hstack([
            phi(u, v, w).reshape(-1, 1) for phi in hat_phis_3D[order]],
            type(uvw))
    return Phi @ pts.reshape(-1, 3)


def gmapping_energy(order, uvw, pts, target):
    return 0.5 * norm_sq(target - gmapping(order, uvw, pts))


def invert_gmapping(order, uvw0, pts, target, eps=1e-4, ignore_uvw=True):
    if ignore_uvw:
        uvw = torch.tensor(
            np.full(3, 1/4), requires_grad=True, dtype=torch.float64)
    else:
        uvw = torch.tensor(uvw0, requires_grad=True, dtype=torch.float64)
    tpts = torch.tensor(pts, dtype=torch.float64)
    ttarget = torch.tensor(target, dtype=torch.float64)

    optimizer = torch.torch.optim.LBFGS(
        [uvw], lr=0.5, history_size=10, max_iter=10,
        line_search_fn="strong_wolfe")
    # optimizer = torch.optim.Adam([uvw])

    def f(uvw): return gmapping_energy(order, uvw, tpts, ttarget)

    prev_dist = np.Inf
    for i in range(100 if ignore_uvw else int(1e5)):
        optimizer.zero_grad()
        objective = f(uvw)
        objective.backward()
        # TODO check grad norm uvw.grad
        optimizer.step(lambda: f(uvw))
        dist = np.sqrt(objective.item())
        if dist <= eps or abs(prev_dist - dist) < 1e-10:
            break
        prev_dist = dist

    if dist > eps:
        if ignore_uvw:
            return invert_gmapping(
                order, uvw0, pts, target, eps=eps, ignore_uvw=False)
        else:
            return invert_gmapping(
                order, 10 * np.random.rand(3) - 5, pts, target, eps=eps, ignore_uvw=False)
            # return np.full(3, np.Inf)  # not converges, so dont use this value

    return uvw.detach().numpy()


def closest_point(order, uvw0, pts, target, eps=1e-10):
    uvw = torch.tensor(uvw0, requires_grad=True, dtype=torch.float64)
    pts = torch.tensor(pts, dtype=torch.float64)
    target = torch.tensor(target, dtype=torch.float64)

    optimizer = torch.torch.optim.LBFGS(
        [uvw], lr=0.1, history_size=20, max_iter=10,
        line_search_fn="strong_wolfe")

    def b(d, dhat=1e-5, kappa=1e3):
        if d > 0:
            return torch.tensor(0, requires_grad=True, dtype=torch.float64)
        return kappa * d**2  # * torch.log(d/dhat + 1)

    def f(uvw):
        bc = [1 - sum(uvw)] + list(uvw)
        return (
            gmapping_energy(order, uvw, pts, target)
            + sum([b(1-c) for c in bc])
        )

    # for _ in labeled_tqdm(range(10), "Finding closest point"):
    for _ in range(10):
        optimizer.zero_grad()
        objective = f(uvw)
        objective.backward()
        optimizer.step(lambda: f(uvw))
        if objective.item() < eps:
            break

    uvw = uvw.detach().numpy()
    t = max(1 - uvw.sum(), 0)
    uvw[uvw < 0] = 0
    uvw /= t + uvw.sum()
    return uvw
