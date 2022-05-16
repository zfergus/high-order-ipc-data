import argparse
import pathlib
import meshio
import numpy
import torch
from tqdm import tqdm


def compute_mass(rho, V, T):
    Vs = V[T, :]
    vol = torch.linalg.det(Vs[:, 1:4, :] - Vs[:, 0:3, :])
    tet_mass = rho * vol
    mass = torch.sum(tet_mass)
    return mass


def compute_com(rho, V, T):
    Vs = V[T, :]
    vol = torch.linalg.det(Vs[:, 1:4, :] - Vs[:, 0:3, :])
    tet_mass = rho * vol
    cm = torch.sum(tet_mass * torch.mean(Vs, axis=1).T, axis=1)
    mass = torch.sum(tet_mass)
    return cm / mass


def compute_energy(rho, target_com, V, T):
    return 0.5 * torch.sum((target_com - compute_com(rho, V, T))**2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('rho', type=float)
    parser.add_argument('ref_mesh', type=pathlib.Path)
    parser.add_argument('mesh', type=pathlib.Path)
    return parser.parse_args()


def main():
    torch.set_printoptions(precision=16)

    args = parse_args()

    ###########################################################################

    ref_mesh = meshio.read(args.ref_mesh)
    assert(len(ref_mesh.cells) == 1 and ref_mesh.cells[0].type == "tetra")
    V = ref_mesh.points
    T = ref_mesh.cells[0].data

    V = torch.tensor(V, dtype=torch.float64)
    T = torch.tensor(T, dtype=torch.int64)

    target_rho = args.rho * torch.ones(T.shape[0], dtype=torch.float64)
    target_com = compute_com(target_rho, V, T).detach()
    target_mass = compute_mass(target_rho, V, T).detach()

    print(f"target:  {target_com.detach().numpy()} {target_mass}")

    ###########################################################################

    mesh = meshio.read(args.mesh)
    assert(len(mesh.cells) == 1 and mesh.cells[0].type == "tetra")
    V = mesh.points
    T = mesh.cells[0].data

    V = torch.tensor(V, dtype=torch.float64)
    T = torch.tensor(T, dtype=torch.int64)

    initial_rho = args.rho * torch.ones(T.shape[0], dtype=torch.float64)
    initial_com = compute_com(initial_rho, V, T).detach()
    initial_mass = compute_mass(initial_rho, V, T).detach()

    print(f"initial: {initial_com.detach().numpy()} {initial_mass}")

    ###########################################################################
    # Optimize
    rho0 = args.rho * numpy.ones(T.shape[0])
    rho0 = torch.tensor(rho0, requires_grad=True, dtype=torch.float64)

    optimizer = torch.optim.Adam([rho0])

    def f(rho):
        return compute_energy(rho, target_com, V, T)

    pbar = tqdm(range(1000))
    loss = []
    for i in pbar:
        optimizer.zero_grad()
        objective = f(rho0)
        pbar.set_description(f"err={objective.item():.5g}")
        objective.backward()
        optimizer.step(lambda: f(rho0))
        loss.append(objective.item())
        if objective.item() == 0:
            break

    opt_rho = rho0.detach()
    opt_mass = compute_mass(opt_rho, V, T).detach()
    opt_rho *= target_mass / opt_mass
    opt_mass = compute_mass(opt_rho, V, T).detach()
    opt_com = compute_com(opt_rho, V, T).detach().numpy()
    opt_rho = opt_rho.numpy()

    print(f"optimal: {opt_com} {opt_mass.numpy()}")

    print(f"error start: {numpy.sqrt(loss[0])} opt: {numpy.sqrt(loss[-1])}")
    print(f"min rho: {opt_rho.min()} max rho: {opt_rho.max()}")

    # print(numpy.linalg.norm(vec_cm(r0, V, T).detach().numpy()-target.numpy()))

    out_path = (pathlib.Path(__file__).parents[1] / "densities" /
                f"{args.ref_mesh.stem}-to-{args.mesh.stem}.txt")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving rhos to {out_path}")
    numpy.savetxt(out_path, opt_rho)


if __name__ == "__main__":
    main()
