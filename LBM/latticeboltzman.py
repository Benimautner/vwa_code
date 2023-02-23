import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""
source: https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c
changes: in accordance with [https://youtu.be/JFWqCQHg-Hs]. furthermore, starting parameters have been changed
licence: GNU GENERAL PUBLIC LICENSE Version 3

original metadata:
Create Your Own Lattice Boltzmann Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate flow past boundary
for an isothermal fluid

"""


def distance(a, b):
    return np.linalg.norm(a - b)


def distance2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def main(rho0_input=0):
    """ Lattice Boltzmann Simulation """

    # Simulation parameters
    Nx = 400  # resolution x-dir
    Ny = 100  # resolution y-dir
    rho0 = 100  # average density
    tau = 0.6  # collision timescale
    Nt = 4000  # number of timesteps
    plotRealTime = False  # switch on for plotting as the simulation goes along

    if rho0_input != 0:
        rho0 = rho0_input

    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])  # sums to 1

    # Initial Conditions
    F = np.ones((Ny, Nx, NL))  # * rho0 / NL
    np.random.seed(42)
    F += 0.01 * np.random.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    # boundary boundary
    boundary = np.full((Ny, Nx), False)

    for y in range(0, Ny):
        for x in range(0, Nx):
            if distance2((Nx // 4, Ny // 2), (x, y)) < 13 ** 2:
                boundary[y][x] = True

    # Prep figure
    fig = plt.figure(figsize=(4, 2), dpi=80)

    # Simulation Main Loop
    for it in tqdm(range(Nt)):

        # Apply boundary condition to remove reflectiveness from right wall
        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        # Drift
        # go through every single node, roll it in direction of corresponding velocity 
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # Calculate fluid variables
        rho = np.sum(F, 2)  # density is sum of velocities
        ux = np.sum(F * cxs, 2) / rho  #
        uy = np.sum(F * cys, 2) / rho

        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = rho * w * (
                    1 + 3 * (cx * ux + cy * uy)
                    + 9 * (cx * ux + cy * uy) ** 2 / 2
                    - 3 * (ux ** 2 + uy ** 2) / 2
            )

        F += -(1.0 / tau) * (F - Feq)

        # Apply boundary 
        # Set reflective boundaries
        bndryF = F[boundary, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        F[boundary, :] = bndryF
        ux[boundary] = 0
        uy[boundary] = 0

        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % 10) == 0) or (it == Nt - 1):
            plt.cla()
            ux[boundary] = 0
            uy[boundary] = 0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
                        np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[boundary] = np.nan
            vorticity = np.ma.array(vorticity, mask=boundary)
            # plt.imshow(vorticity, cmap='bwr')
            # plt.imshow(~boundary, cmap='gray', alpha=0.3)
            plt.imshow(np.sqrt(ux ** 2, uy ** 2))
            plt.clim(-.1, .1)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.pause(0.001)

    # Save figure
    plt.savefig('latticeboltzmann.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()