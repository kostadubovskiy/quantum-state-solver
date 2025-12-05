import numpy as np
from Hamiltonian import MassType

# --- Atomic Unit Constants (Needed by the functions) ---
m_e = 1.0
m_p = 1836.15267343


def V_box_1D_AU(x, L=None):
    return np.zeros_like(x)


def V_qho_1D_AU(x, L=None, k=1.0):
    return 0.5 * k * ((x - L / 2) ** 2)


def V_qho_1D_perturbed(x, L=None, k=1.0):
    return 0.5 * k * ((x - L / 2) ** 2) + (x - L / 2) ** 5


def E_box_1D_analytic_AU(n_states, L, mass: MassType = MassType.ELECTRON):
    n = np.arange(1, n_states + 1)
    energies = (n**2 * np.pi**2) / (2.0 * mass * L**2)
    return energies


def E_qho_1D_analytic_AU(n_states, L, mass: MassType = MassType.ELECTRON, k=1.0):
    # L is ignored for analytic formula, but passed by runner
    n = np.arange(0, n_states)
    energies = (n + 0.5) * np.sqrt(k / mass)
    return energies


# 2D TEST CASES


def V_box_2D_AU(x, y, L=None):
    return np.zeros_like(x + y)


def V_qho_2D_AU(x, y, L=None, k=1.0):
    return 0.5 * k * ((x - L / 2) ** 2 + (y - L / 2) ** 2)


def E_box_2D_analytic_AU(n_states, L, mass: MassType = MassType.ELECTRON):
    base_energy = (np.pi**2) / (2.0 * mass * L**2)
    energies = []
    max_n = int(np.ceil(np.sqrt(n_states))) + 2
    for nx in range(1, max_n + 1):
        for ny in range(1, max_n + 1):
            E = (nx**2 + ny**2) * base_energy
            energies.append(E)
    energies.sort()
    return np.array(energies[:n_states])


def E_qho_2D_analytic_AU(n_states, L, mass: MassType = MassType.ELECTRON, k=1.0):
    # L is ignored for analytic formula
    base_energy = np.sqrt(k / mass)
    energies = []
    N = 0  # N = nx + ny
    while len(energies) < n_states:
        E = (N + 1.0) * base_energy
        degeneracy = N + 1
        for _ in range(degeneracy):
            energies.append(E)
        N += 1
    return np.array(energies[:n_states])


# 3D TEST CASES


def V_box_3D_AU(x, y, z, L=None):
    return np.zeros_like(x + y + z)


def V_qho_3D_AU(x, y, z, L=None, k=1.0):
    return 0.5 * k * ((x - L / 2) ** 2 + (y - L / 2) ** 2 + (z - L / 2) ** 2)


def E_box_3D_analytic_AU(n_states, L, mass: MassType = MassType.ELECTRON):
    base_energy = (np.pi**2) / (2.0 * mass * L**2)
    energies = []
    max_n = int(np.ceil(n_states ** (1 / 3.0))) + 3
    for nx in range(1, max_n + 1):
        for ny in range(1, max_n + 1):
            for nz in range(1, max_n + 1):
                E = (nx**2 + ny**2 + nz**2) * base_energy
                energies.append(E)
    energies.sort()
    return np.array(energies[:n_states])


def E_qho_3D_analytic_AU(n_states, L, mass: MassType = MassType.ELECTRON, k=1.0):
    # L is ignored for analytic formula
    base_energy = np.sqrt(k / mass)
    energies = []
    N = 0  # N = nx + ny + nz
    while len(energies) < n_states:
        E = (N + 1.5) * base_energy
        degeneracy = (N + 1) * (N + 2) // 2
        for _ in range(degeneracy):
            energies.append(E)
        N += 1
    return np.array(energies[:n_states])
