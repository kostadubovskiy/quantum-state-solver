import numpy as np

# --- Atomic Unit Constants (Needed by the functions) ---
m_e = 1.0
m_p = 1836.15267343


# 1D 

def V_box_1D(x, L=None):
    return np.zeros_like(x)

def V_qho_1D(x, L=None, k=1.0):
    return 0.5 * k * x**2

def E_box_1D_analytic(n_states, L, mass=m_e):
    n = np.arange(1, n_states + 1)
    energies = (n**2 * np.pi**2) / (2.0 * mass * L**2)
    return energies

def E_qho_1D_analytic(n_states, L, mass=m_e, k=1.0):
    n = np.arange(0, n_states)
    energies = (n + 0.5) * np.sqrt(k / mass)
    return energies

# 2D 

def V_box_2D(x, y, L=None):
    return np.zeros_like(x + y)

def V_qho_2D(x, y, L=None, k=1.0):
    return 0.5 * k * (x**2 + y**2)

def E_box_2D_analytic(n_states, L, mass=m_e):
    base_energy = (np.pi**2) / (2.0 * mass * L**2)
    energies = []
    max_n = int(np.ceil(np.sqrt(n_states))) + 2
    for nx in range(1, max_n + 1):
        for ny in range(1, max_n + 1):
            E = (nx**2 + ny**2) * base_energy
            energies.append(E)
    energies.sort()
    return np.array(energies[:n_states])

def E_qho_2D_analytic(n_states, L, mass=m_e, k=1.0):
    base_energy = np.sqrt(k / mass)
    energies = []
    N = 0 # N = nx + ny
    while len(energies) < n_states:
        E = (N + 1.0) * base_energy
        degeneracy = N + 1
        for _ in range(degeneracy):
            energies.append(E)
        N += 1
    return np.array(energies[:n_states])

# 3D

def V_box_3D(x, y, z, L=None):
    return np.zeros_like(x + y + z)

def V_qho_3D(x, y, z, L=None, k=1.0):
    return 0.5 * k * (x**2 + y**2 + z**2)

def E_box_3D_analytic(n_states, L, mass=m_e):
    base_energy = (np.pi**2) / (2.0 * mass * L**2)
    energies = []
    max_n = int(np.ceil(n_states**(1/3.0))) + 3
    for nx in range(1, max_n + 1):
        for ny in range(1, max_n + 1):
            for nz in range(1, max_n + 1):
                E = (nx**2 + ny**2 + nz**2) * base_energy
                energies.append(E)
    energies.sort()
    return np.array(energies[:n_states])
    
def E_qho_3D_analytic(n_states, L, mass=m_e, k=1.0):
    base_energy = np.sqrt(k / mass)
    energies = []
    N = 0 # N = nx + ny + nz
    while len(energies) < n_states:
        E = (N + 1.5) * base_energy
        degeneracy = (N + 1) * (N + 2) // 2
        for _ in range(degeneracy):
            energies.append(E)
        N += 1
    return np.array(energies[:n_states])