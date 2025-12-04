import numpy as np
from scipy.fft import fft, ifft, fftfreq


class Split_Operator:
    def __init__(self, N, L, potential_func, mass=1.0, dt=1e-3, bc="periodic", **pot_kwargs):
        
        self.N = N
        self.L = L
        self.dx = L / (N - 1)
        self.x = np.linspace(0, L, N)
        self.mass = mass
        self.dt = dt

        # potential on grid
        self.V = potential_func(self.x, L=L, **pot_kwargs)

        # FFT momentum grid
        self.k = fftfreq(N, d=self.dx) * 2 * np.pi

        # time evolution exponentials
        self.V_half = np.exp(-0.5j * self.V * dt)
        self.T = np.exp(-1j * (self.k**2) * dt / (2 * mass))

        #absorbing mask for absorbing boundaries
        self.mask = np.ones_like(self.x)

    def absorbing_boundary(self, start_frac=0.8, p=6):
        """smooth mask (0 < mask <= 1) for absorbing outgoing waves."""
        x0 = start_frac * self.L
        idx = self.x > x0
        scaled = (self.x[idx] - x0) / (self.L - x0)
        self.mask[idx] = np.exp(-(scaled**p))

    def step(self, psi):
        #split operator algo
        psi = self.V_half * psi
        psi = fft(psi)
        psi = self.T * psi
        psi = ifft(psi)
        psi = self.V_half * psi
        psi *= self.mask
        return psi


def time_evolve_state(initial_state, eigenstates, eigenvalues, t, hbar=1.0):
    """
    Time evolve a quantum state using eigenstate expansion.

    The time evolution is performed by:
    1. Expanding the initial state in the eigenbasis: c_n = <φ_n|ψ(0)>
    2. Applying time evolution: ψ(t) = Σ_n c_n φ_n exp(-i E_n t / ℏ)

    Parameters:
    -----------
    initial_state : array_like
        Initial wavefunction (1D array of length N^ndim)
    eigenstates : array_like
        Eigenstates as columns (shape: (N^ndim, num_states))
    eigenvalues : array_like
        Eigenvalues (energy levels) in atomic units (Hartrees)
    t : float
        Time to evolve to (in atomic units)
    hbar : float, optional
        Reduced Planck constant (default: 1.0 for atomic units)

    Returns:
    --------
    evolved_state : array_like
        Wavefunction at time t
    """
    # Expand initial state in eigenbasis: c_n = <φ_n|ψ(0)>
    coefficients = np.dot(eigenstates.T, initial_state)

    # Time evolve: ψ(t) = Σ_n c_n φ_n exp(-i E_n t / ℏ)
    time_phases = np.exp(-1j * eigenvalues * t / hbar)
    evolved_coefficients = coefficients * time_phases

    # Reconstruct wavefunction
    evolved_state = np.dot(eigenstates, evolved_coefficients)

    return evolved_state


def time_evolve_sequence(initial_state, eigenstates, eigenvalues, times, hbar=1.0):
    """
    Time evolve a quantum state over a sequence of times.

    Parameters:
    -----------
    initial_state : array_like
        Initial wavefunction (1D array of length N^ndim)
    eigenstates : array_like
        Eigenstates as columns (shape: (N^ndim, num_states))
    eigenvalues : array_like
        Eigenvalues (energy levels) in atomic units (Hartrees)
    times : array_like
        Array of time values (in atomic units)
    hbar : float, optional
        Reduced Planck constant (default: 1.0 for atomic units)

    Returns:
    --------
    evolved_states : array_like
        Wavefunctions at each time (shape: (len(times), N^ndim))
    """
    evolved_states = []
    for t in times:
        psi_t = time_evolve_state(initial_state, eigenstates, eigenvalues, t, hbar=hbar)
        evolved_states.append(psi_t)

    return np.array(evolved_states)


def calculate_oscillation_period(eigenvalues, state1_idx=0, state2_idx=1, hbar=1.0):
    """
    Calculate the oscillation period between two eigenstates.

    The period is given by: T = 2πℏ / |E₂ - E₁|

    Parameters:
    -----------
    eigenvalues : array_like
        Eigenvalues (energy levels) in atomic units (Hartrees)
    state1_idx : int
        Index of first eigenstate
    state2_idx : int
        Index of second eigenstate
    hbar : float, optional
        Reduced Planck constant (default: 1.0 for atomic units)

    Returns:
    --------
    period : float
        Oscillation period in atomic units
    """
    energy_diff = abs(eigenvalues[state2_idx] - eigenvalues[state1_idx])
    if energy_diff == 0:
        raise ValueError("Energy difference is zero - states are degenerate")
    period = 2 * np.pi * hbar / energy_diff
    return period


def create_superposition(eigenstates, coefficients=None, states_to_mix=None):
    """
    Create a superposition of eigenstates.

    Parameters:
    -----------
    eigenstates : array_like
        Eigenstates as columns (shape: (N^ndim, num_states))
    coefficients : array_like, optional
        Coefficients for each eigenstate (default: equal superposition of first 2)
    states_to_mix : list, optional
        Indices of states to mix (default: [0, 1])

    Returns:
    --------
    initial_state : array_like
        Normalized superposition wavefunction
    """
    num_states = eigenstates.shape[1]

    if coefficients is None:
        if states_to_mix is None:
            states_to_mix = [0, 1]

        coefficients = np.zeros(num_states, dtype=complex)
        n_mix = len(states_to_mix)
        for idx in states_to_mix:
            coefficients[idx] = 1.0 / np.sqrt(n_mix)
    else:
        coefficients = np.array(coefficients, dtype=complex)

    # Normalize coefficients
    coefficients = coefficients / np.linalg.norm(coefficients)

    # Create superposition
    initial_state = np.dot(eigenstates, coefficients)

    return initial_state
