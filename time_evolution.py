import numpy as np
from scipy.fft import fft, ifft, fftfreq
from Hamiltonian import Constants


class Split_Operator:
    def __init__(
        self, N, L, potential_func, mass=1.0, dt=1e-3, bc="periodic", **pot_kwargs
    ):
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

        # absorbing mask for absorbing boundaries
        self.mask = np.ones_like(self.x)

    def absorbing_boundary(self, start_frac=0.8, p=6):
        """smooth mask (0 < mask <= 1) for absorbing outgoing waves."""
        x0 = start_frac * self.L
        idx = self.x > x0
        scaled = (self.x[idx] - x0) / (self.L - x0)
        self.mask[idx] = np.exp(-(scaled**p))

    def step(self, psi):
        # split operator algo
        psi = self.V_half * psi
        psi = fft(psi)
        psi = self.T * psi
        psi = ifft(psi)
        psi = self.V_half * psi
        psi *= self.mask
        return psi


class Simple_Unitary_Time_Evolution:
    def __init__(
        self,
        N,
        L,
        potential_func,
        eigenvalues,
        eigenstates,
        mass=1.0,
        dt=1e-3,
        **pot_kwargs,
    ):
        self.N = N
        self.L = L
        self.dx = L / (N - 1)
        self.x = np.linspace(0, L, N)
        self.mass = mass
        self.dt = dt
        self.eigenvalues = eigenvalues
        self.eigenstates = eigenstates
        # potential on grid
        self.V = potential_func(self.x, L=L, **pot_kwargs)

    def psi_t(self, psi0, t=None):
        """
        Evolve psi0 forward in time.
        If t is None, uses self.dt. Otherwise uses t.
        """
        if t is None:
            t = self.dt

        # Expand psi0 in eigenbasis: c_n = <eigenstate_n | psi0>
        # For continuous normalization, inner product needs dx factor
        c = np.dot(self.eigenstates.T, psi0) * self.dx

        # Time evolve coefficients: c_n(t) = c_n(0) * exp(-i E_n t)
        c_t = c * np.exp(-1j * self.eigenvalues * t)

        # Reconstruct wavefunction: ψ(t) = Σ_n c_n(t) * eigenstate_n
        psi_t = np.dot(self.eigenstates, c_t)
        return psi_t
