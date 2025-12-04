import numpy as np
from scipy.fft import fft, ifft, fftfreq


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
