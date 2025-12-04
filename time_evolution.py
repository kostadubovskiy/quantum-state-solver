import numpy as np
from scipy.fft import fft, ifft, fftfreq

#split operator requires periodic boundary conditions

class Split_Operator:
    def __init__(self, N, L, potential_func, mass=1.0, dt=1e-3):
        
        self.N = N
        self.L = L
        self.dx = L / N #needed for periodic bc
        self.x = np.linspace(0, L, N)
        self.mass = mass
        self.dt = dt

        # potential on grid
        self.V = potential_func(self.x, L=L)

        # FFT momentum grid
        self.k = fftfreq(N, d=self.dx) * 2 * np.pi

        # time evolution exponentials
        self.V_half = np.exp(-0.5j * self.V * dt)
        self.T = np.exp(-1j * (self.k**2) * dt / (2 * mass))


    def step(self, psi):
        #split operator algo
        psi = self.V_half * psi
        psi = fft(psi)
        psi = self.T * psi
        psi = ifft(psi)
        psi = self.V_half * psi
        return psi


