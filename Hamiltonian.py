import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh
import time

# using atomic units for simplicity and stability
m_e = 1.0
m_p = 1836.15267343 

# Conversion factors
BOHR_TO_ANGSTROM = 0.529177
HARTREE_TO_EV = 27.211386

class Hamiltonian:
    def __init__(self, N, L, potential_func, ndim, num_states, bc='dirichlet'):
        """
        N: Number of grid points
        L: Physical length of space in ATOMIC UNITS (Bohr radii)
        potential_func: Function that returns potential in ATOMIC UNITS (Hartrees)
        ndim: 1, 2, or 3
        num_states: Number of eigenvalues to find
        mass: 'electron' or 'proton'
        """
        self.N = N
        self.L = L  # L, dx is in Bohr
        self.potential_func = potential_func
        self.ndim = ndim
        self.num_states = num_states
        self.m = m_e
        self.bc = bc
        self.analytic_energies = None
        
        

        # x, y, z are all in Bohr
        x = np.linspace(0, L , N)

        if ndim == 1:
            self.X = x
        elif ndim == 2:
            y = np.linspace(0, L , N)
            self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        elif ndim == 3:
            y = np.linspace(0, L , N)
            z = np.linspace(0, L, N)
            self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        self.dx = x[1] - x[0]
        self.V = self.potential_matrix()
        self.T = self.kinetic_matrix()

    def potential_matrix(self):
        # grid in Bohr, energy in Hartrees
        if self.ndim == 1:
            Vgrid = self.potential_func(self.X, L=self.L)
        elif self.ndim == 2:
            Vgrid = self.potential_func(self.X, self.Y, L=self.L)
        elif self.ndim == 3:
            Vgrid = self.potential_func(self.X, self.Y, self.Z, L=self.L)
        
        V_flat = Vgrid.reshape(self.N ** self.ndim)
        return diags(V_flat, 0, format='csr')


    def kinetic_matrix(self):
        N = self.N
        dx = self.dx
        coeff = -1.0 / (2.0 * self.m * dx**2)

        main = -2.0 * np.ones(N)
        off = 1.0 * np.ones(N-1)

        Lap = diags([off, main, off], [-1, 0, 1], shape=(N, N), format='lil')

        if self.bc == "periodic":
            Lap[0, -1] = 1.0
            Lap[-1, 0] = 1.0

        Lap = Lap.tocsr()

        I = eye(N, format='csr')

        if self.ndim == 1:
            K = coeff * Lap

        elif self.ndim == 2:
            K = coeff * (kron(Lap, I) + kron(I, Lap))

        elif self.ndim == 3:
            K = coeff * (
                kron(Lap, kron(I, I)) +
                kron(I, kron(Lap, I)) +
                kron(I, kron(I, Lap))
            )

        return K
        
    def solve(self):
        H = self.T + self.V
        # Eigenvalues will be in Hartrees
        eigenvalues, eigenvectors = eigsh(H, k=self.num_states, which='SA')
        self.numeric_energies = eigenvalues
        self.eigenvectors = eigenvectors
        return eigenvalues, eigenvectors
        
    def set_analytic_energies(self, energies):
        self.analytic_energies = np.array(energies)

    
