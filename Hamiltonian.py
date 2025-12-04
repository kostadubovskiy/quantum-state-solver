import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh
from enum import Enum


# using atomic units for simplicity and stability
class Constants(Enum):
    ELECTRON_MASS = 1.0
    PROTON_MASS = 1836.15267343


class MassType(Enum):
    ELECTRON = "electron"
    PROTON = "proton"


def get_mass(mass: MassType) -> float | None:
    if mass == MassType.ELECTRON:
        return Constants.ELECTRON_MASS.value
    elif mass == MassType.PROTON:
        return Constants.PROTON_MASS.value
    else:
        return None


class Hamiltonian:
    def __init__(
        self,
        N,
        L,
        potential_func,
        ndim,
        num_states,
        mass: MassType = MassType.ELECTRON,
    ):
        """
        inputs:
        N: Number of grid points, power of 2 is good for fft
        L: Physical length (Bohr radii) should be around 20
        ndim: 1, 2, or 3
        num_states: Number of eigenvalues to find
        mass: MassType.ELECTRON or MassType.PROTON

        sets:
        various properties of our system for the Hamiltonian
        """
        self.N = N
        self.N_interior = N-1
        self.L = L
        self.potential_func = potential_func
        self.ndim = ndim
        self.num_states = num_states
        self.mass = get_mass(mass)
        self.analytic_energies = None

        # x, y, z are all in Bohr
        x = np.linspace(0, L, N, endpoint=False) #periodic grid like for split op
        x_interior = x[1:] #only interior pts for dirichlet solver cuz we force psi at 0,L to be 0
        self.dx = L/N
        
        if ndim == 1:
            self.X = x_interior
        elif ndim == 2:
            y = x_interior
            self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        elif ndim == 3:
            y = x_interior
            z = x_interior
            self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")

        self.V = self.potential_matrix()
        self.T = self.kinetic_matrix()

    def potential_matrix(self):
        if self.ndim == 1:
            Vgrid = self.potential_func(self.X, L=self.L)
        elif self.ndim == 2:
            Vgrid = self.potential_func(self.X, self.Y, L=self.L)
        elif self.ndim == 3:
            Vgrid = self.potential_func(self.X, self.Y, self.Z, L=self.L)

        V_flat = Vgrid.reshape(self.N_interior**self.ndim)
        return diags(V_flat, 0, format="csr")

    def kinetic_matrix(self):
        #building finite difference kinetic energy operator T = -ihbar/2m * p**2
        dx = self.dx
        
        coeff = -1.0 / (2.0 * self.mass * dx**2)
        main = -2.0 * np.ones(self.N_interior)
        off = 1.0 * np.ones(self.N_interior - 1)

        Lap = diags(
            [off, main, off], [-1, 0, 1], shape=(self.N_interior, self.N_interior), format="csr"
        )  # csr for better access to rows

        I = eye(self.N_interior, format="csr")

        if self.ndim == 1:
            K = coeff * Lap
        #use kron to upgrade laplacian to higher dimension
        elif self.ndim == 2:
            K = coeff * (kron(Lap, I) + kron(I, Lap))

        elif self.ndim == 3:
            K = coeff * (
                kron(Lap, kron(I, I)) + kron(I, kron(Lap, I)) + kron(I, kron(I, Lap))
            )

        return K

    def solve(self):
        H = self.T + self.V
        # Eigenvalues will be in Hartrees, SA smallest algebraic eigenvalue, sigma tells it where to start looking for eigenvalues
        eigenvalues, eigenvectors = eigsh(H, k=self.num_states, which="SA", sigma=0)

        # We want continuous normalization
        norm_factor = 1.0 / (self.dx ** (self.ndim / 2.0))

        self.numeric_energies = eigenvalues
        self.eigenvectors = eigenvectors* norm_factor
        return eigenvalues, eigenvectors

    def set_analytic_energies(self, energies):
        #for when we validate
        self.analytic_energies = np.array(energies)
