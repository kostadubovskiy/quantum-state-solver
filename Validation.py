from Hamiltonian import Hamiltonian, MassType
import Potentials as p
from Visualization import Visualize
import numpy as np


class Validate(Hamiltonian):
    def __init__(self, test_case, N, L, num_states, mass='electron'):
        """ 
        Args:
            test_case options: "box_1d", "qho_1d","box_2d", "qho_2d","box_3d", "qho_3d" """
        
        self.test_case = test_case
        self.k = 1
        
        if test_case == 'box_1d':
            ndim = 1
            potential_func = V_box_1D
            self.analytic_func = E_box_1D_analytic
            self.test_case_name = "1D Particle in a Box"

        elif test_case == "qho_1d":
            ndim = 1
            potential_func = lambda x, L: p.V_qho_1D_AU(x, L, k=self.k)
            self.analytic_func = lambda n, L, m: p.E_qho_1D_analytic_AU(
                n, L, m, k=self.k
            )
            self.test_case_name = "1D Quantum Harmonic Oscillator"

        elif test_case == "box_2d":
            ndim = 2
            potential_func = p.V_box_2D_AU
            self.analytic_func = p.E_box_2D_analytic_AU
            self.test_case_name = "2D Particle in a Box"

        elif test_case == "qho_2d":
            ndim = 2
            potential_func = lambda x, y, L: p.V_qho_2D_AU(x, y, L, k=self.k)
            self.analytic_func = lambda n, L, m: p.E_qho_2D_analytic_AU(
                n, L, m, k=self.k
            )
            self.test_case_name = "2D Quantum Harmonic Oscillator"

        elif test_case == "box_3d":
            ndim = 3
            potential_func = p.V_box_3D_AU
            self.analytic_func = p.E_box_3D_analytic_AU
            self.test_case_name = "3D Particle in a Box"

        elif test_case == "qho_3d":
            ndim = 3
            potential_func = lambda x, y, z, L: p.V_qho_3D_AU(x, y, z, L, k=self.k)
            self.analytic_func = lambda n, L, m: p.E_qho_3D_analytic_AU(
                n, L, m, k=self.k
            )
            self.test_case_name = "3D Quantum Harmonic Oscillator"

        else:
            raise ValueError(f"Unknown test_case: '{test_case}'.")

        # 2. Call the parent's __init__ with the correct, determined args
        super().__init__(
            N=N,
            L=L,
            potential_func=potential_func,
            ndim=ndim,
            num_states=num_states,
            mass=mass,
        )

    def valid_solve(self):
        """
        Runs the solver and immediately calculates the analytic solution
        for comparison.
        """
        print("\n" + "=" * 70)
        print(f"RUNNING VALIDATION: {self.test_case_name}")
        print(f"N={self.N} (per dim), L={self.L} Bohr, num_states={self.num_states}")
        print("=" * 70)

        # 1. Solve numerically (calls parent's .solve())
        self.numeric_energies, self.eigenvectors = self.solve()

        # 2. Get analytic solution
        self.analytic_energies = self.analytic_func(self.num_states, self.L, self.mass)

        print("\n--- Validation Results ---")
        print(
            f"{'State (i)':<10} | {'Numeric E':<15} | {'Analytic E':<15} | {'% Error':<10}"
        )
        print("-" * 54)
        for i in range(self.num_states):
            num_E = self.numeric_energies[i]
            an_E = self.analytic_energies[i]
            error = 100 * np.abs(num_E - an_E) / an_E
            print(f"{i:<10} | {num_E:<15.6f} | {an_E:<15.6f} | {error:<10.4f}%")

        return self.numeric_energies, self.analytic_energies

    def plot_test(self, states_to_plot=3):
        if not hasattr(self, "numeric_energies"):
            print("Error: Must run .valid_solve() before plotting.")
            return
        vis = Visualize(self, states_to_plot=states_to_plot)
        vis.plot_all()
