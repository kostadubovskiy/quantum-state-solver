import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from Hamiltonian import Hamiltonian
from time_evolution import Split_Operator, Simple_Unitary_Time_Evolution


class Visualize:
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        time_evolver: Split_Operator | Simple_Unitary_Time_Evolution | None = None,
        states_to_plot: int = 3,
    ):
        # Store the hamiltonian and its results
        self.time_evolver = time_evolver
        self.hamiltonian = hamiltonian
        self.numeric_energies = hamiltonian.numeric_energies
        self.analytic_energies = hamiltonian.analytic_energies
        self.eigenvectors = hamiltonian.eigenvectors
        self.title = hamiltonian.test_case_name

        self.ndim = hamiltonian.ndim
        self.N = hamiltonian.N
        self.L = hamiltonian.L

        self.states_to_plot = min(states_to_plot, hamiltonian.num_states)

    def plot_energy_levels(self):
        """
        energy level diagram comparing numeric and analytic solutions.
        """

        plt.hlines(
            self.numeric_energies,
            0,
            1,
            colors="black",
            linestyles="solid",
            lw=2,
            label="Numeric",
        )

        if self.analytic_energies is not None:
            plt.hlines(
                self.analytic_energies,
                1,
                2,
                colors="red",
                linestyles="dashed",
                lw=2,
                label="Analytic",
            )
            plt.xticks([0.5, 1.5], ["Numeric", "Analytic"])
        else:
            plt.xticks([0.5], ["Numeric"])
        plt.legend()
        plt.title(self.title)
        plt.ylabel("Energy (Hartrees)")
        plt.xlabel("Solution")
        plt.grid(True, axis="y", linestyle=":")
        plt.show()

    def visualize_wavefunctions(self):
        """
        Plots the wavefunctions and probability densities.
        """
        for i in range(self.states_to_plot):
            psi_flat = self.eigenvectors[:, i]

            if self.ndim == 1:
                psi = psi_flat.reshape(self.N)
                prob_density = np.abs(psi) ** 2
                grid = self.hamiltonian.X
            elif self.ndim == 2:
                psi = psi_flat.reshape(self.N, self.N)
                prob_density = np.abs(psi) ** 2
                extent = [-self.L / 2, self.L / 2, -self.L / 2, self.L / 2]
            elif self.ndim == 3:
                psi_3d = psi_flat.reshape(self.N, self.N, self.N)
                slice_idx = self.N // 2
                psi = psi_3d[:, :, slice_idx]
                prob_density = np.abs(psi) ** 2
                extent = [-self.L / 2, self.L / 2, -self.L / 2, self.L / 2]

                z_coords = np.linspace(-self.L / 2, self.L / 2, self.N)
                z_slice_val = z_coords[slice_idx]
                print(f"Plotting 3D State {i} as 2D slice at z={z_slice_val:.2f}")

            # --- Create Plots ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(
                f"State {i} (E = {self.numeric_energies[i]:.4f} Hartrees)", fontsize=16
            )

            if self.ndim == 1:
                ax1.plot(grid, psi, label=r"Wavefunction $\psi$")
                ax1.set_title(r"Wavefunction ($\psi$)")
                ax1.set_xlabel("x (Bohr)")
                ax1.set_ylabel(r"$\psi(x)$")
                ax1.grid(True)

                ax2.plot(
                    grid,
                    prob_density,
                    label=r"Probability Density $|\psi^{2}|$",
                    color="r",
                )
                ax2.fill_between(grid, prob_density, color="red", alpha=0.3)
                ax2.set_title(r"Probability Density ($|\psi^{2}|$)")
                ax2.set_xlabel("x (Bohr)")
                ax2.set_ylabel(r"$|\psi(x)|^2$")
                ax2.grid(True)

            elif self.ndim == 2 or self.ndim == 3:
                im1 = ax1.imshow(psi.T, extent=extent, cmap="seismic", origin="lower")
                ax1.set_title(r"Wavefunction ($\psi$)")
                ax1.set_xlabel("x (Bohr)")
                ax1.set_ylabel("y (Bohr)")
                fig.colorbar(im1, ax=ax1, label=r"$\psi$")

                im2 = ax2.imshow(
                    prob_density.T, extent=extent, cmap="viridis", origin="lower"
                )
                ax2.set_title(r"Probability Density ($|\psi^{2}|$)")
                ax2.set_xlabel("x (Bohr)")
                ax2.set_ylabel("y (Bohr)")
                fig.colorbar(im2, ax=ax2, label=r"$|\psi^{2}|$")

            plt.tight_layout()
            plt.show()

    def plot_all(self):
        print(f"--- Generating plots for {self.title} ---")
        self.plot_energy_levels()
        self.visualize_wavefunctions()

    # animation wrapped!
    def animate_wavefunctions(
        self,
        psi0: np.ndarray,
        steps: int,
        snapshot_interval: int,
        save_path: str | None = None,
        fps: int = 20,
    ) -> FuncAnimation | None:
        if self.time_evolver is None:
            raise ValueError(
                "Time evolver needs to be set to animate wavefunctions!!! :<("
            )

        self._collect_snapshots(psi0, steps=steps, snapshot_interval=snapshot_interval)
        return self._create_animation_from_existing_snapshots(
            save_path=save_path, fps=fps
        )

    def _collect_snapshots(
        self, psi0: np.ndarray, steps: int, snapshot_interval: int
    ) -> bool:
        if (
            getattr(self, "snapshots", None) is not None
            and getattr(self, "times", None) is not None
            and getattr(self, "prob_densities", None) is not None
        ):
            print(
                "WARNING: Visualizer contains existing probability density snapshots. Clearing previous snapshots..."
            )

        self.snapshots = []
        self.times = []
        self.prob_densities = []

        psi_t = psi0.copy()  # time evolving wavefunction

        # Run evolution and collect snapshots
        for n in range(steps):
            if isinstance(self.time_evolver, Simple_Unitary_Time_Evolution):
                t = n * self.time_evolver.dt  # Accumulated time
                psi_t = self.time_evolver.psi_t(psi0, t=t)
            else:
                psi_t = self.time_evolver.step(psi_t)

            if n % snapshot_interval == 0:
                self.snapshots.append(psi_t.copy())
                self.times.append(n * self.time_evolver.dt)
                self.prob_densities.append(np.abs(psi_t) ** 2)

        self.times = np.array(self.times)
        self.prob_densities = np.array(self.prob_densities)

        return True

    def _create_animation_from_existing_snapshots(
        self,
        save_path: str | None = None,
        fps: int = 20,
        figsize: tuple[float, float] = (10, 6),
        title: str = "Time Evolution",
        v_minmax: tuple[float, float] = (-5, 60),
    ) -> FuncAnimation | None:
        if self.snapshots is None or self.times is None or self.prob_densities is None:
            raise ValueError("No snapshots collected. Please collect snapshots first.")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Plot potential on right y-axis
        ax_twin = ax.twinx()
        ax_twin.plot(
            self.time_evolver.x,
            self.time_evolver.V,
            "k--",
            alpha=0.3,
            linewidth=1,
            label="Potential",
        )

        # Initial probability density on left y-axis
        (line,) = ax.plot(
            self.time_evolver.x,
            self.prob_densities[0],
            "b-",
            linewidth=2,
            label=r"|$\psi^{2}$|",
        )

        # Set up axes
        ax.set_xlabel("x (Bohr)", fontsize=12)
        ax.set_ylabel(r"|$\psi^{2}(x)$|", fontsize=12, color="b")
        ax_twin.set_ylabel("Potential (Hartree)", fontsize=12, color="k")
        ax.set_xlim(self.time_evolver.x.min(), self.time_evolver.x.max())
        ax.set_ylim(0, self.prob_densities.max() * 1.1)

        # TODO: can't figure out how to intelligently do this automatically. Hardcoded to a nice range for our quartic for now.
        # maybe can use smth like this, trouble is getting center peak to show up nicely.
        # V_lower, V_upper = self.hamiltonian.V.min(), self.hamiltonian.V.max()
        # V_range = V_upper - V_lower
        # (V_lower - 0.1 * V_range, V_upper + 0.1 * V_range)
        ax_twin.set_ylim(v_minmax[0], v_minmax[1])

        title = ax.set_title(f"{title} - t = {self.times[0]:.6f} a.u.", fontsize=14)
        ax.legend(loc="upper left")
        ax_twin.legend(loc="upper right")  # Add legend for potential on right side
        ax.grid(True, alpha=0.3)

        def _animate(frame: int):
            line.set_ydata(self.prob_densities[frame])
            # Update fill_between by removing old and adding new
            if len(ax.collections) > 1:  # More than just the potential line
                ax.collections[-1].remove()  # Remove the last fill_between
            ax.fill_between(
                self.time_evolver.x, self.prob_densities[frame], alpha=0.3, color="blue"
            )
            title.set_text(f"Time Evolution - t = {self.times[frame]:.6f} a.u.")

            return [line, title]

        anim = FuncAnimation(
            fig,
            _animate,
            frames=len(self.prob_densities),
            interval=1000 // fps,  # ms
            blit=False,
            repeat=True,
        )

        if save_path is not None:
            anim.save(save_path, fps=fps, extra_args=["-vcodec", "libx264"])
            print(f"Animation saved to {save_path}")
        else:
            print("Animation not saved.")
        return anim
