import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Visualize:
    def __init__(self, solver, states_to_plot=3):
        # Store the solver and its results
        self.solver = solver
        self.numeric_energies = solver.numeric_energies
        self.analytic_energies = solver.analytic_energies
        self.eigenvectors = solver.eigenvectors
        self.title = solver.test_case_name

        self.ndim = solver.ndim
        self.N = solver.N
        self.L = solver.L

        self.states_to_plot = min(states_to_plot, solver.num_states)

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
                grid = self.solver.X
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

    def visualize_3d_state(
        self,
        state_idx,
        method="isosurface",
        iso_levels=5,
        slice_axis="z",
        slice_positions=None,
    ):
        """
        Visualize a 3D eigenstate using various methods.

        Parameters:
        -----------
        state_idx : int
            Index of the state to visualize
        method : str
            'isosurface', 'slices', or 'volume'
        iso_levels : int
            Number of isosurface levels (for isosurface method)
        slice_axis : str
            'x', 'y', or 'z' (for slices method)
        slice_positions : list
            Positions along slice_axis to take slices (for slices method)
        """
        if self.ndim != 3:
            print("This method is for 3D systems only.")
            return

        psi_flat = self.eigenvectors[:, state_idx]
        psi_3d = psi_flat.reshape(self.N, self.N, self.N)
        prob_density = np.abs(psi_3d) ** 2

        if method == "isosurface":
            self._plot_isosurface(prob_density, state_idx, iso_levels)
        elif method == "slices":
            if slice_positions is None:
                slice_positions = [-self.L / 4, 0.0, self.L / 4]
            self._plot_multiple_slices(
                prob_density, state_idx, slice_axis, slice_positions
            )
        elif method == "volume":
            self._plot_volume_rendering(prob_density, state_idx)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _plot_isosurface(self, prob_density, state_idx, iso_levels):
        """Plot 3D isosurfaces of probability density."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Create coordinate grids
        x = np.linspace(0, self.L, self.N)
        y = np.linspace(0, self.L, self.N)
        z = np.linspace(0, self.L, self.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Find isosurface levels
        prob_max = prob_density.max()
        iso_values = np.linspace(prob_max * 0.1, prob_max * 0.9, iso_levels)

        # Plot isosurfaces using contour plots
        for i, iso_val in enumerate(iso_values):
            # Use a slice through the center for visualization
            z_slice = self.N // 2
            ax.contour(
                X[:, :, z_slice],
                Y[:, :, z_slice],
                prob_density[:, :, z_slice],
                levels=[iso_val],
                alpha=0.3,
            )

        ax.set_xlabel("x (Bohr)")
        ax.set_ylabel("y (Bohr)")
        ax.set_zlabel("z (Bohr)")
        ax.set_title(
            f"3D Isosurfaces - State {state_idx} (E = {self.numeric_energies[state_idx]:.4f} Hartrees)"
        )
        plt.show()

    def _plot_multiple_slices(
        self, prob_density, state_idx, slice_axis, slice_positions
    ):
        """Plot multiple 2D slices through the 3D state."""
        n_slices = len(slice_positions)
        cols = min(3, n_slices)
        rows = (n_slices + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if n_slices == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        extent = [0, self.L, 0, self.L]

        for idx, pos in enumerate(slice_positions):
            ax = axes[idx]

            # Convert position to index
            if slice_axis == "x":
                slice_idx = int((pos / self.L) * self.N)
                slice_idx = np.clip(slice_idx, 0, self.N - 1)
                slice_data = prob_density[slice_idx, :, :]
                title = f"x = {pos:.2f} Bohr"
            elif slice_axis == "y":
                slice_idx = int((pos / self.L) * self.N)
                slice_idx = np.clip(slice_idx, 0, self.N - 1)
                slice_data = prob_density[:, slice_idx, :]
                title = f"y = {pos:.2f} Bohr"
            else:  # z
                slice_idx = int((pos / self.L) * self.N)
                slice_idx = np.clip(slice_idx, 0, self.N - 1)
                slice_data = prob_density[:, :, slice_idx]
                title = f"z = {pos:.2f} Bohr"

            im = ax.imshow(slice_data.T, extent=extent, cmap="viridis", origin="lower")
            ax.set_title(title)
            ax.set_xlabel("x (Bohr)" if slice_axis != "x" else "y (Bohr)")
            ax.set_ylabel("y (Bohr)" if slice_axis == "z" else "z (Bohr)")
            plt.colorbar(im, ax=ax, label=r"$|\psi^{2}|$")

        # Hide unused subplots
        for idx in range(n_slices, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            f"State {state_idx} - Multiple Slices along {slice_axis}-axis", fontsize=14
        )
        plt.tight_layout()
        plt.show()

    def _plot_volume_rendering(self, prob_density, state_idx):
        """Volume rendering using scatter plot with opacity."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Sample points for visualization (to avoid too many points)
        step = max(1, self.N // 20)
        x = np.linspace(0, self.L, self.N)[::step]
        y = np.linspace(0, self.L, self.N)[::step]
        z = np.linspace(0, self.L, self.N)[::step]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Sample probability density
        prob_sample = prob_density[::step, ::step, ::step]
        prob_flat = prob_sample.flatten()

        # Normalize for opacity
        prob_norm = prob_flat / prob_flat.max()

        # Plot with opacity
        scatter = ax.scatter(
            X.flatten(),
            Y.flatten(),
            Z.flatten(),
            c=prob_flat,
            s=20 * prob_norm,
            alpha=0.3 * prob_norm,
            cmap="viridis",
        )

        ax.set_xlabel("x (Bohr)")
        ax.set_ylabel("y (Bohr)")
        ax.set_zlabel("z (Bohr)")
        ax.set_title(
            f"Volume Rendering - State {state_idx} (E = {self.numeric_energies[state_idx]:.4f} Hartrees)"
        )
        plt.colorbar(scatter, ax=ax, label=r"$|\psi^{2}|$")
        plt.show()

    def animate_3d_slices(self, state_idx, slice_axis="z", fps=30, save_path=None):
        """
        Animate slices through a 3D state.

        Parameters:
        -----------
        state_idx : int
            Index of the state to animate
        slice_axis : str
            'x', 'y', or 'z'
        fps : int
            Frames per second
        save_path : str, optional
            Path to save animation (e.g., 'animation.mp4')

        Returns:
        --------
        anim : FuncAnimation
            Animation object
        """
        if self.ndim != 3:
            print("This method is for 3D systems only.")
            return None

        psi_flat = self.eigenvectors[:, state_idx]
        psi_3d = psi_flat.reshape(self.N, self.N, self.N)
        prob_density = np.abs(psi_3d) ** 2

        fig, ax = plt.subplots(figsize=(8, 8))

        x_coords = np.linspace(0, self.L, self.N)
        y_coords = np.linspace(0, self.L, self.N)
        extent = [0, self.L, 0, self.L]

        # Initial frame
        slice_idx = 0
        if slice_axis == "x":
            slice_data = prob_density[slice_idx, :, :]
            xlabel, ylabel = "y (Bohr)", "z (Bohr)"
        elif slice_axis == "y":
            slice_data = prob_density[:, slice_idx, :]
            xlabel, ylabel = "x (Bohr)", "z (Bohr)"
        else:  # z
            slice_data = prob_density[:, :, slice_idx]
            xlabel, ylabel = "x (Bohr)", "y (Bohr)"

        im = ax.imshow(
            slice_data.T,
            extent=extent,
            cmap="viridis",
            origin="lower",
            animated=True,
            vmin=0,
            vmax=prob_density.max(),
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, label=r"$|\psi^{2}|$")

        title = ax.set_title(
            f"State {state_idx} - Slice {slice_axis}[0] = {0:.2f} Bohr"
        )

        def animate(frame):
            slice_idx = frame
            if slice_axis == "x":
                slice_data = prob_density[slice_idx, :, :]
                pos = x_coords[slice_idx]
            elif slice_axis == "y":
                slice_data = prob_density[:, slice_idx, :]
                pos = y_coords[slice_idx]
            else:  # z
                slice_data = prob_density[:, :, slice_idx]
                pos = y_coords[slice_idx]  # Using y_coords for z position

            im.set_array(slice_data.T)
            title.set_text(
                f"State {state_idx} - Slice {slice_axis}[{slice_idx}] = {pos:.2f} Bohr"
            )
            return [im, title]

        anim = FuncAnimation(
            fig, animate, frames=self.N, interval=1000 / fps, blit=True, repeat=True
        )

        if save_path:
            anim.save(save_path, fps=fps, extra_args=["-vcodec", "libx264"])

        plt.show()
        return anim

    def animate_time_evolution_3d(
        self, psi_trajectory, times, slice_axis="z", fps=60, save_path=None
    ):
        """
        Animate time evolution of a 3D wavefunction.

        Parameters:
        -----------
        psi_trajectory : array_like
            Wavefunction trajectory, shape (n_steps, N, N, N)
        times : array_like
            Time values (in atomic units)
        slice_axis : str
            'x', 'y', or 'z'
        fps : int
            Frames per second
        save_path : str, optional
            Path to save animation (e.g., 'time_evolution.mp4')

        Returns:
        --------
        anim : FuncAnimation
            Animation object
        """
        if self.ndim != 3:
            print("This method is for 3D systems only.")
            return None

        n_steps = len(times)
        prob_trajectory = np.abs(psi_trajectory) ** 2

        # Find global max for consistent color scale
        prob_max = prob_trajectory.max()

        fig, ax = plt.subplots(figsize=(8, 8))

        extent = [0, self.L, 0, self.L]

        # Initial frame
        slice_idx = self.N // 2
        if slice_axis == "x":
            slice_data = prob_trajectory[0, slice_idx, :, :]
            xlabel, ylabel = "y (Bohr)", "z (Bohr)"
        elif slice_axis == "y":
            slice_data = prob_trajectory[0, :, slice_idx, :]
            xlabel, ylabel = "x (Bohr)", "z (Bohr)"
        else:  # z
            slice_data = prob_trajectory[0, :, :, slice_idx]
            xlabel, ylabel = "x (Bohr)", "y (Bohr)"

        im = ax.imshow(
            slice_data.T,
            extent=extent,
            cmap="viridis",
            origin="lower",
            animated=True,
            vmin=0,
            vmax=prob_max,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, label=r"$|\psi^{2}|$")

        title = ax.set_title(f"Time Evolution - t = {times[0]:.4f} a.u.")

        def animate(frame):
            if slice_axis == "x":
                slice_data = prob_trajectory[frame, slice_idx, :, :]
            elif slice_axis == "y":
                slice_data = prob_trajectory[frame, :, slice_idx, :]
            else:  # z
                slice_data = prob_trajectory[frame, :, :, slice_idx]

            im.set_array(slice_data.T)
            title.set_text(f"Time Evolution - t = {times[frame]:.4f} a.u.")
            return [im, title]

        anim = FuncAnimation(
            fig, animate, frames=n_steps, interval=1000 / fps, blit=True, repeat=True
        )

        if save_path:
            anim.save(save_path, fps=fps, extra_args=["-vcodec", "libx264"])

        plt.show()
        return anim

    def visualize_potential_with_wavefunction_interactive(
        self,
        state_idx,
        z_slice_idx=None,
        show_potential_surface=True,
        show_prob_isosurface=True,
        potential_opacity=0.6,
        prob_opacity=0.4,
    ):
        """
        Interactive 3D visualization overlaying probability density on potential surface.

        Parameters:
        -----------
        state_idx : int
            Index of the state to visualize
        z_slice_idx : int, optional
            Which z-slice to use (default: middle)
        show_potential_surface : bool
            Whether to show the potential as a 3D surface
        show_prob_isosurface : bool
            Whether to show probability density as isosurface
        potential_opacity : float
            Opacity of potential surface
        prob_opacity : float
            Opacity of probability isosurface
        """
        import plotly.graph_objects as go

        if self.ndim != 3:
            print("This method is for 3D systems only.")
            return

        if z_slice_idx is None:
            z_slice_idx = self.N // 2

        # Get wavefunction and probability density
        psi_flat = self.eigenvectors[:, state_idx]
        psi_3d = psi_flat.reshape(self.N, self.N, self.N)
        prob_density = np.abs(psi_3d) ** 2

        # Get coordinates
        x = np.linspace(0, self.L, self.N)
        y = np.linspace(0, self.L, self.N)
        z_coords = np.linspace(0, self.L, self.N)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Get probability density at z-slice
        prob_slice = prob_density[:, :, z_slice_idx]

        # Calculate potential on the grid
        # Need to reconstruct the potential function
        # For the 4-well potential: V = V_x(x) + V_y(y)
        # Using the same parameters as in the notebook
        a = self.L
        b = 1.0
        V_max = 10.0

        V_x = (V_max / b**4) * (((X - (a / 2)) ** 2) - b**2) ** 2
        V_y = (V_max / b**4) * (((Y - (a / 2)) ** 2) - b**2) ** 2
        V_surface = V_x + V_y

        # Create figure
        fig = go.Figure()

        # Add potential surface
        if show_potential_surface:
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=V_surface,
                    colorscale="RdBu",
                    colorbar=dict(title="Potential (Hartree)", x=1.02),
                    opacity=potential_opacity,
                    name="Potential",
                    showscale=True,
                )
            )

        # Add probability density as colormap on potential surface
        if show_potential_surface:
            # Color the potential surface by probability density
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=V_surface,
                    surfacecolor=prob_slice,
                    colorscale="Viridis",
                    colorbar=dict(title="|ψ|²", x=1.15),
                    opacity=potential_opacity + 0.2,
                    name="Probability Density",
                    showscale=True,
                )
            )

        # Add probability density as isosurface above potential
        if show_prob_isosurface:
            # Create 3D isosurface of probability density
            prob_max = prob_density.max()
            Z_full = np.full_like(X, z_coords[z_slice_idx])

            # Add isosurface at the z-slice level
            fig.add_trace(
                go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z_full.flatten(),
                    value=prob_slice.flatten(),
                    isomin=prob_max * 0.3,
                    isomax=prob_max,
                    opacity=prob_opacity,
                    surface_count=3,
                    colorscale="Viridis",
                    name="Probability Isosurface",
                )
            )

        # Update layout
        fig.update_layout(
            title=f"State {state_idx} - Potential & Probability Density (z = {z_coords[z_slice_idx]:.2f} Bohr)",
            scene=dict(
                xaxis_title="x (Bohr)",
                yaxis_title="y (Bohr)",
                zaxis_title="Potential (Hartree) / z (Bohr)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            width=900,
            height=700,
        )

        fig.show()
