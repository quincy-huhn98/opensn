import numpy as np
import matplotlib.pyplot as plt
from utils import *
import scipy.interpolate
from matplotlib.colors import LogNorm


def plot_2d_flux(file_pattern, ranks, moment=0, prefix="fom", grid_res=200, pid=0):
    """Create smooth full-color plots (not scatter) for each energy group."""
    xs, ys, vals, G = load_2d_flux(file_pattern, ranks, moment=moment)

    for g in range(G):
        # Create regular grid
        xi = np.linspace(xs[g].min(), xs[g].max(), grid_res)
        yi = np.linspace(ys[g].min(), ys[g].max(), grid_res)
        X, Y = np.meshgrid(xi, yi)

        # Interpolate data onto grid
        Z = scipy.interpolate.griddata((xs[g], ys[g]), vals[g], (X, Y), method="linear")

        vmin = max(np.nanmin(Z), 1e-10)
        vmax = np.nanmax(Z)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        plt.figure(figsize=(6, 5))
        im = plt.imshow(
            Z,
            extent=[xi.min(), xi.max(), yi.min(), yi.max()],
            origin="lower",
            aspect="equal",
            cmap="viridis",
            norm=norm
        )
        plt.title(f"{prefix.upper()} Group {g} (moment {moment})")
        plt.xlabel("x")
        plt.ylabel("y")
        cbar = plt.colorbar(im)
        cbar.set_label("Flux")
        outpath = f"results/{prefix}_group_{g}_{pid}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

def plot_2d_lineout(ranks, y_target=4.0, moment=0, grid_res=200, pid=0):
    """Plot lineout at y_target of ROM and FOM."""
    xs, ys, vals, G = load_2d_flux("output/rom{}.h5", ranks, moment=moment)
    xs_, ys_, vals_, G = load_2d_flux("output/fom{}.h5", ranks, moment=moment)

    for g in range(G):
        # Create regular grid
        xi = np.linspace(xs[g].min(), xs[g].max(), grid_res)
        yi = np.linspace(ys[g].min(), ys[g].max(), grid_res)
        X, Y = np.meshgrid(xi, yi)

        # Interpolate data onto grid
        Z = scipy.interpolate.griddata((xs[g], ys[g]), vals[g], (X, Y), method="linear")
        Z_ = scipy.interpolate.griddata((xs[g], ys[g]), vals_[g], (X, Y), method="linear")

        # Find the row index closest to y=4
        row_idx = np.argmin(np.abs(yi - y_target))

        # Extract data along y = 4
        rom_line = Z[row_idx, :]
        fom_line = Z_[row_idx, :]

        # Plot ROM vs FOM
        plt.figure(figsize=(8,5))
        plt.plot(xi, rom_line, label='ROM', color='blue')
        plt.plot(xi, fom_line, label='FOM', color='orange', linestyle='--')
        plt.xlabel('X')
        plt.ylabel('Scalar Value')
        plt.title(f'ROM vs FOM at y={y_target}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/line_y{y_target}_rom_fom_{pid}_{g}.jpg")
        plt.close()

    error = np.linalg.norm(np.asarray(vals_)-np.asarray(vals))/np.linalg.norm(np.asarray(vals_))
    return error


def plot_sv(num_groups):
    for i in range(num_groups):
        S = np.loadtxt("data/singular_values_g{}.txt".format(i))
        plt.semilogy(S, 'o-')
        plt.xlabel("Mode index")
        plt.ylabel("Singular value")
        plt.title("Singular value decay")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/svd_decay_{}.jpg".format(i))
        plt.close()


def plot_1d_flux(fom_pattern, rom_pattern, ranks, moment=0, prefix="reed_ommi", pid=0):
    """Compare FOM vs ROM 1-D flux."""
    fom_x, fom_vals, G = load_1d_flux(fom_pattern, ranks, moment=moment)
    rom_x, rom_vals, G = load_1d_flux(rom_pattern, ranks, moment=moment)

    errors = []
    for g in range(G):
        plt.figure(figsize=(6, 4))
        plt.plot(fom_x[g], fom_vals[g], "-", label="FOM")
        plt.plot(rom_x[g], rom_vals[g], "--", label="ROM")
        plt.xlabel("x")
        plt.ylabel("Flux")
        plt.grid()
        plt.legend()
        outpath = f"results/{prefix}_{pid}_g_{g}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    error = np.linalg.norm(np.array(rom_vals) - np.array(fom_vals)) / np.linalg.norm(fom_vals)
    return error