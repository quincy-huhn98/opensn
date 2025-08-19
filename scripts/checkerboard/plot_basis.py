import numpy as np
import subprocess
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib.colors import LogNorm

file_pattern = "data/mean_snapshot{}.h5"

basis_x = []
basis_y = []
basis_values = []

for j in range(4):
    with h5py.File(file_pattern.format(j), "r") as f:
        x = f["mesh/nodes_x"][:]
        y = f["mesh/nodes_y"][:]
        values = f["values"][:]

    basis_x.append(x)
    basis_y.append(y)
    basis_values.append(values)

# Concatenate arrays across ranks
basis_X = np.concatenate(basis_x)
basis_Y = np.concatenate(basis_y)
basis_V = np.concatenate(basis_values)

# --- Interpolated heatmap ---
basis_xi = np.linspace(np.min(basis_X), np.max(basis_X), 200)
basis_yi = np.linspace(np.min(basis_Y), np.max(basis_Y), 200)
basis_Xi, basis_Yi = np.meshgrid(basis_xi, basis_yi)

basis_Zi = scipy.interpolate.griddata((basis_X, basis_Y), basis_V, (basis_Xi, basis_Yi), method='linear')

plt.figure(figsize=(8,6))
bar = plt.pcolormesh(basis_Xi, basis_Yi, basis_Zi, shading='auto', cmap='viridis')
plt.colorbar(bar, label="Scalar Value")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Mean Snapshot")
plt.axis("equal")
plt.tight_layout()
plt.savefig("results/mean.jpg")
plt.close()

for r in range(1):
    # Update this glob pattern to match your filenames
    # e.g., "mode_*.h5", "rhs*.h5", etc.
    file_pattern = "basis/basis_"+str(r)+"{}.h5"

    basis_x = []
    basis_y = []
    basis_values = []

    for j in range(4):
        with h5py.File(file_pattern.format(j), "r") as f:
            x = f["mesh/nodes_x"][:]
            y = f["mesh/nodes_y"][:]
            values = f["values"][:]

        basis_x.append(x)
        basis_y.append(y)
        basis_values.append(values)

    # Concatenate arrays across ranks
    basis_X = np.concatenate(basis_x)
    basis_Y = np.concatenate(basis_y)
    basis_V = np.concatenate(basis_values)

    # --- Interpolated heatmap ---
    basis_xi = np.linspace(np.min(basis_X), np.max(basis_X), 200)
    basis_yi = np.linspace(np.min(basis_Y), np.max(basis_Y), 200)
    basis_Xi, basis_Yi = np.meshgrid(basis_xi, basis_yi)

    basis_Zi = scipy.interpolate.griddata((basis_X, basis_Y), basis_V, (basis_Xi, basis_Yi), method='linear')

    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(basis_Xi, basis_Yi, -basis_Zi, shading='auto', cmap='viridis', norm=LogNorm())
    plt.colorbar(bar, label="Scalar Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Basis r={}".format(r))
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/basis{}.jpg".format(r))
    plt.close()