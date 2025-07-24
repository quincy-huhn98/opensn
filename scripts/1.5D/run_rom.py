import numpy as np
import subprocess
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib.colors import LogNorm
import os

sigmas = np.random.uniform(0.1,2,48)

sigmas = np.append(sigmas, [0.1,2], axis=0)

phase = 0

for i, sigma in enumerate(sigmas):
    os.system("mpiexec -n 4 ../../build/python/opensn -i base_15D.py -p phase={} -p param_q={} -p p_id={}".format(phase, sigma, i))

phase = 1

print("Merge")
os.system("mpiexec -n 4 ../../build/python/opensn -i base_15D.py -p phase={} -p p_id={}".format(phase, i))

S = np.loadtxt("data/singular_values.txt")
plt.semilogy(S, 'o-')
plt.xlabel("Mode index")
plt.ylabel("Singular value")
plt.title("Singular value decay")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/svd_decay.jpg")
plt.close()

# phase = 2

# for i, sigma in enumerate(sigmas):
#     os.system("mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p p_id={}".format(phase, sigma, i))

np.savetxt("data/sigmas.txt", sigmas)

test = np.linspace(0.1,2,10)

error = 0

for i, param in enumerate(test):
    phase = 3
    os.system("mpiexec -n 4 ../../build/python/opensn -i base_15D.py -p phase={} -p param_q={} -p id=0".format(phase, param))

    phase = 0
    os.system("mpiexec -n 4 ../../build/python/opensn -i base_15D.py -p phase={} -p param_q={} -p id=0".format(phase, param))

    # Update this glob pattern to match your filenames
    # e.g., "mode_*.h5", "rhs*.h5", etc.
    file_pattern = "output/mi_rom{}.h5"

    all_x = []
    all_y = []
    all_values = []

    for j in range(4):
        with h5py.File(file_pattern.format(j), "r") as f:
            x = f["mesh/nodes_x"][:]
            y = f["mesh/nodes_y"][:]
            values = f["values"][:]

        all_x.append(x)
        all_y.append(y)
        all_values.append(values)

    # Concatenate arrays across ranks
    X = np.concatenate(all_x)
    Y = np.concatenate(all_y)
    V = np.concatenate(all_values)

    # --- Interpolated heatmap ---
    xi = np.linspace(np.min(X), np.max(X), 200)
    yi = np.linspace(np.min(Y), np.max(Y), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = scipy.interpolate.griddata((X, Y), V, (Xi, Yi), method='linear')
    Zi_masked = np.ma.masked_where(Zi <= 0, Zi)

    # Update this glob pattern to match your filenames
    # e.g., "mode_*.h5", "rhs*.h5", etc.
    file_pattern = "output/fom{}.h5"

    fom_x = []
    fom_y = []
    fom_values = []

    for j in range(4):
        with h5py.File(file_pattern.format(j), "r") as f:
            x = f["mesh/nodes_x"][:]
            y = f["mesh/nodes_y"][:]
            values = f["values"][:]

        fom_x.append(x)
        fom_y.append(y)
        fom_values.append(values)

    # Concatenate arrays across ranks
    fom_X = np.concatenate(fom_x)
    fom_Y = np.concatenate(fom_y)
    fom_V = np.concatenate(fom_values)

    # --- Interpolated heatmap ---
    fom_xi = np.linspace(np.min(fom_X), np.max(fom_X), 200)
    fom_yi = np.linspace(np.min(fom_Y), np.max(fom_Y), 200)
    fom_Xi, fom_Yi = np.meshgrid(fom_xi, fom_yi)

    fom_Zi = scipy.interpolate.griddata((fom_X, fom_Y), fom_V, (fom_Xi, fom_Yi), method='linear')
    fom_Zi_masked = np.ma.masked_where(fom_Zi <= 0, fom_Zi)

    # Shared color limits
    combined_min = min(np.min(Zi_masked), np.min(fom_Zi_masked))
    combined_max = max(np.max(Zi_masked), np.max(fom_Zi_masked))

    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(Xi, Yi, Zi_masked, shading='auto', cmap='viridis',
                     norm=LogNorm(vmin=combined_min, vmax=combined_max))
    plt.colorbar(bar, label="Scalar Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("ROM Solution Interpolated Heatmap c={}".format(param))
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_rom{}.jpg".format(i))
    plt.close()



    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(Xi, Yi, fom_Zi_masked, shading='auto', cmap='viridis',
                     norm=LogNorm(vmin=combined_min, vmax=combined_max))
    plt.colorbar(bar, label="Scalar Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("FOM Solution Interpolated Heatmap c={}".format(param))
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_fom{}.jpg".format(i))
    plt.close()
    
    error += np.linalg.norm(fom_V-V)/np.linalg.norm(fom_V)
    eps = 1e-12
    relative_error = np.abs(fom_Zi - Zi_masked) / (np.abs(fom_Zi) + eps)
    error_min = np.min(relative_error)
    error_max = np.max(relative_error)

    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(Xi, Yi, relative_error, shading='auto', cmap='viridis',
                        norm=LogNorm(vmin=max(error_min, 1e-6), vmax=error_max))
    plt.colorbar(bar, label="Scalar Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Error Interpolated Heatmap c={}".format(param))
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_err{}.jpg".format(i))

print(error/10)