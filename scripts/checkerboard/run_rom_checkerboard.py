import numpy as np
import subprocess
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib.colors import LogNorm

qs = np.random.uniform(0,2,48)

qs = np.append(qs, [0,2], axis=0)

for i, q in enumerate(qs):
    cmd = "srun --ntasks=4 ../../build/python/opensn -i offline_checkerboard.py -p param_q={} -p id={}".format(q, i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = process.communicate()
    print("Output:", output)
    print("Errors:", errors)


cmd = "srun --ntasks=4 ../../build/python/opensn -i merge_checkerboard.py -p id={}".format(49)
args = cmd.split(" ")
print(args)
process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
output, errors = process.communicate()
print("Output:", output)
print("Errors:", errors)

myoutput = open("errors.txt", "w")
for i, q in enumerate(qs):
    cmd = "srun --ntasks=4 ../../build/python/opensn -i systems_checkerboard.py -p param_q={} -p id={}".format(q,i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=myoutput, text=True)
    output, errors = process.communicate()
    print("Output:", output)
    print("Errors:", errors)


np.savetxt("params.txt", qs)

test = np.random.uniform(0,2,10)

# test = np.linspace(5,15,10)

#test = [9.0]

error = 0

for i, q in enumerate(test):
    myoutput = open("errors.txt", "w")
    cmd = "srun --ntasks=4 ../../build/python/opensn -i online_checkerboard.py -p param_q={} -p id=0".format(q)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=myoutput, text=True)
    output = process.communicate()
    print("Output:", output)

    # Update this glob pattern to match your filenames
    # e.g., "mode_*.h5", "rhs*.h5", etc.
    file_pattern = "rom{}.h5"

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

    print(f"Combined 4 files, total nodes: {len(X)}")

    # --- Interpolated heatmap ---
    xi = np.linspace(np.min(X), np.max(X), 200)
    yi = np.linspace(np.min(Y), np.max(Y), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = scipy.interpolate.griddata((X, Y), V, (Xi, Yi), method='linear')
    Zi_masked = np.ma.masked_where(Zi <= 0, Zi)

    plt.figure(figsize=(8,6))
    c = plt.pcolormesh(Xi, Yi, Zi_masked, shading='auto', cmap='viridis', norm=LogNorm())
    plt.colorbar(c, label="Scalar Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("ROM Solution Interpolated Heatmap q={}".format(q))
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("fig_checkerboard_rom{}.jpg".format(i))
    plt.close()

    # Update this glob pattern to match your filenames
    # e.g., "mode_*.h5", "rhs*.h5", etc.
    file_pattern = "fom{}.h5"

    all_x = []
    all_y = []
    fom_values = []

    for j in range(4):
        with h5py.File(file_pattern.format(j), "r") as f:
            x = f["mesh/nodes_x"][:]
            y = f["mesh/nodes_y"][:]
            values = f["values"][:]

        all_x.append(x)
        all_y.append(y)
        fom_values.append(values)

    # Concatenate arrays across ranks
    X = np.concatenate(all_x)
    Y = np.concatenate(all_y)
    fom_V = np.concatenate(fom_values)

    print(f"Combined 4 files, total nodes: {len(X)}")

    # --- Interpolated heatmap ---
    xi = np.linspace(np.min(X), np.max(X), 200)
    yi = np.linspace(np.min(Y), np.max(Y), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    fom_Zi = scipy.interpolate.griddata((X, Y), fom_V, (Xi, Yi), method='linear')
    fom_Zi_masked = np.ma.masked_where(fom_Zi <= 0, fom_Zi)

    plt.figure(figsize=(8,6))
    c = plt.pcolormesh(Xi, Yi, fom_Zi_masked, shading='auto', cmap='viridis', norm=LogNorm())
    plt.colorbar(c, label="Scalar Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("FOM Solution Interpolated Heatmap q={}".format(q))
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("fig_checkerboard_fom{}.jpg".format(i))
    plt.close()
    
    error += np.linalg.norm(V-fom_V)/np.linalg.norm(fom_V)

    plt.figure(figsize=(8,6))
    c = plt.pcolormesh(Xi, Yi, np.abs(fom_Zi_masked-Zi_masked)/fom_Zi_masked, shading='auto', cmap='viridis')
    plt.colorbar(c, label="Scalar Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Error Interpolated Heatmap q={}".format(q))
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("fig_checkerboard_err{}.jpg".format(i))

print(error/i)