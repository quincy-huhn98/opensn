import numpy as np
import subprocess
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib.colors import LogNorm

param_sigma = np.random.uniform(5,10,196)
param_c = np.random.uniform(0.5,1,196)

param_sigma = np.append([5,10,5,10], param_sigma , axis=0)
param_c = np.append([0.5,0.5,1,1], param_c,  axis=0)

# params = np.loadtxt("data/params.txt")

# param_q = params[:,0]
# param_c = params[:,1]

phase = 0

for i, param in enumerate(param_sigma):
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py -p phase={} -p sigma_t={} -p scatt={} -p p_id={}"\
                                                                            .format(phase,        param,      param_c[i],i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = process.communicate()
    print("Output:", output)
    print("Errors:", errors)

phase = 1

cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py -p phase={} -p p_id={}".format(phase, i)
args = cmd.split(" ")
print(args)
process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
output, errors = process.communicate()
print("Output:", output)
print("Errors:", errors)

S = np.loadtxt("data/singular_values.txt")
plt.semilogy(S, 'o-')
plt.xlabel("Mode index")
plt.ylabel("Singular value")
plt.title("Singular value decay")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/svd_decay.jpg")

phase = 2
myoutput = open("errors.txt", "w")
for i, param in enumerate(param_sigma):
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py -p phase={} -p sigma_t={} -p scatt={} -p p_id={}"\
                                                                            .format(phase,        param,      param_c[i],i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=myoutput, text=True)
    output, errors = process.communicate()
    print("Output:", output)
    print("Errors:", errors)

params = np.append(param_sigma[:,np.newaxis], param_c[:,np.newaxis], axis=1)
np.savetxt("data/params.txt", params)

test_sigma = np.random.uniform(5,10,10)
test_c = np.random.uniform(0.5,1,10)
test = np.append(test_sigma[:,np.newaxis], test_c[:,np.newaxis], axis=1)
np.savetxt("data/validation.txt", test)
test = np.loadtxt("data/validation.txt")

errors = []
speedups = []

for i, param in enumerate(test):
    phase = 4
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py -p phase={} -p sigma_t={} -p scatt={} -p p_id={}"\
                                                                            .format(phase,        param[0],   param[1],  i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error_out = process.communicate()
    print("Output:", output)
    print("Errors:", error_out)
    rom_time = np.loadtxt("results/online.txt")

    phase = 0
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py -p phase={} -p sigma_t={} -p scatt={} -p p_id={}"\
                                                                            .format(phase,        param[0],   param[1],  i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error_out = process.communicate()
    print("Output:", output)
    print("Errors:", error_out)
    fom_time = np.loadtxt("results/offline.txt")

    # Update this glob pattern to match your filenames
    # e.g., "mode_*.h5", "rhs*.h5", etc.
    file_pattern = "output/rom{}.h5"

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
    plt.title("ROM Solution Interpolated Heatmap params={}".format(param))
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
    plt.title("FOM Solution Interpolated Heatmap params={}".format(param))
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_fom{}.jpg".format(i))
    plt.close()
    
    errors.append(np.linalg.norm(fom_V-V)/np.linalg.norm(fom_V))
    eps = 1e-12
    relative_error = np.abs(fom_Zi - Zi_masked) #/ (np.abs(fom_Zi)+eps)
    speedups.append(fom_time/rom_time)
    error_min = np.min(relative_error)
    error_max = np.max(relative_error)

    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(Xi, Yi, relative_error, shading='auto', cmap='viridis')
    plt.colorbar(bar, label="Scalar Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Error Interpolated Heatmap params={}".format(param))
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_err{}.jpg".format(i))

        # Find the row index closest to y=4
    y_target = 3.5
    row_idx = np.argmin(np.abs(yi - y_target))

    # Extract data along y = 4
    rom_line = Zi[row_idx, :]
    fom_line = fom_Zi[row_idx, :]
    error_line = np.abs(fom_line - rom_line) #/ (np.abs(fom_line)+1e-12)

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
    plt.savefig(f"results/line_y{y_target}_rom_fom_{i}.jpg")
    plt.close()

    # Plot Error
    plt.figure(figsize=(8,5))
    plt.plot(xi, error_line, label='Relative Error', color='red')
    plt.xlabel('X')
    plt.ylabel('Relative Error')
    plt.title(f'Relative Error at y={y_target}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/line_y{y_target}_error_{i}.jpg")
    plt.close()

print("Avg Error ", np.mean(errors))
np.savetxt("results/errors.txt", errors)
print("Avg Speedup ", np.mean(speedups))
np.savetxt("results/speedups.txt", speedups)