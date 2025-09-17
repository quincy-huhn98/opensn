import numpy as np
import subprocess
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib.colors import LogNorm

def sample_parameter_space(bounds, n_samples):
    n_dim = len(bounds)
    n_vertices = 2**n_dim
    n_random = n_samples - n_vertices
    if n_random < 0:
        raise ValueError(f"n_samples must be at least {n_vertices} to include all vertices")

    # Random interior samples
    random_samples = np.array([
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(n_random)
    ])

    # Vertices of domain
    vertices = np.zeros((n_vertices, n_dim))
    for i in range(n_vertices):
        for d, (low, high) in enumerate(bounds):
            if (i >> d) & 1:
                vertices[i, d] = high
            else:
                vertices[i, d] = low

    samples = np.vstack([random_samples, vertices])
    return samples

# Sampling training points
bounds = [[0,5.0],[0.5,1.5],[7.5,12.5],[0.0,0.5]]
num_params = 100

params = sample_parameter_space(bounds, num_params)

# OFFLINE PHASE
phase = 0

for i in range(num_params):
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py \
                   -p phase={} -p scatt_1={} -p scatt_2={} -p abs_1={} -p abs_2={} -p p_id={}"\
                .format(phase,params[i][0],params[i][1],params[i][2],params[i][3],i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = process.communicate()
    print("Output:", output)
    print("Errors:", errors)

# MERGE PHASE
phase = 1

cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py -p phase={} -p p_id={}".format(phase, i)
args = cmd.split(" ")
print(args)
process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
output, errors = process.communicate()
print("Output:", output)
print("Errors:", errors)

# Plot singular values
S = np.loadtxt("data/singular_values.txt")
plt.semilogy(S, 'o-')
plt.xlabel("Mode index")
plt.ylabel("Singular value")
plt.title("Singular value decay")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/svd_decay.jpg")

# SYSTEMS PHASE
phase = 2
for i in range(num_params):
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py \
                   -p phase={} -p scatt_1={} -p scatt_2={} -p abs_1={} -p abs_2={} -p p_id={}"\
                .format(phase,params[i][0],params[i][1],params[i][2],params[i][3],i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = process.communicate()
    print("Output:", output)
    print("Errors:", errors)

np.savetxt("data/params.txt", params)

# Generate Test Data
test_scatt_1 = np.random.uniform(0,5.0,10)
test_scatt_2 = np.random.uniform(0.5,1.5,10)
test_abs_1 = np.random.uniform(7.5,12.5,10)
test_abs_2 = np.random.uniform(0.0,0.5,10)
test = np.append(test_scatt_1[:,np.newaxis], test_scatt_2[:,np.newaxis], axis=1)
test = np.append(test, test_abs_1[:,np.newaxis], axis=1)
test = np.append(test, test_abs_2[:,np.newaxis], axis=1)
np.savetxt("data/validation.txt", test)
test = np.loadtxt("data/validation.txt")

num_test = 10
errors = []
speedups = []

for i in range(num_test):
    # ONLINE PHASE
    phase = 4
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py \
                -p phase={} -p scatt_1={} -p scatt_2={} -p abs_1={} -p abs_2={} -p p_id={}"\
            .format(phase,test[i][0],test[i][1],test[i][2],test[i][3],i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error_out = process.communicate()
    print("Output:", output)
    print("Errors:", error_out)
    rom_time = np.loadtxt("results/online_time.txt")

    # Reference FOM solution
    phase = 0
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py \
                -p phase={} -p scatt_1={} -p scatt_2={} -p abs_1={} -p abs_2={} -p p_id={}"\
            .format(phase,test[i][0],test[i][1],test[i][2],test[i][3],i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error_out = process.communicate()
    print("Output:", output)
    print("Errors:", error_out)
    fom_time = np.loadtxt("results/offline_time.txt")

    # Plotting and Error Calculation
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
    plt.colorbar(bar)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_rom{}.jpg".format(i))
    plt.close()


    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(Xi, Yi, fom_Zi_masked, shading='auto', cmap='viridis',
                     norm=LogNorm(vmin=combined_min, vmax=combined_max))
    plt.colorbar(bar)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_fom{}.jpg".format(i))
    plt.close()
    
    errors.append(np.linalg.norm(fom_V-V)/np.linalg.norm(fom_V))
    eps = 1e-12
    relative_error = np.abs(fom_Zi - Zi_masked) / (np.abs(fom_Zi)+eps)
    speedups.append(fom_time/rom_time)
    error_min = np.min(relative_error)
    error_max = np.max(relative_error)

    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(Xi, Yi, relative_error, shading='auto', cmap='viridis', norm=LogNorm())
    plt.colorbar(bar)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_err{}.jpg".format(i))


print("Avg Error ", np.mean(errors))
np.savetxt("results/errors.txt", errors)
print("Avg Speedup ", np.mean(speedups))
np.savetxt("results/speedups.txt", speedups)