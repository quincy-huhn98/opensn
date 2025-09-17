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


def update_xs(in_file, out_file, sigma_t_vec, S):
    with open(in_file, "r") as f:
        lines = f.readlines()

    # --- SIGMA_T block ---
    b = next(i for i, s in enumerate(lines) if "SIGMA_T_BEGIN" in s)
    e = next(i for i, s in enumerate(lines) if "SIGMA_T_END"   in s)
    for i in range(b+1, e):
        toks = lines[i].split()
        g = int(toks[0])
        toks[1] = f"{float(sigma_t_vec[g]):.12g}"
        lines[i] = " ".join(toks) + "\n"

    # --- TRANSFER_MOMENTS block ---
    tb = next(i for i, s in enumerate(lines) if "TRANSFER_MOMENTS_BEGIN" in s)
    te = next(i for i, s in enumerate(lines) if "TRANSFER_MOMENTS_END"   in s)

    G = len(sigma_t_vec)
    new_tm = []
    for gprime in range(G):
        for g in range(G):
            val = float(S[gprime][g])
            new_tm.append(f"M_GPRIME_G_VAL 0 {gprime} {g} {val:.12g}\n")

    lines[tb+1:te] = new_tm

    with open(out_file, "w") as f:
        f.writelines(lines)



# Sampling training points
bounds = [[0.5,1.0],[0.5,1.0],[7.5,12.5],[7.5,12.5]]
num_params = 100

params = sample_parameter_space(bounds, num_params)

S_abs = [[0.0, 0.0],
         [0.0, 0.0]]
sigma_t_scatt = [1.0, 1.0]

# OFFLINE PHASE
phase = 0

for i in range(num_params):
    S_scatt = [[1-params[i,0], params[i,0]],
            [0.0, params[i,1]]]
    update_xs("scatterer_base.txt", "scatterer.xs", sigma_t_scatt, S_scatt)
    
    sigma_t_abs = [params[i,2], params[i,3]]
    update_xs("absorber_base.txt", "absorber.xs", sigma_t_abs, S_abs)
    
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_2gcheckerboard.py \
                   -p phase={} -p p_id={}".format(phase,i)
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = process.communicate()
    print("Output:", output)
    print("Errors:", errors)

# MERGE PHASE
phase = 1

cmd = "mpiexec -n=4 ../../build/python/opensn -i base_2gcheckerboard.py -p phase={} -p p_id={}".format(phase, i)
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
myoutput = open("errors.txt", "w")
for i in range(num_params):
    S_scatt = [[1-params[i,0], params[i,0]],
            [0.0, params[i,1]]]
    update_xs("scatterer_base.txt", "scatterer.xs", sigma_t_scatt, S_scatt)

    sigma_t_abs = [params[i,2], params[i,3]]
    update_xs("absorber_base.txt", "absorber.xs", sigma_t_abs, S_abs)

    
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_2gcheckerboard.py \
                   -p phase={} -p p_id={}".format(phase,i)

    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=myoutput, text=True)
    output, errors = process.communicate()
    print("Output:", output)
    print("Errors:", errors)

np.savetxt("data/params.txt", params)

# Generate Test Data
test_scatt_1 = np.random.uniform(0.5,1.0,10)
test_scatt_2 = np.random.uniform(0.5,1.0,10)
test_abs_1 = np.random.uniform(7.5,12.5,10)
test_abs_2 = np.random.uniform(7.5,12.5,10)
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
    S_scatt = [[1-test[i,0], test[i,0]],
            [0.0, test[i,1]]]
    update_xs("scatterer_base.txt", "scatterer.xs", sigma_t_scatt, S_scatt)
    
    sigma_t_abs = [test[i,2], test[i,3]]
    update_xs("absorber_base.txt", "absorber.xs", sigma_t_abs, S_abs)

    phase = 4
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_2gcheckerboard.py \
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
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_2gcheckerboard.py \
                -p phase={} -p p_id={}".format(phase,i)
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
            num_groups = f.attrs["num_groups"]

        all_x.append(x)
        all_y.append(y)
        all_values.append(values)

    # Concatenate arrays across ranks
    rom_X = np.concatenate(all_x)
    rom_Y = np.concatenate(all_y)
    rom_V = np.zeros([len(rom_X),num_groups])
    for g in range(num_groups):
        rom_V[:,g] = np.concatenate(all_values)[len(rom_X)*g:len(rom_X)*(g+1)]

    # --- Interpolated heatmap ---
    rom_xi = np.linspace(np.min(rom_X), np.max(rom_X), 200)
    rom_yi = np.linspace(np.min(rom_Y), np.max(rom_Y), 200)
    rom_Xi, rom_Yi = np.meshgrid(rom_xi, rom_yi)

    rom_Zi_0 = scipy.interpolate.griddata((rom_X, rom_Y), rom_V[:,0], (rom_Xi, rom_Yi), method='linear')
    rom_Zi_masked_0 = np.ma.masked_where(rom_Zi_0 <= 0, rom_Zi_0)

    rom_Zi_1 = scipy.interpolate.griddata((rom_X, rom_Y), rom_V[:,1], (rom_Xi, rom_Yi), method='linear')
    rom_Zi_masked_1 = np.ma.masked_where(rom_Zi_1 <= 0, rom_Zi_1)


    file_pattern = "output/fom{}.h5"

    fom_x = []
    fom_y = []
    fom_values = []

    for j in range(4):
        with h5py.File(file_pattern.format(j), "r") as f:
            x = f["mesh/nodes_x"][:]
            y = f["mesh/nodes_y"][:]
            values = f["values"][:]
            num_groups = f.attrs["num_groups"]

        fom_x.append(x)
        fom_y.append(y)
        fom_values.append(values)

    # Concatenate arrays across ranks
    fom_X = np.concatenate(fom_x)
    fom_Y = np.concatenate(fom_y)
    fom_V = np.zeros([len(fom_X),num_groups])
    for g in range(num_groups):
        fom_V[:,g] = np.concatenate(fom_values)[len(fom_X)*g:len(fom_X)*(g+1)]

    # --- Interpolated heatmap ---
    fom_xi = np.linspace(np.min(fom_X), np.max(fom_X), 200)
    fom_yi = np.linspace(np.min(fom_Y), np.max(fom_Y), 200)
    fom_Xi, fom_Yi = np.meshgrid(fom_xi, fom_yi)

    fom_Zi_0 = scipy.interpolate.griddata((rom_X, rom_Y), fom_V[:,0], (rom_Xi, rom_Yi), method='linear')
    fom_Zi_masked_0 = np.ma.masked_where(fom_Zi_0 <= 0, fom_Zi_0)

    fom_Zi_1 = scipy.interpolate.griddata((rom_X, rom_Y), fom_V[:,1], (rom_Xi, rom_Yi), method='linear')
    fom_Zi_masked_1 = np.ma.masked_where(fom_Zi_1 <= 0, fom_Zi_1)

    # Shared color limits
    combined_min = min(np.min(rom_Zi_masked_0), np.min(fom_Zi_masked_0), np.min(rom_Zi_masked_1), np.min(fom_Zi_masked_1))
    combined_max = max(np.max(rom_Zi_masked_0), np.max(fom_Zi_masked_0), np.max(rom_Zi_masked_1), np.max(fom_Zi_masked_1))

    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(rom_Xi, rom_Yi, rom_Zi_masked_0, shading='auto', cmap='viridis',
                    norm=LogNorm(vmin=combined_min, vmax=combined_max))
    plt.colorbar(bar)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_rom{}_g0.jpg".format(i))
    plt.close()

    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(rom_Xi, rom_Yi, rom_Zi_masked_1, shading='auto', cmap='viridis',
                    norm=LogNorm(vmin=combined_min, vmax=combined_max))
    plt.colorbar(bar)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_rom{}_g1.jpg".format(i))
    plt.close()


    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(rom_Xi, rom_Yi, fom_Zi_masked_0, shading='auto', cmap='viridis',
                    norm=LogNorm(vmin=combined_min, vmax=combined_max))
    plt.colorbar(bar)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_fom{}_g0.jpg".format(i))
    plt.close()

    plt.figure(figsize=(8,6))
    bar = plt.pcolormesh(rom_Xi, rom_Yi, fom_Zi_masked_1, shading='auto', cmap='viridis',
                    norm=LogNorm(vmin=combined_min, vmax=combined_max))
    plt.colorbar(bar)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("results/fig_checkerboard_fom{}_g1.jpg".format(i))
    plt.close()

    errors.append(np.linalg.norm(fom_V-rom_V)/np.linalg.norm(fom_V))
    speedups.append(fom_time/rom_time)

    
print("Avg Error ", np.mean(errors))
np.savetxt("results/errors.txt", errors)
print("Avg Speedup ", np.mean(speedups))
np.savetxt("results/speedups.txt", speedups)