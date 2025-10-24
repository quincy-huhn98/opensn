import numpy as np
import subprocess
import h5py

def run_opensn(cmd):
    args = cmd.split(" ")
    print(args)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = process.communicate()
    print("Output:", output)
    print("Errors:", errors)


def load_2d_flux(file_pattern, ranks, moment=0):
    """Load (x, y, flux) grouped by energy group from HDF5 files."""
    with h5py.File(file_pattern.format(ranks[0]), "r") as f0:
        num_groups = int(f0.attrs["num_groups"])
        num_moments = int(f0.attrs["num_moments"])

    xs = [[] for _ in range(num_groups)]
    ys = [[] for _ in range(num_groups)]
    vals = [[] for _ in range(num_groups)]

    for r in ranks:
        fp = file_pattern.format(r)
        with h5py.File(fp, "r") as f:
            x = f["mesh/nodes_x"][:]
            y = f["mesh/nodes_y"][:]
            values = f["values"][:]
            num_nodes = x.size

        values = values.reshape(num_nodes, num_moments, num_groups)
        for g in range(num_groups):
            xs[g].append(x)
            ys[g].append(y)
            vals[g].append(values[:, moment, g])

    for g in range(num_groups):
        xs[g] = np.concatenate(xs[g])
        ys[g] = np.concatenate(ys[g])
        vals[g] = np.concatenate(vals[g])

    return xs, ys, vals, num_groups

def load_1d_flux(file_pattern, ranks, moment=0):
    """Load concatenated 1-D (x, flux) data per energy group."""
    with h5py.File(file_pattern.format(ranks[0]), "r") as f0:
        num_groups = int(f0.attrs["num_groups"])
        num_moments = int(f0.attrs["num_moments"])

    xs = [[] for _ in range(num_groups)]
    vals = [[] for _ in range(num_groups)]

    for r in ranks:
        fp = file_pattern.format(r)
        with h5py.File(fp, "r") as f:
            x = f["mesh/nodes_z"][:]
            values = f["values"][:]
            num_nodes = x.size

        values = values.reshape(num_nodes, num_moments, num_groups)

        for g in range(num_groups):
            xs[g].append(x)
            vals[g].append(values[:, moment, g])

    for g in range(num_groups):
        xs[g] = np.concatenate(xs[g])
        vals[g] = np.concatenate(vals[g])

    return xs, vals, num_groups


def sample_parameter_space(bounds, n_samples):
    n_dim = len(bounds)
    n_vertices = 2**n_dim
    n_random = n_samples - n_vertices

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