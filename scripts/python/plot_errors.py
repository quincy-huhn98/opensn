import numpy as np
import matplotlib.pyplot as plt

problem = "../checkerboard"

# List of sample sizes
sample_sizes = [100, 150, 200, 250, 300, 350, 400, 450, 500]

# Collect error data
all_errors = []
for size in sample_sizes:
    filename = f"{problem}/results/errors_{size}.txt"
    data = np.loadtxt(filename)
    all_errors.append(data)

# Make the boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(all_errors, positions=sample_sizes, widths=20, showfliers=False)

plt.xlabel("Training Set Size")
plt.ylabel("$L_2$ Error")
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig(f'{problem}/results/errors_set_size.jpg')
plt.close()

# Collect speedup data
all_speedups = []
for size in sample_sizes:
    filename = f"{problem}/results/speedups_{size}.txt"
    data = np.loadtxt(filename)
    all_speedups.append(data)

# Make the boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(all_speedups, positions=sample_sizes, widths=20, showfliers=False)

plt.xlabel("Training Set Size")
plt.ylabel("Speedup")
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig(f'{problem}/results/speedups_set_size.jpg')

# List of ranks
ranks = [5, 10, 15, 20]

# Collect error data
all_errors = []
for rank in ranks:
    filename = f"{problem}/results/errors_{rank}.txt"
    data = np.loadtxt(filename)
    all_errors.append(data)

# Make the boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(all_errors, positions=ranks, widths=2, showfliers=False)

plt.xlabel("Rank")
plt.ylabel("$L_2$ Error")
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig(f'{problem}/results/errors_rank.jpg')
plt.close()

# Collect speedup data
all_speedups = []
for rank in ranks:
    filename = f"{problem}/results/speedups_{rank}.txt"
    data = np.loadtxt(filename)
    all_speedups.append(data)

# Make the boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(all_speedups, positions=ranks, widths=2, showfliers=False)

plt.xlabel("Rank")
plt.ylabel("Speedup")
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig(f'{problem}/results/speedups_rank.jpg')