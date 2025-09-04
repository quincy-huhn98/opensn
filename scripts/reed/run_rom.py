import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

cs = np.random.uniform(0,1,96)

qs = np.random.uniform(0,1,96)

cs = np.append(cs, [0,1,0,1], axis=0)

qs = np.append(qs, [0,0,1,1], axis=0)

phase = 0

for i, c in enumerate(cs):
    os.system("mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p param_q={} -p p_id={}"\
                                                                        .format(phase,      c,            qs[i],     i))

phase = 1

print("Merge")
os.system("mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p p_id={}".format(phase, i))

S = np.loadtxt("data/singular_values.txt")
plt.semilogy(S, 'o-')
plt.xlabel("Mode index")
plt.ylabel("Singular value")
plt.title("Singular value decay")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/svd_decay.jpg")
plt.close()

phase = 2

for i, c in enumerate(cs):
    os.system("mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p param_q={} -p p_id={}"\
                                                                        .format(phase,      c,            qs[i],     i))

params = np.append(qs[:,np.newaxis], cs[:,np.newaxis], axis=1)

np.savetxt("data/params.txt", params)

test = np.random.uniform(0,1,[20,2])

errors = []
speedups = []
int_errors = []

for i, param in enumerate(test):
    phase = 4
    os.system("mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p param_q={} -p scatt={}  -p p_id={}"\
                                                                        .format(phase,       param[0],param[1],    i))
    rom_time = np.loadtxt("results/online.txt")

    print(param)
    phase = 0
    os.system("mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p param_q={} -p scatt={}  -p p_id={}"\
                                                                        .format(phase,     param[0],param[1],    i))

    fom_time = np.loadtxt("results/offline.txt")
    speedups.append(fom_time/rom_time)
    int_errors.append(np.loadtxt("results/int_error.txt"))

    rom = np.zeros([2000,1])
    file_path = 'output/rom0.h5'
    with h5py.File(file_path, 'r') as file:
        length = file['values'].size
        rom[:length,0] = file['values']

    file_path = 'output/rom1.h5'
    with h5py.File(file_path, 'r') as file:
        rom[length:,0] = file['values']

    fom = np.zeros([2000,1])
    file_path = 'output/fom0.h5'
    with h5py.File(file_path, 'r') as file:
        length = file['values'].size
        fom[:length,0] = file['values']

    file_path = 'output/fom1.h5'
    with h5py.File(file_path, 'r') as file:
        fom[length:,0] = file['values']
    
    errors.append(np.linalg.norm(rom-fom)/np.linalg.norm(fom))

    plt.plot(rom, "-", label="ROM")
    plt.plot(fom, "--", label="FOM")
    plt.grid()
    plt.legend()
    plt.savefig('results/reed_ommi_{}.jpg'.format(i))
    plt.close()

print(np.mean(errors))
print(np.mean(speedups))
print(np.mean(int_errors))
print(int_errors)