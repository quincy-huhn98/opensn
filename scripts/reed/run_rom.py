import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

sigmas = np.random.uniform(0,1,48)

sigmas = np.append(sigmas, [0,1], axis=0)

phase = 0

for i, sigma in enumerate(sigmas):
    os.system("mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p p_id={}".format(phase, sigma, i))

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

# phase = 2

# for i, sigma in enumerate(sigmas):
#     os.system("mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p p_id={}".format(phase, sigma, i))

np.savetxt("data/sigmas.txt", sigmas)

test = np.linspace(0,1,10)

error = 0

for i, sigma in enumerate(test):
    phase = 3
    os.system("mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p id=0".format(phase, sigma))
    print(sigma)
    phase = 0
    os.system("mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p id=0".format(phase, sigma))

    rom = np.zeros([2000,1])
    file_path = 'output/mi_rom0.h5'
    with h5py.File(file_path, 'r') as file:
        length = file['values'].size
        rom[:length,0] = file['values']

    file_path = 'output/mi_rom1.h5'
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
    
    error += np.linalg.norm(rom-fom)/np.linalg.norm(fom)

    plt.plot(rom, "-", label="ROM")
    plt.plot(fom, "--", label="FOM")
    plt.grid()
    plt.legend()
    plt.savefig('results/reed_mi_{}.jpg'.format(i))
    plt.close()

print(error/10)