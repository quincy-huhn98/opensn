import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

# sigmas = np.random.uniform(0,1,48)

# sigmas = np.append(sigmas, [0,1], axis=0)

# for i, sigma in enumerate(sigmas):
#     os.system("../build/python/opensn -i offline_reed.py -p scatt={} -p id={}".format(sigma, i))

# print("Merge")
# os.system("../build/python/opensn -i merge_reed.py -p id={}".format(i))

# for i, sigma in enumerate(sigmas):
#     os.system("../build/python/opensn -i systems_reed.py -p scatt={} -p id={}".format(sigma,i))

# np.savetxt("sigmas.txt", sigmas)

test = np.linspace(0,1,10)

error = 0

for i, sigma in enumerate(test):
    os.system("../build/python/opensn -i online_reed.py -p scatt={}".format(sigma))
    print(sigma)

    rom = np.zeros([2000,1])
    file_path = 'rom0.h5'
    with h5py.File(file_path, 'r') as file:
        rom[:,0] = file['values']

    fom = np.zeros([2000,1])
    file_path = 'fom0.h5'
    with h5py.File(file_path, 'r') as file:
        fom[:,0] = file['values']
    
    error += np.linalg.norm(rom-fom)/np.linalg.norm(fom)

    plt.plot(rom, "-", label="ROM")
    plt.plot(fom, "--", label="FOM")
    plt.grid()
    plt.legend()
    plt.savefig('reed_mi_{}.jpg'.format(i))
    plt.close()

print(error/10)