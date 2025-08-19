import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

for r in range(4):
    mean = np.zeros([2000,1])
    file_path = 'basis/centered_basis_{}0.h5'.format(r)
    with h5py.File(file_path, 'r') as file:
        length = file['values'].size
        mean[:length,0] = file['values']

    file_path = 'basis/centered_basis_{}1.h5'.format(r)
    with h5py.File(file_path, 'r') as file:
        mean[length:,0] = file['values']

    plt.plot(mean, "-", label="rank={}".format(r))
plt.grid()
plt.legend()
plt.savefig('results/centered_basis.jpg')
plt.close()