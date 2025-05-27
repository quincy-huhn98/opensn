import numpy as np 
import h5py
import matplotlib.pyplot as plt

rom = np.zeros([2000,1])
file_path = 'rom0.h5'
with h5py.File(file_path, 'r') as file:
    rom[:,0] = file['values']

fom = np.zeros([2000,1])
file_path = 'fom0.h5'
with h5py.File(file_path, 'r') as file:
    fom[:,0] = file['values']

plt.plot(rom, "-", label="ROM")
plt.plot(fom, "--", label="FOM")
plt.grid()
plt.legend()
plt.savefig('reed_mi.jpg')