import numpy as np
import sys, os
sys.path.insert(0, os.path.realpath("../python"))

import plotting
import utils

# Sampling training points
bounds = [[0.0,1.0],[0.0,1.0]]
num_params = 100

params = utils.sample_parameter_space(bounds, num_params)

# OFFLINE PHASE
phase = 0

for i, param in enumerate(params):
    cmd = "mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p param_q={} -p p_id={}"\
                                                                    .format(phase,param[0], param[1],     i)
    utils.run_opensn(cmd)

# MERGE PHASE
phase = 1

cmd = "mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p p_id={}".format(phase, i)
utils.run_opensn(cmd)

plotting.plot_sv(num_groups=1)


# SYSTEMS PHASE
phase = 2

for i, param in enumerate(params):
    cmd = "mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p param_q={} -p p_id={}"\
                                                                    .format(phase,param[0], param[1],     i)
    utils.run_opensn(cmd)

np.savetxt("data/params.txt", params)

# Generate Test Data
test = np.random.uniform(0,1,[10,2])

errors = []
speedups = []

for i, param in enumerate(test):
    # ONLINE PHASE
    phase = 3
    cmd = "mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p param_q={} -p p_id={}"\
                                                                    .format(phase,param[0],param[1],     i)
    utils.run_opensn(cmd)
    rom_time = np.loadtxt("results/online_time.txt")

    # Reference FOM solution
    phase = 0
    cmd = "mpiexec -n 2 ../../build/python/opensn -i base_reed.py -p phase={} -p scatt={} -p param_q={} -p p_id={}"\
                                                                    .format(phase,param[0],param[1],     i)
    utils.run_opensn(cmd)
    fom_time = np.loadtxt("results/offline_time.txt")

    error = plotting.plot_1d_flux("output/fom{}.h5", "output/rom{}.h5", ranks=range(2), pid=i)

    errors.append(error)
    speedups.append(fom_time/rom_time)

print("Avg Error ", np.mean(errors))
np.savetxt("results/errors.txt", errors)
print("Avg Speedup ", np.mean(speedups))
np.savetxt("results/speedups.txt", speedups)