import numpy as np
import sys, os
sys.path.insert(0, os.path.realpath("../python"))

import plotting
import utils

# Sampling training points
bounds = [[0,5.0],[0.5,1.5],[7.5,12.5],[0.0,0.5],[0.1,1]]
num_params = 50

params = utils.sample_parameter_space(bounds, num_params)
#params = np.loadtxt("data/params.txt")

#params = params[:num_params,:]
np.savetxt("data/interpolation_params.txt", params)
# OFFLINE PHASE
phase = 0

for i in range(num_params):
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py \
                   -p phase={} -p scatt_1={} -p scatt_2={} -p abs_1={} -p abs_2={} -p param_q={} -p p_id={}"\
                .format(phase,params[i][0],params[i][1],params[i][2],params[i][3],params[i][4],i)
    utils.run_opensn(cmd)

# MERGE PHASE
phase = 1

cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py -p phase={} -p p_id={}".format(phase, num_params-1)
utils.run_opensn(cmd)

plotting.plot_sv(num_groups=1)


# SYSTEMS PHASE
phase = 2
for i in range(num_params):
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py \
                   -p phase={} -p scatt_1={} -p scatt_2={} -p abs_1={} -p abs_2={} -p param_q={} -p p_id={}"\
                .format(phase,params[i][0],params[i][1],params[i][2],params[i][3],params[i][4],i)
    utils.run_opensn(cmd)

np.savetxt("data/params.txt", params)


# Generate Test Data
test_scatt_1 = np.random.uniform(0,5.0,10)
test_scatt_2 = np.random.uniform(0.5,1.5,10)
test_abs_1 = np.random.uniform(7.5,12.5,10)
test_abs_2 = np.random.uniform(0.0,0.5,10)
test_q = np.random.uniform(0.1,1,10)
test = np.append(test_scatt_1[:,np.newaxis], test_scatt_2[:,np.newaxis], axis=1)
test = np.append(test, test_abs_1[:,np.newaxis], axis=1)
test = np.append(test, test_abs_2[:,np.newaxis], axis=1)
test = np.append(test, test_q[:,np.newaxis], axis=1)
np.savetxt("data/validation.txt", test)
# test = np.loadtxt("data/validation.txt")

num_test = 10
errors = []
speedups = []

for i in range(num_test):
    # ONLINE PHASE
    phase = 3
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py \
            -p phase={} -p scatt_1={} -p scatt_2={} -p abs_1={} -p abs_2={} -p param_q={} -p p_id={}"\
            .format(phase,test[i][0],test[i][1],test[i][2],test[i][3],test[i][4],i)
    utils.run_opensn(cmd)
    rom_time = np.loadtxt("results/online_time.txt")

    # Reference FOM solution
    phase = 0
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_checkerboard.py \
            -p phase={} -p scatt_1={} -p scatt_2={} -p abs_1={} -p abs_2={} -p param_q={} -p p_id={}"\
            .format(phase,test[i][0],test[i][1],test[i][2],test[i][3],test[i][4],i)
    utils.run_opensn(cmd)
    fom_time = np.loadtxt("results/offline_time.txt")

    plotting.plot_2d_flux("output/fom{}.h5", ranks=range(4), prefix="fom", pid=i)
    plotting.plot_2d_flux("output/rom{}.h5", ranks=range(4), prefix="rom", pid=i)

    error = plotting.plot_2d_lineout(ranks=range(4), pid=i)

    errors.append(error)
    speedups.append(fom_time/rom_time)


print("Avg Error ", np.mean(errors))
np.savetxt("results/errors.txt", errors)
print("Avg Speedup ", np.mean(speedups))
np.savetxt("results/speedups.txt", speedups)