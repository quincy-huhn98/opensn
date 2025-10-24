import numpy as np
import sys, os
sys.path.insert(0, os.path.realpath("../python"))

import plotting
import utils

# Sampling training points
bounds = [[0.5,1.0],[7.5,12.5]]
num_params = 100

params = utils.sample_parameter_space(bounds, num_params)

S_abs = [[0.0, 0.0],
         [0.0, 0.0]]
sigma_t_scatt = [1.0, 1.0]

# OFFLINE PHASE
phase = 0

for i in range(num_params):
    S_scatt = [[1-params[i,0], params[i,0]],
            [0.0, 1.0]]
    utils.update_xs("scatterer_base.txt", "data/scatterer.xs", sigma_t_scatt, S_scatt)
    
    sigma_t_abs = [params[i,1], params[i,1]]
    utils.update_xs("absorber_base.txt", "data/absorber.xs", sigma_t_abs, S_abs)
    
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_2gcheckerboard.py \
                   -p phase={} -p p_id={}".format(phase,i)
    utils.run_opensn(cmd)

# MERGE PHASE
phase = 1

cmd = "mpiexec -n=4 ../../build/python/opensn -i base_2gcheckerboard.py -p phase={} -p p_id={}".format(phase, i)
utils.run_opensn(cmd)

plotting.plot_sv(num_groups=2)


# SYSTEMS PHASE
phase = 2
for i in range(num_params):
    S_scatt = [[1-params[i,0], params[i,0]],
            [0.0, 1.0]]
    utils.update_xs("scatterer_base.txt", "data/scatterer.xs", sigma_t_scatt, S_scatt)

    sigma_t_abs = [params[i,1], params[i,1]]
    utils.update_xs("absorber_base.txt", "data/absorber.xs", sigma_t_abs, S_abs)
    
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_2gcheckerboard.py \
                   -p phase={} -p p_id={}".format(phase,i)
    utils.run_opensn(cmd)

np.savetxt("data/params.txt", params)


# Generate Test Data
test_scatt_1 = np.random.uniform(0.5,1.0,10)
test_abs_1 = np.random.uniform(7.5,12.5,10)
test = np.append(test_scatt_1[:,np.newaxis], test_abs_1[:,np.newaxis], axis=1)
np.savetxt("data/validation.txt", test)

test = np.loadtxt("data/validation.txt")

num_test = 10
errors = []
speedups = []

for i in range(num_test):
    # ONLINE PHASE
    S_scatt = [[1-test[i,0], test[i,0]],
            [0.0, 1.0]]
    utils.update_xs("scatterer_base.txt", "data/scatterer.xs", sigma_t_scatt, S_scatt)
    
    sigma_t_abs = [test[i,1], test[i,1]]
    utils.update_xs("absorber_base.txt", "data/absorber.xs", sigma_t_abs, S_abs)

    phase = 3
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_2gcheckerboard.py \
                -p phase={} -p scatt_1={} -p abs_1={} -p p_id={}"\
            .format(phase,test[i][0],test[i][1],i)
    utils.run_opensn(cmd)
    rom_time = np.loadtxt("results/online_time.txt")

    # Reference FOM solution
    phase = 0
    cmd = "mpiexec -n=4 ../../build/python/opensn -i base_2gcheckerboard.py \
                -p phase={} -p p_id={}".format(phase,i)
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