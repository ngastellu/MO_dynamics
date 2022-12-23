#!/usr/bin/env python

from os import path
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from trajectory import Trajectory


#datadir = '../../../simulation_outputs/test_MO_dynamics/'
datadir = '/Volumes/D-Rive2/Research/simulation_outputs/qcffpi_data/lammps_MD_300Kog' 

quick_traj = Trajectory(datadir,80420,80462,1,step=2)


rMOs = quick_traj.fetch_rMOs()
print(rMOs)

start_1 = perf_counter()
for k, rMO in enumerate(rMOs):
    r, M = rMO
    print(f'{k}\t{r.shape}\t{M.shape}')
end_1 = perf_counter()

start_2 = perf_counter()

files = [path.join(datadir,'sample-{k}/MO_coefs.dat') for k in range(80420,80462,2)]
