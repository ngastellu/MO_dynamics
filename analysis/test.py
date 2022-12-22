#!/usr/bin/env python

from os import path
import numpy as np
import matplotlib.pyplot as plt
from trajectory import Trajectory


#datadir = '../../../simulation_outputs/test_MO_dynamics/'
datadir = '/Volumes/D-Rive2/Research/simulation_outputs/qcffpi_data/lammps_MD_300Kog' 

quick_traj = Trajectory(datadir,80420,80462,1,step=2)

ee = np.array(list(quick_traj.fetch_energies()))
print(ee.shape)

plt.plot(ee[:,ee.shape[1]//2],'b-')
plt.plot(ee[:,1+ee.shape[1]//2],'r-')
plt.plot(ee[:,2+ee.shape[1]//2],'g-')
plt.show()

LUMOind = ee.shape[1] // 2
LUMOp1ind = LUMOind + 1

rMOs = quick_traj.fetch_rMOs()
print(rMOs)

for k, rMO in enumerate(rMOs):
    r, M = rMO
    print(f'{k}\t{r.shape}\t{M.shape}')





