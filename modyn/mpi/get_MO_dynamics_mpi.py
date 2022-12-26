#!/usr/bin/env python

import numpy as np
from ao_hamiltonian import read_MO_file, MO_rgyr
from mpi4py import MPI


def AO_data(n,nframe):
    MOfile = '../qcffpi_300K/frame-%d/MO_coefs.dat'%nframe
    pos, M = read_MO_file(MOfile, Natoms=3534)

    psi = np.abs(M[:,n])**2
    com =  (pos.T @ psi).T
    rgyr = MO_rgyr(pos, M, n, center_of_mass=com)
    ipr_metric = 1.0/np.sqrt(np.sum(psi**2))

    return com, rgyr, ipr_metric

def get_MO_trajectory(n,frames):
    """Returns the time evolution (obtained from NVT MD) of the COM, Rgyr and 1/sqrt(IPR) of the nth MO
    of a MAC structure."""

    dt = 0.0005 #in ps

    nframes = len(frames)
    print('* %d  %d  %d  %d *'%(rank, proc_frames[0], proc_frames[-1], nframes))

    if isinstance(n, (np.ndarray, list, tuple)):
        nlevels = len(n)
        com = np.zeros((nframes,nlevels,3), dtype=np.float64)
        rgyr = np.zeros((nframes,nlevels), dtype=np.float64)
        ipr_metric = np.zeros((nframes,nlevels), dtype=np.float64)

    else:
        com = np.zeros((nframes,3), dtype=np.float64)
        rgyr = np.zeros(nframes, dtype=np.float64)
        ipr_metric = np.zeros(nframes, dtype=np.float64)

    t = np.zeros(nframes, dtype=np.float64)
    
    for k, nstep in enumerate(frames):
        print(nstep)
        com[k,:], rgyr[k], ipr_metric[k] = AO_data(n,nstep)
        t[k] = (nstep - start) * dt

    return t, com, rgyr, ipr_metric




# ******* MAIN *******


comm = MPI.COMM_WORLD

nprocs = comm.Get_size()
rank = comm.Get_rank()

Natoms = 3534

nLUMO = Natoms // 2
start = 80000
end = 81000
frame_step = 2

frames = np.arange(start, end+frame_step, frame_step)
nframes = frames.shape[0]

m = nframes // nprocs

if rank < nprocs - 1:
    proc_frames = frames[m*rank:m*(rank+1)]

else:
    proc_frames = frames[m*rank:]

#end = 76700 + 1

interesting_lvls = [nLUMO-2, nLUMO+1]
lvl_names = ['HOMO-1', 'LUMO+1']

t, coms, rgyrs, iprs  = get_MO_trajectory(interesting_lvls, proc_frames)

for n, lvl in enumerate(lvl_names):
    data = np.vstack((t, coms[:,n,0], coms[:,n,1], rgyrs[:,n], iprs[:,n]))
    np.save('%s_traj_data_%d-%d-%d-%d.npy'%(lvl, start, end, frame_step, rank), data)


#t, coms, rgyrs, iprs = get_MO_trajectory(nLUMO, start, end, 2)
#t, coms, rgyrs, iprs = get_MO_trajectory(nLUMO)
#
#data = np.vstack((t, coms[:,0], coms[:,1], rgyrs, iprs))
#
#np.save('LUMO_traj_2e4-1e5-100.npy', data)
#
#t, coms, rgyrs, iprs = get_MO_trajectory(nLUMO-1)
#
#data = np.vstack((t, coms[:,0], coms[:,1], rgyrs, iprs))
#
#np.save('HOMO_traj_2e4-1e5-100.npy', data)
