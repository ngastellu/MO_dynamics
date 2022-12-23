#!/usr/bin/env python

import numpy as np
from ao_hamiltonian import read_MO_file, AO_gammas, MO_gammas, read_energies
from mpi4py import MPI


Ha2eV = 27.2114


def MO_data(n,nframe):
    MOfile = '../qcffpi_300K/frame-%d/MO_coefs.dat'%nframe
    orbfile = '../qcffpi_300K/frame-%d/orb_energy.dat'%nframe

    occ, virt = read_energies(orbfile)
    energies = np.hstack((occ[:,1],virt[:,1])) * Ha2eV

    pos, M = read_MO_file(MOfile, Natoms=3534)
    
    agl, agr = AO_gammas(pos, 0.1)

    gamL, gamR = MO_gammas(M, agl, agr, return_diag=True) 

    return energies[n], gamL[n], gamR[n]

#def fetch_data(n,start=20000,end=100000,framestep=100):
def fetch_data(n,frames):
    """Returns the time evolution (obtained from NVT MD) of the COM, Rgyr and 1/sqrt(IPR) of the nth MO
    of a MAC structure."""

    dt = 0.0005 #in ps

    #nframes = ( (end - start) // framestep )
    nframes = len(frames)
    print('* %d  %d  %d  %d *'%(rank, start, end, nframes))

    if isinstance(n, (np.ndarray, list, tuple)): # want data from several energy levels
        nstates = len(n)
        gamLs = np.zeros((nframes,nstates), dtype=np.float64)
        gamRs = np.zeros((nframes,nstates), dtype=np.float64)
        energies = np.zeros((nframes,nstates), dtype=np.float64)

    else: #only want data about 1 energy level 
        gamLs = np.zeros(nframes, dtype=np.float64)
        gamRs = np.zeros(nframes, dtype=np.float64)
        energies = np.zeros(nframes, dtype=np.float64)


    ts = np.zeros(nframes, dtype=np.float64) 

    #for k, nstep in enumerate(range(start,end,framestep)):
    for k, nstep in enumerate(frames):
        print(nstep, flush=True)
        ts[k] = (nstep - start) * dt
        e, gamL, gamR = MO_data(n, nstep)

        energies[k] = e
        gamLs[k] = gamL
        gamRs[k] = gamR


    return ts, energies, gamLs, gamRs


# ****** MAIN *******

comm = MPI.COMM_WORLD

nprocs = comm.Get_size()
rank = comm.Get_rank()


Natoms = 3534

nLUMO = Natoms // 2

#start = 20000
#end = 100000
#frame_step = 100

start = 80000
end = 81000
frame_step=2


# This is hacky as heck. 

frames = np.arange(start, end+frame_step, frame_step)
nframes = frames.shape[0]
print(nframes)

m = nframes // nprocs

#start_proc = frames[m * rank]

if rank < nprocs - 1:
    #end_proc = frames[m * (rank+1) - 1]
    #end_proc = frames[m * (rank+1)]
    proc_frames = frames[m*rank:m*(rank+1)]

else:
    #end_proc = end
    proc_frames = frames[m*rank:]

interesting_lvls = np.arange(nLUMO-2,nLUMO+2)
lvl_names = ['HOMO-1', 'HOMO', 'LUMO', 'LUMO+1']

#t, es, gamLs, gamRs  = fetch_data(interesting_lvls, start_proc, end_proc, framestep=frame_step)
t, es, gamLs, gamRs  = fetch_data(interesting_lvls, proc_frames)

for n, lvl in enumerate(lvl_names):

    data = np.vstack((t, es[:,n], gamLs[:,n], gamRs[:,n]))
    np.save('%s_energy_Gammas_%d-%d-%d-%d.npy'%(lvl, start, end, frame_step, rank), data)

