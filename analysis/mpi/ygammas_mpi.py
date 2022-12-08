#!/usr/bin/env python

import numpy as np
from find_edge_carbons import concave_hull
from ao_hamiltonian import read_MO_file, AO_gammas, MO_gammas, read_energies
from inputoutput_nico import read_xsf
from mpi4py import MPI


Ha2eV = 27.2114


def AO_gammas_y(pos, gamma, edge_tol=3.0):
    if pos.shape[1] == 3:
        pos = pos[:,:2] #keep only x and y coords
    

    edge_bois = concave_hull(pos,3)
    ymin = np.min(pos[:,1])
    ymax = np.max(pos[:,1])
    top_edge = edge_bois[edge_bois[:,1] > ymax - edge_tol]
    bot_edge = edge_bois[edge_bois[:,1] < ymin + edge_tol]

    top_inds = np.zeros(top_edge.shape[0],dtype=int)
    bot_inds = np.zeros(bot_edge.shape[0],dtype=int)
    
    for k, r in enumerate(top_edge):
        top_inds[k] = np.all(pos == r, axis=1).nonzero()[0]

    for k, r in enumerate(bot_edge):
        bot_inds[k] = np.all(pos == r, axis=1).nonzero()[0]
    

    N = pos.shape[0]
    gammaT = np.zeros((N,N),dtype=float)
    gammaB = np.zeros((N,N),dtype=float)

    gammaT[top_inds,top_inds] = gamma
    gammaB[bot_inds,bot_inds] = gamma

    return gammaT, gammaB


#def MO_data(n,nframe, init_top, init_bot):
def MO_data(n,nframe,agt0,agb0):
    MOfile = '../qcffpi_300K/frame-%d/MO_coefs.dat'%nframe
    pos, M = read_MO_file(MOfile, Natoms=3534)
    
    agt, agb = AO_gammas_y(pos, 0.1)

    #agt0 = np.zeros((3534,3534),dtype=float)
    #agb0 = np.zeros((3534,3534),dtype=float)
    #agt0[init_top,init_top] = 0.1
    #agb0[init_bot,init_bot] = 0.1

    

    gamT, gamB = MO_gammas(M, agt, agb, return_diag=True) 
    gamT0, gamB0 = MO_gammas(M, agt0, agb0, return_diag=True) 

    return gamT[n], gamB[n], gamT0[n], gamB0[n]

#def fetch_data(n,start=20000,end=100000,framestep=100):
def fetch_data(n,frames, agt0, agb0):
    """Returns the time evolution (obtained from NVT MD) of the COM, Rgyr and 1/sqrt(IPR) of the nth MO
    of a MAC structure."""

    dt = 0.0005 #in ps

    #nframes = ( (end - start) // framestep )
    nframes = len(frames)
    print('* %d  %d  %d  %d *'%(rank, start, end, nframes))

    if isinstance(n, (np.ndarray, list, tuple)): # want data from several energy levels
        nstates = len(n)
        gamTs = np.zeros((nframes,nstates), dtype=np.float64)
        gamBs = np.zeros((nframes,nstates), dtype=np.float64)
        gamT0s = np.zeros((nframes,nstates), dtype=np.float64)
        gamB0s = np.zeros((nframes,nstates), dtype=np.float64)

    else: #only want data about 1 energy level 
        gamTs = np.zeros(nframes, dtype=np.float64)
        gamBs = np.zeros(nframes, dtype=np.float64)
        gamT0s = np.zeros(nframes, dtype=np.float64)
        gamB0s = np.zeros(nframes, dtype=np.float64)


    ts = np.zeros(nframes, dtype=np.float64) 

    #for k, nstep in enumerate(range(start,end,framestep)):
    for k, nstep in enumerate(frames):
        print(nstep, flush=True)
        ts[k] = (nstep - start) * dt
        gamT, gamB, gamT0, gamB0 = MO_data(n, nstep, agt0, agb0)

        gamTs[k] = gamT
        gamBs[k] = gamB
        gamT0s[k] = gamT0
        gamB0s[k] = gamB0


    return ts, gamTs, gamBs, gamT0s, gamB0s


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


# Get initial couplings

init_pos, _ = read_xsf('../lammps_MD/300K/bigMAC-276_relaxed.xsf')
agt0, agb0 = AO_gammas_y(init_pos, 0.1)

np.save('ao_gammas_top_init-bigMAC-276.npy',agt0)
np.save('ao_gammas_bot_init-bigMAC-276.npy',agb0)

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
t, gamTs, gamBs, gamT0s, gamB0s  = fetch_data(interesting_lvls, proc_frames, agt0, agb0)

for n, lvl in enumerate(lvl_names):

    data = np.vstack((t, gamTs[:,n], gamBs[:,n], gamT0s[:,n], gamB0s[:,n]))
    np.save('%s_energy_Gammas_%d-%d-%d-%d.npy'%(lvl, start, end, frame_step, rank), data)
