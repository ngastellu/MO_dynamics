#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

for orbname in ['LUMO','LUMO+1']:

    #data = np.load('../../simulation_outputs/MO_dynamics/LUMO_traj_data_3.npy')
    #data = np.load('/Volumes/D-Rive2/Research/simulation_outputs/MO_dynamics/LUMO+1_traj_data_20000-100000-100.npy')
    data = np.load('/Users/nico/Desktop/simulation_outputs/MO_dynamics/300K_og/%s_traj_data_20000-100000-100.npy'%orbname)


    t, xs, ys, rgyrs, ipr_ms = data
    print(t.shape)

    t = 0.5*t/t[-1]


    rcParams['text.usetex'] = True
    rcParams['font.size'] = 28
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}  \usepackage{bm}'
    #rcParams['figure.figsize'] = [6.4, 4.8] #default
    rcParams['figure.figsize'] = [12.8,9.6] #default

    fig, ax1 = plt.subplots()

    minX = 147.617#150.543
    maxX = 252.333 - minX #248.405

    minY = 148.86#152.115
    maxY = 256.049 - minY#250.17

    yo = ax1.plot(xs-minX, ys-minY, '-', c='k', lw=0.5, alpha=0.5,zorder=1)
    ye = ax1.scatter(xs-minX, ys-minY, c=t, cmap='inferno', s=30.0, label='$\langle\\bm{R}(t)\\rangle$',zorder=2)
    ax1.plot([0, 0], [0, maxY], 'k--', lw=1.0)
    ax1.plot([0, maxX], [maxY, maxY], 'k--', lw=1.0)
    ax1.plot([maxX, maxX], [0, maxY], 'k--', lw=1.0)
    ax1.plot([0, maxX], [0, 0], 'k--', lw=1.0)

    cbar = fig.colorbar(ye, ax=ax1)

    ax1.set_aspect('equal')
    ax1.set_xlabel('$x$ [\AA]')
    ax1.set_ylabel('$y$ [\AA]')

    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig('%s_traj.png'%orbname, transparent=True,dpi=120)
    plt.show()


#fig, ax2 = plt.subplots()
#
#yo = ax2.plot(ipr_ms, rgyrs, 'k-', lw=0.5,zorder=1, alpha=0.5)
#ye = ax2.scatter(ipr_ms, rgyrs, c=t, cmap='inferno', s=50.0,zorder=2)
##ax2.plot([minX, minX], [minY, maxY], 'k--', lw=1.0)
##ax2.plot([minX, maxX], [maxY, maxY], 'k--', lw=1.0)
##ax2.plot([maX, maxX], [minY, maxY], 'k--', lw=1.0)
##ax2.plot([minX, maxX], [minY, minY], 'k--', lw=1.0)
#
#cbar = fig.colorbar(ye, ax=ax2)
#
##ax2.set_aspect('equal')
#ax2.set_xlabel('$1/\sqrt{\\text{IPR}}$')
#ax2.set_ylabel('$R_g$ [\AA]')
#
#
##plt.legend()
#plt.show()
