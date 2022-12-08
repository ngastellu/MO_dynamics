#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


#data = np.load('../../simulation_outputs/MO_dynamics/LUMO_traj_data_3.npy')

lvl = 'LUMO'

traj_data = np.load('/Volumes/D-Rive2/Research/simulation_outputs/MO_dynamics/%s_traj_data_2e4-1e5-100.npy'%lvl)

e_data = np.load('/Volumes/D-Rive2/Research/simulation_outputs/MO_dynamics/%s_energy_Gammas_20000-100000-100.npy'%lvl) 

t, xs, ys, rgyrs, ipr_ms = traj_data
t2, e, gL, gR = e_data[:,:-1]

#assert np.all(t == t2), 'Time arrays don\'t match!\nShape of t1 = %d.\nShape of t2 = %d'%(t.shape[0],t2.shape[0])


rcParams['text.usetex'] = True
rcParams['font.size'] = 16
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{bm}']

fig, ax1 = plt.subplots()

minX = 147.617#150.543
maxX = 252.333#248.405

minY = 148.86#152.115
maxY = 256.049#250.17

yo = ax1.plot(xs, ys, '-', c='k', lw=0.8, alpha=0.5,zorder=1)
ye = ax1.scatter(xs, ys, c=t*1000, cmap='inferno', s=10.0, label='$\langle\\bm{R}(t)\\rangle$',zorder=2)
ax1.plot([minX, minX], [minY, maxY], 'k--', lw=1.0)
ax1.plot([minX, maxX], [maxY, maxY], 'k--', lw=1.0)
ax1.plot([maxX, maxX], [minY, maxY], 'k--', lw=1.0)
ax1.plot([minX, maxX], [minY, minY], 'k--', lw=1.0)

cbar = fig.colorbar(ye, ax=ax1)

ax1.set_aspect('equal')
ax1.set_xlabel('$x$ [\AA]')
ax1.set_ylabel('$y$ [\AA]')

plt.legend(loc='upper right')
plt.show()

fig, ax2 = plt.subplots()


minX = 147.617#150.543
maxX = 252.333#248.405

minY = 148.86#152.115
maxY = 256.049#250.17

ya = ax2.plot(xs, ys, '-', c='k', lw=0.8, alpha=0.5,zorder=1)
yi = ax2.scatter(xs, ys, c=e, cmap='viridis', s=10.0, label='$\langle\\bm{R}(t)\\rangle$',zorder=2)
ax2.plot([minX, minX], [minY, maxY], 'k--', lw=1.0)
ax2.plot([minX, maxX], [maxY, maxY], 'k--', lw=1.0)
ax2.plot([maxX, maxX], [minY, maxY], 'k--', lw=1.0)
ax2.plot([minX, maxX], [minY, minY], 'k--', lw=1.0)

cbar = fig.colorbar(yi, ax=ax2)

ax2.set_aspect('equal')
ax2.set_xlabel('$x$ [\AA]')
ax2.set_ylabel('$y$ [\AA]')

plt.legend(loc='upper right')
plt.show()
