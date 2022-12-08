#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams



data_dir = '/Users/nico/Desktop/simulation_outputs/MO_dynamics/300K_og'

rcParams['text.usetex'] = True
rcParams['font.size'] = 25
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}  \usepackage{bm}'
#rcParams['figure.figsize'] = [6.4, 4.8] #default
rcParams['figure.figsize'] = [15,5] #default

lvls = ['LUMO', 'LUMO+1']
clrs = ['#4f5bd5', '#fa7e1e']

for l, clr in zip(lvls[-2:],clrs):
    data = np.load('%s/%s_energy_Gammas_20000-100000-100.npy'%(data_dir,l))
    #data = np.load('/Volumes/D-Rive2/Research/simulation_outputs/MO_dynamics/%s_energy_Gammas_80000-81000-2.npy'%l)
    t, e = data[:2,:]
    plt.plot(t,e,ls='-',c=clr, lw=0.8,label = l)

plt.legend()
plt.show()


for l, clr in zip(lvls[-2:],clrs):
    #data = np.load('/Volumes/D-Rive2/Research/simulation_outputs/MO_dynamics/%s_energy_Gammas_20000-100000-100.npy'%l)
    data = np.load('%s/%s_energy_Gammas_80000-81000-2.npy'%(data_dir,l))
    t, gL, gR = data[[0,2,3],:]
    plt.plot(t,gL-gR,ls='-',c=clr, lw=0.8,label = l)
    #plt.plot(t,gR,ls='--',c=clr, lw=0.8,label = l)

plt.legend()
plt.show()

for l, clr in zip(lvls[-2:],clrs):
    data = np.load('%s/%s_yGammas_80000-81000-2.npy'%(data_dir,l))
    t, gL, gR = data[0:3,:]
    plt.plot(t,gL-gR,ls='-',c=clr, lw=0.8,label = l)
    #plt.plot(t,gR,ls='--',c=clr, lw=0.8,label = l)

plt.legend()
plt.show()

for l, clr in zip(lvls[-2:],clrs):
    data = np.load('%s/%s_yGammas_80000-81000-2.npy'%(data_dir,l))
    t, gL, gR = data[[0,3,4],:]
    plt.plot(t,gL-gR,ls='-',c=clr, lw=1.0,label = l)
    #plt.plot(t,gR,ls='--',c=clr, lw=0.8,label = l)

plt.xlabel('$t$ [ps]')
plt.ylabel('$\langle\Gamma_{top}\\rangle - \langle\Gamma_{bot}\\rangle$ [eV]')
plt.legend()
plt.tight_layout()
#plt.savefig('MOgams.png',dpi=100)
plt.show()


#for l, clr in zip(lvls[-2:],clrs):
#    #data = np.load('/Volumes/D-Rive2/Research/simulation_outputs/MO_dynamics/%s_energy_Gammas_20000-100000-100.npy'%l)
#    data = np.load('/Volumes/D-Rive2/Research/simulation_outputs/MO_dynamics/%s_energy_Gammas_80000-81000-2.npy'%l)
#    t, gR = data[[0,2],:]
#    plt.plot(t,gR,ls='-',c=clr, lw=0.8,label = l)
#
#plt.show()
