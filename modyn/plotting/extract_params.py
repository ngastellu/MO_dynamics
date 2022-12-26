#!/usr/bin/env pythonw

import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib import rcParams



data_dir = '/Volumes/D-Rive2/Research/simulation_outputs/MO_dynamics/300K_og'
lvls = ['LUMO', 'LUMO+1']
#lvls = ['LUMO']



# *** Plotting parameters ***

rcParams['text.usetex'] = True
rcParams['font.size'] = 20
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{bm}']
#rcParams['figure.figsize'] = [6.4, 4.8] #default
rcParams['figure.figsize'] = [15,5] #default

clrs = ['#4f5bd5', '#fa7e1e']

fig, axs = plt.subplots(2,1)


for l, clr in zip(lvls,clrs):
    data = np.load('%s/%s_yGammas_80000-81000-2.npy'%(data_dir,l))
    t, gL, gR = data[[0,3,4],:]
    signal = gL - gR
    #avg = np.mean(signal)
    transform = rfft(gL - gR)
    N = signal.shape[0]
    dts = t[1:] - t[:-1]
    dt = dts[0]
    print('SAMPLE RATE = ', dt)
    print('Max diff from SAMPLE RATE = ', np.max(np.abs(dts - dt)))
    freqs = rfftfreq(N,dt)
    axs[0].plot(t, gL-gR, ls='-', c=clr, lw=1.0, label=l)
    #axs[0].plot(t,avg*np.ones(N),c=clr,ls='--',lw=1.0)
    axs[1].plot(freqs,np.abs(transform),ls='-',c=clr, lw=1.0,label = l)
    #plt.plot(t,gR,ls='--',c=clr, lw=0.8,label = l)



axs[0].set_xlabel('$t$ [ps]')
axs[1].set_ylabel('$\langle\Gamma_{\\text{top}}\\rangle - \langle\Gamma_{\\text{bottom}}\\rangle$ [eV]')
axs[1].set_xlabel('$\omega$ [THz]')
axs[1].set_ylabel('$\langle\Delta\\tilde{\Gamma}\\rangle$ ')
plt.legend()
plt.tight_layout()
#plt.savefig('MOgams.png',dpi=100)
plt.show()
