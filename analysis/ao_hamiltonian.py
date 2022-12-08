#!/usr/bin/env pythonw

import os
import numpy as np
import scipy
from scipy import sparse
import scipy.sparse.linalg as sLA
import matplotlib.pyplot as plt
from matplotlib import rcParams
from find_edge_carbons import concave_hull

def get_Natoms(infile):
    with open(infile) as fo:
        line1 = fo.readline()
        L = len(line1.split())
        Natoms = L - 5
        if L < 502:
            return Natoms
        else:
            init_line = False
            L = 0
            while not init_line:
                Natoms += L
                l = fo.readline()
                L = len(l.split())
                init_line = (L == 502)

    return Natoms


def read_MO_file(infile,Natoms=None):
    """Reads MO coefs output file from QCFFPI and returns a list of atomic positions and a AO -> MO
    transformation matrix with elements M_ij = <AO_i|MO_j>."""
    
    if Natoms == None:
        Natoms = get_Natoms(infile)
    with open(infile) as fo:
        lines = fo.readlines()

    positions = np.zeros((Natoms,3),dtype=float)
    MO_matrix = np.zeros((Natoms,Natoms),dtype=np.float64)

    if Natoms <= 497:
        nlines_per_atom = 1
    else:
        nlines_per_atom = int(1 + np.ceil((Natoms-497)/500))

    for k, line in enumerate(lines):
        #print(k)
        atom_index = k // nlines_per_atom
        if atom_index == Natoms: break
        split_line = line.split()
        if k % nlines_per_atom == 0:
            counter = 0
            positions[atom_index,:] = list(map(float,split_line[2:5]))
            MO_matrix[atom_index,:497] = list(map(float,split_line[5:]))
            counter += 497
        else:
            n = len(split_line)
            MO_matrix[atom_index,counter:counter+n] = list(map(float,split_line))
            counter += n

    return positions, MO_matrix


def read_energies(orb_file,Natoms=-1):
    """Reads energies from QCCFPI output file `orb_file` and returns two arrays [i,e_i] 
    (where i labels the MOs) of the energies of occupied and virtual MOs.
    
    *** ASSUMES ENERGIES ARE SORTED *** 

    If Natoms is specified, only `Natoms` lines are read from `orbfile`. Otherwise, precisely
    the first half of the lines are read (regular QCFFPI runs return the energies for the 
    initial config and for the config after the 1st MD timestep.
    """

    with open(orb_file) as fo:
        lines = fo.readlines()
    if Natoms == -1:
        nlines_to_read = int(len(lines)/2)

    else:
        nlines_to_read = Natoms
    all_energies = np.array([list(map(float,l.split())) for l in lines[:nlines_to_read]])
    lumo_index = int(len(all_energies)/2)

    occupied = all_energies[:lumo_index,:]
    virtual = all_energies[lumo_index:,:]

    #print(np.max(all_energies[:,1]) - np.min(all_energies[:,1]))

    return occupied, virtual

def read_Hao(Hao_file, Natoms):
    """Reads AO Hamiltonian from Hao.dat file output by QCFFPI."""
    
    Hao = np.zeros((Natoms,Natoms),dtype=float)

    with open(Hao_file) as fo:
        lines = fo.readlines()[:Natoms]

    if Natoms <= 500:
        nlines_per_row = 1
    elif Natoms % 500 == 0:
        nlines_per_row = Natoms // 500
    else:
        nlines_per_row = 1 + (Natoms // 500)
    
    for k, line in enumerate(lines):
        row_index = k // nlines_per_row
        split_line = line.split()

        if k % nlines_per_row == 0:
            counter = 0
            Hao[row_index,:500] = list(map(float, split_line))
            counter += 500
        else:
            n = len(split_line)
            Hao[row_index,counter:counter+n] = list(map(float, split_line))
            counter += n

    return Hao

        

def AO_hamiltonian(M,energy_lvls,delta=-1):
    """Expresses the reduced Hamiltonian of MOs within `delta` hartrees of the HOMO/LUMO
    in the AO basis. If `delta` = -1, then the full Hamiltonian in the AO basis is returned;
    it is furthermore not split into an occupied and virtual Hamiltonian.
    *** ASSUMES ENERGIES ARE SORTED *** """

    N = M.shape[0]
    #_, M = read_MO_file(MO_file)
    #occ, virt = read_energies(orb_file)
    occ = energy_lvls[:int(N/2)]
    virt = energy_lvls[int(N/2):]

    for orbs in [occ,virt]:
        sorted_indices = np.argsort(orbs)

        if not np.all(sorted_indices == np.arange(N/2)):
            print('Energies unsorted in orb_file!')
        
    if delta > 0:
        E_homo = occ[-1]
        E_lumo = virt[0]

        relevant_occ_inds = (occ >= E_homo - delta).nonzero()[0]
        relevant_virt_inds = (virt <= E_lumo + delta).nonzero()[0] 

        print('Number of occupied MOs in reduced hamiltonian = ',relevant_occ_inds.shape)
        print('Number of virtual MOs in reduced hamiltonian = ',relevant_virt_inds.shape)
        
        occ_levels = occ[relevant_occ_inds]
        virt_levels = virt[relevant_virt_inds]

        D_occ = np.zeros((N,N))
        D_occ[relevant_occ_inds,relevant_occ_inds] = occ_levels
        print('D_occ:\n')
        print(D_occ)
        print('\n')

        D_virt = np.zeros((N,N))
        D_virt[relevant_virt_inds+(N//2),relevant_virt_inds+(N//2)] = virt_levels
        print('D_virt:\n')
        print(D_virt)
        print('\n')
        
        AO_hamiltonian_occ = M @ D_occ @ (M.T)
        AO_hamiltonian_virt = M @ D_virt @ (M.T)

        return AO_hamiltonian_occ, AO_hamiltonian_virt
    
    else: #delta = -1 ==> return full Hamiltonian in AO basis
        D = np.diag(energy_lvls)
        AO_hamiltonian = M @ D @ (M.T)
        return AO_hamiltonian


def greens_function_dense(e,Hao,gamL,gamR):

    Heff = Hao - (gamL + gamR)*(1j/2.0)
    E = np.eye(Heff.shape[0])*e

    G = scipy.linalg.inv(E - Heff)

    return G

def MCOs(Hao,gammaL,gammaR,sort=False,return_evals=False):
    
    Heff = Hao - (gammaL + gammaR) *(1j/2.0) #effective Hamiltonian for open system
    Heff_dagger = np.conj(Heff.T)

    if not sort:
        _, MCOs = scipy.linalg.eig(Heff)
        _, MCOs_bar = scipy.linalg.eig(Heff_dagger)

        return MCOs, MCOs_bar

    #sort eigenvectors in order of the real part of their corresponding eigenvalue
    else: 
        lambdas, MCOs = scipy.linalg.eig(Heff)
        lambdas2, MCOs_bar = scipy.linalg.eig(Heff_dagger)

        inds1 = np.argsort(np.real(lambdas))
        inds2 = np.argsort(np.real(lambdas2))

        MCOs = MCOs[:,inds1]
        lambdas = lambdas[inds1]

        MCOs_bar = MCOs_bar[:,inds2]
        lambdas2 = lambdas2[inds2]


        if return_evals:
            return lambdas, MCOs, lambdas2, MCOs_bar

        else:
            return MCOs, MCOs_bar

def MCOs_inv(Hao, gamL,gamR):

    Heff = Hao - (gamL+gamR)*(1j/2.0)

    zs, P = scipy.linalg.eig(Heff)

    # sort eigenvalues (and their corresponding eigenvectors using their real parts
    inds = np.argsort(np.real(zs))
    zs = zs[inds]
    P = P[:,inds]

    Pbar = scipy.linalg.inv(P).T
    zsbar = np.conj(zs)

    return zs, P, zsbar, Pbar


def inverse_participation_ratios(MO_matrix):

    return np.sum(np.abs(MO_matrix)**4, axis = 0)


def AO_gammas(pos, gamma, edge_tol=3.0, return_separate=True, graphene=False):
    """Returns the coupling matrices gamma_L and gamma_R represented in AO space (i.e. diagonal matrices).
    Edge atoms are detected using the concave hull algorithm.
    
    Parameters
    ----------
    pos: `numpy.ndarray`, shape=(N,2) or (N,3)
        Cartesian coordinates of the atoms in the conductor.
    M: `numpy.ndarray`, shape=(N,N)
        MO matrix of the conductor, represented in AO space. Elements of the MO matrix are given by:
        M_ij = <AO_i|MO_j>.
    gamma: `float`
        Coupling parameter (in eV). NOTE: We assume each edge atom is equally coupled to the leads.
    edge_tol: `float`
        Maximum allowed distance from left-/rightmost atom in the conductor for an edge atom to be considered
        of the structures left/right edge.
    return_separate: `bool` 
        If `True`, this function will return the coupling matrix to the left (gamma_L) and right (gamma_R) 
        leads separately. If it is set to `False`, gamma_L + gamma_R is returned.
    graphene: `bool`
        If `True`, the edge_atoms are identified by hand. Only use for graphene.
    
    Outputs
    ------
    gammaL, gammaR: `numpy.ndarray`, shape=(N,N)
        Matrices encoding the conductor's coupling to the left and right leads, respectively.
        If `return_separate` is set to `False`, gammaL+gammaR is returned."""

    if pos.shape[1] == 3:
        pos = pos[:,:2] #keep only x and y coords
    
    # for graphene; the edge is defined at the 
    if graphene: 
        sorted_xs = np.unique(pos[:,0])
        right_bools = (pos[:,0] == sorted_xs[-2]) + (pos[:,0] == sorted_xs[-1]) 
        right_inds = right_bools.nonzero()[0]

        left_bools = (pos[:,0] == sorted_xs[0]) + (pos[:,0] == sorted_xs[1]) 
        left_inds = left_bools.nonzero()[0]
    
    # if the edge is nontrivial, find it using the concave hull algorithm
    else: 
        edge_bois = concave_hull(pos,3)
        xmin = np.min(pos[:,0])
        xmax = np.max(pos[:,0])
        right_edge = edge_bois[edge_bois[:,0] > xmax - edge_tol]
        left_edge = edge_bois[edge_bois[:,0] < xmin + edge_tol]

        right_inds = np.zeros(right_edge.shape[0],dtype=int)
        left_inds = np.zeros(left_edge.shape[0],dtype=int)
        
        for k, r in enumerate(right_edge):
            right_inds[k] = np.all(pos == r, axis=1).nonzero()[0]

        for k, r in enumerate(left_edge):
            left_inds[k] = np.all(pos == r, axis=1).nonzero()[0]
    

    N = pos.shape[0]
    gammaR = np.zeros((N,N),dtype=float)
    gammaL = np.zeros((N,N),dtype=float)

    gammaR[right_inds,right_inds] = gamma
    gammaL[left_inds,left_inds] = gamma

    if return_separate:
        return gammaL, gammaR
    else:
        return gammaL + gammaR


def MO_gammas(M,gamL,gamR,return_diag=False,return_separate=True):
    """Transforms the coupling (broadening) matrices from their AO representation to their MCO
    representations."""


    Mdagger = M.conj().T
    gamL_MO = np.linalg.multi_dot((Mdagger,gamL,M))
    gamR_MO = np.linalg.multi_dot((Mdagger,gamR,M))

    if return_diag:
        couplingsR = np.diag(gamR_MO)
        couplingsL = np.diag(gamL_MO)

        if return_separate:
            return couplingsL, couplingsR
        else:
            return couplingsL + couplingsR 
    else:
        if return_separate:
            return gamL_MO, gamR_MO
        else:
            return gamL_MO + gamR_MO


def MCO_gammas(P,Pbar,gamL,gamR,return_diag=False,return_separate=True):
    """Transforms the coupling (broadening) matrices from their AO representation to their MCO
    representations."""

    Pbar_dagger = np.conj(Pbar.T)

    gamL_MCO = np.linalg.multi_dot((Pbar_dagger,gamL,P))
    gamR_MCO = np.linalg.multi_dot((Pbar_dagger,gamR,P))

    if return_diag:
        couplingsR = np.diag(gamR_MCO)
        couplingsL = np.diag(gamL_MCO)

        if return_separate:
            return couplingsL, couplingsR
        else:
            return couplingsL + couplingsR 
    else:
        if return_separate:
            return gamL_MCO, gamR_MCO
        else:
            return gamL_MCO + gamR_MCO


def interference_matrix_MCO_evals(e, Hao, gamL, gamR, diag_gf=False, dense_gf=True):

    #Hao = AO_hamiltonian(M, energy_lvls)
    N = Hao.shape[0]

    if diag_gf:

        if dense_gf:
            G = greens_function_dense(e, Hao, gamL, gamR)
        else:
            edgeL = np.diag(gamL).nonzero()[0]
            edgeR = np.diag(gamR).nonzero()[0]
            gamma = np.max(gamL)
            
            G = greens_function(e, Hao, gamma, edgeL, edgeR)
        
        evals, P = scipy.linalg.eig(G)
        
        #sort eigenvalues in order of their real parts
        inds = np.argsort(np.real(1/evals))
        evals = evals[inds]
        P = P[:,inds]

        evals_bar, Pbar = scipy.linalg.eig(np.conj(G.T))

        #sort eigenvalues in order of their real parts
        inds = np.argsort(np.real(1/evals_bar))
        evals_bar = evals_bar[inds]
        Pbar = Pbar[:,inds]

        gammaL = np.linalg.multi_dot((np.conj(P.T), gamL, P))
        gammaR = np.linalg.multi_dot((np.conj(Pbar.T), gamR, Pbar))

        A = gammaL * evals.reshape(N,1)
        B = gammaR * evals_bar.reshape(N,1)
            

    else:
        zs, P, zbars, Pbar = MCOs(Hao, gamL, gamR, sort=True, return_evals=True)
        #zs, P, zbars, Pbar = MCOs_inv(Hao, gamL, gamR)

        eZ = e - zs.reshape(1,N)
        eZbar = e - zbars.reshape(1,N)

        gammaL = np.linalg.multi_dot((np.conj(P.T), gamL, P))
        gammaR = np.linalg.multi_dot((np.conj(Pbar.T), gamR, Pbar))

        A = gammaL / eZ
        B = gammaR / eZbar

    return A * (B.T)

def interference_matrix_MCO_matrix_product(e,Hao,gamL,gamR,diag_gf=False, dense_gf=True):
    
    #Hao = AO_hamiltonian(M, energy_lvls)

    if dense_gf:
        G = greens_function_dense(e, Hao, gamL, gamR)

    else:
        edgeL = np.diag(gamL).nonzero()[0]
        edgeR = np.diag(gamR).nonzero()[0]
        gamma = np.max(gamL)
        
        G = greens_function(e, Hao, gamma, edgeL, edgeR)

    if diag_gf:
        evals, P = scipy.linalg.eig(G)
        
        #sort eigenvalues in order of their real parts
        inds = np.argsort(np.real(1/evals))
        evals = evals[inds]
        P = P[:,inds]

        evals_bar, Pbar = scipy.linalg.eig(np.conj(G.T))

        #sort eigenvalues in order of their real parts
        inds = np.argsort(np.real(1/evals_bar))
        evals_bar = evals_bar[inds]
        Pbar = Pbar[:,inds]

    else:
        P, Pbar = MCOs(Hao, gamL, gamR, sort=True)

    A = np.linalg.multi_dot((np.conj(P.T), gamL, G, P))
    B = np.linalg.multi_dot((np.conj(Pbar.T), gamR, np.conj(G.T), Pbar))

    return A * (B.T)

def interference_matrix_MO(e,M,energy_lvls,gamL,gamR):

    d = np.diag(e - energy_lvls)
    
    Sigma = (gamL+gamR)*(-1j/2)

    G = scipy.linalg.inv((d - Sigma)) #Green's function

    Gdagger = np.conj(G.T) #Hermitian adjoint of G

    A = gamL @ G
    B = gamR @ Gdagger

    return A * (B.T)


def MO_rgyr(pos,MO_matrix,n,center_of_mass=None):

    psi = np.abs(MO_matrix[:,n])**2

    if np.all(center_of_mass) == None:
        com = psi @ pos

    else: #if center of mass has already been computed, do not recompute
        com = center_of_mass

    R_squared = (pos*pos).sum(axis=-1) #fast way to compute square length of all position vectors
    R_squared_avg = R_squared @ psi

    #return np.sqrt(R_squared_avg - (com @ com))
    return np.sqrt(R_squared_avg - (com*com).sum(1))


def MCO_com(pos, P, Pbar, n):

    Pbar_dagger = np.conj(Pbar).T
    psi = P[:,n]*Pbar_dagger[n,:]
    center_of_mass = psi @ pos

    return center_of_mass
    

def MCO_rgyr(pos,P,Pbar,n,center_of_mass=None):

    Pbar_dagger = np.conj(Pbar).T
    psi = P[:,n]*Pbar_dagger[n,:]

    if np.all(center_of_mass) == None:
        center_of_mass = psi @ pos

    else: #if center of mass has already been computed, do not recompute
        com = center_of_mass

    R_squared = (pos*pos).sum(axis=-1) #fast way to compute square length of all position vectors
    R_squared_avg = R_squared @ psi

    return np.sqrt(R_squared_avg - (com @ com))


def all_rgyrs(pos,MO_matrix,centers_of_mass=None):

    psis = np.abs(MO_matrix)**2

    if np.all(centers_of_mass) == None:
        coms = (psis.T) @ pos

    else: #if centers of mass have already been computed, do not recompute
        coms = centers_of_mass

    R_squared = (pos*pos).sum(-1)
    R_squared_avg = R_squared @ psis

    coms_squared = (coms*coms).sum(-1)

    return np.sqrt(R_squared_avg - coms_squared)


def all_rgyrs_MCO(pos,P,Pbar,centers_of_mass=None):
    """Potentially finished."""

    N = pos.shape[0]

    xs = pos[:,0]
    ys = pos[:,1]
    
    Pbar_dagger = np.conj(Pbar.T)

    if np.any(centers_of_mass) == None:
        xs_MCOs = np.diag(Pbar_dagger @ (P * xs.reshape(N,1)))
        ys_MCOs = np.diag(Pbar_dagger @ (P * ys.reshape(N,1)))

        coms = np.vstack((xs_MCOs, ys_MCOs)).T

    else: #if centers of mass have already been computed, do not recompute
        coms = centers_of_mass

    R_squared = (pos*pos).sum(-1) #square all positions and sum along last axis (i.e. each vector's coords)
    R_squared_avg = np.diag(Pbar_dagger @ (R_squared.reshape(N,1) * P))

    coms_squared = (coms*coms).sum(-1)

    return np.sqrt(R_squared_avg - coms_squared)


def plot_MO(pos,MO_matrix,n,dotsize=45.0,show_COM=False,show_rgyr=False):

    if pos.shape[1] == 3:
        pos = pos[:,:2]

    psi = np.abs(MO_matrix[:,n])**2

    rcParams['text.usetex'] = True
    rcParams['font.size'] = 16
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    #if plot_type == 'nanoribbon':
    #    #rcParams['figure.figsize'] = [30.259946/2,7/2]
    #    figsize = [12,11/2]
    #elif plot_type == 'square':
    #    figsize = [4,4]  
    #else:
    #    print('Invalid plot type. Using default square plot type.')
    #    figsize = [4,4]

    fig, ax1 = plt.subplots()
    #fig.set_size_inches(figsize,forward=True)

    ye = ax1.scatter(pos.T[0,:],pos.T[1,:],c=psi,s=dotsize,cmap='plasma')
    cbar = fig.colorbar(ye,ax=ax1,orientation='vertical')
    plt.suptitle('$|\langle\\varphi_n|\psi_{%d}\\rangle|^2$'%n)
    ax1.set_xlabel('$x$ [\AA]')
    ax1.set_ylabel('$y$ [\AA]')
    ax1.set_aspect('equal')
    if show_COM or show_rgyr:
        com = psi @ pos
        ax1.scatter(*com, s=dotsize+1,marker='*',c='r')
    if show_rgyr:
        rgyr = MO_rgyr(pos,MO_matrix,n,center_of_mass=com)
        loc_circle = plt.Circle(com, rgyr, fc='none', ec='r', ls='--', lw=1.0)
        ax1.add_patch(loc_circle)

    #line below turns off x and y ticks 
    #ax1.tick_params(axis='both',which='both',bottom=False,top=False,right=False, left=False)

    plt.show()


def plot_MCO(pos,P,Pbar,n,dotsize=45.0,show_COM=False,show_rgyr=False,plot_dual=False):

    if pos.shape[1] == 3:
        pos = pos[:,:2]

    if plot_dual:
        psi = np.abs(Pbar[:,n])**2
        plot_title = '$|\langle\\varphi_n|\\bar{\psi}_{%d}\\rangle|^2$'%n
    else:
        psi = np.abs(P[:,n])**2
        plot_title = '$|\langle\\varphi_n|\psi_{%d}\\rangle|^2$'%n

    rcParams['text.usetex'] = True
    rcParams['font.size'] = 16
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    fig, ax1 = plt.subplots()
    #fig.set_size_inches(figsize,forward=True)

    ye = ax1.scatter(pos.T[0,:],pos.T[1,:],c=psi,s=dotsize,cmap='plasma')
    cbar = fig.colorbar(ye,ax=ax1,orientation='vertical')
    plt.suptitle(plot_title)
    ax1.set_xlabel('$x$ [\AA]')
    ax1.set_ylabel('$y$ [\AA]')
    ax1.set_aspect('equal')
    if show_COM or show_rgyr:
        com = MCO_com(pos, P, Pbar, n)
        print(com)
        ax1.scatter(*com, s=dotsize+1,marker='*',c='r')
    if show_rgyr:
        rgyr = MCO_rgyr(pos,P,Pbar,n,center_of_mass=com)
        loc_circle = plt.Circle(com, rgyr, fc='none', ec='r', ls='--', lw=1.0)
        ax1.add_patch(loc_circle)

    plt.show()

    

# ******* MAIN *******

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    eV2Ha = 0.0367493 #eV to Ha conversion factor

    qcffpi_datadir = '../../simulation_outputs/qcffpi_data'
    mo_datadir = os.path.join(qcffpi_datadir,'MO_coefs')
    orb_datadir = os.path.join(qcffpi_datadir,'orbital_energies')

    L = 102

    #MOfile = os.path.join(mo_datadir,'MOs_pCNN_MAC_%dx%d.dat'%(L,L))
    #orbfile = os.path.join(orb_datadir,'orb_energy_pCNN_MAC_%dx%d.dat'%(L,L))

    MOfile = os.path.join(mo_datadir,'MOs_kMC_MAC_clean.dat')
    orbfile = os.path.join(orb_datadir,'orb_energy_kMC_MAC_clean.dat')

    energy_window = 1 #eV
    H_occ, H_virt = AO_hamiltonian(MOfile,orbfile,energy_window*eV2Ha)

    print(H_occ.shape)
    print(H_virt.shape)

    Jocc, Jvirt = read_energies(orbfile)
    Jall = np.vstack((Jocc,Jvirt))

    Ehomo = Jocc[-1,1]
    print('HOMO energy = %f Ha'%Ehomo)
    Elumo = Jvirt[0,1]
    print('LUMO energy = %f Ha'%Elumo)

    plt.plot(*Jall.T,'ro',ms=10)
    #plt.axhline(Ehomo,'k--',lw=0.8)
    #plt.axhline(Elumo,'k--',lw=0.8)
    plt.show()


    plt.imshow(np.abs(H_occ))
    plt.colorbar()
    plt.suptitle('HOMO-27:HOMO')
    plt.show()

    plt.imshow(np.abs(H_virt))
    plt.colorbar()
    plt.suptitle('LUMO:LUMO+35')
    plt.show()

    H = AO_hamiltonian(MOfile,orbfile,-1)
    plt.imshow(np.abs(H))
    plt.suptitle('Full Hamiltonian in AO_basis')
    plt.colorbar()
    plt.show()

    pos, _ = read_MO_file(MOfile)
    total_couplings = np.sum(H,axis=0)

    fig, ax = plt.subplots()
    ye = ax.scatter(*pos.T[:2],c=np.abs(total_couplings),s=10.0,cmap='plasma')
    cbar = fig.colorbar(ye,ax=ax)
    ax.set_aspect('equal')
    plt.show()

    np.save('H_occ.npy',H_occ)
    np.save('H_virt.npy',H_virt)
    np.save('H_full.npy',H)

    #check off-diagonal elements

    off_diagonal_elements = H[~np.eye(H.shape[0],dtype=bool)]
    print(off_diagonal_elements.shape)
    print('Average coupling: ', np.mean(off_diagonal_elements))
    print('Coupling standard deviation: ', np.std(off_diagonal_elements))

    fig, ax = plt.subplots(1,1)

    hist1, bins1 = np.histogram(off_diagonal_elements,300)
    width1 = bins1[1] - bins1[0]
    center1 = (bins1[1:] + bins1[:-1])/2
    ax.bar(center1,hist1,align='center',width=width1)
    ax.set_title('Couplings')
    plt.show()

    large_coupling_inds = (center1 >= -0.12)*(center1 <= -0.06)
    print(np.sum(hist1[large_coupling_inds]))
