from functools import partial
from itertools import accumulate, cycle, starmap
from os import path
import numpy as np
import qcffpi_utils as qutils
import qchemMAC as qcM


def MOGammas(pos, M, gamma=0.1, n=None, rotate90=True, edge_tol=3.0):
    '''Computes the MO-lead couplings. If `n` is specified, only the MOs indexed by n are
    considered. This function is very similar to `MO_gammas` in `qchemMAC.py` however, there are
    some important differences:
        * this function can alternatively compute the couplings along the top/bottom edges of the MAC structure if `rotate90` is set to `True`.
        * whereas `qchemMAC.MO_gammas` returns the full matrices (by default), this function only
        returns the diagonal elements.
        * the arguments accepted by both functions differ; the code for this function rather
        transparently displays how it is related to `qchemMAC.MO_gammas`.
    This function was written for more convenient use by the `Trajectory` class.
    '''

    if rotate90:
        rot = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        pos = (rot @ pos.T).T
    gamAO = qcM.AO_gammas(pos, gamma, return_separate=False, edge_tol=edge_tol)
    Mdagger = M.conj().T
    if n:
        return np.diag(np.multidot(Mdagger, gamAO, M))[n]
    else:
        return np.diag(np.multidot(Mdagger, gamAO, M))

    




class Trajectory:
    '''
    Class for easily obtaining and processing the QCFFPI data of the frames from a LAMMPS MD trajectory of MAC.

    To initialise it needs:
    
    QCFFPI_dir: `str`
        String containing path to the diretory of the QCFFPI data. If no QCFFPI calculation has
        been run on the frames of interest prior to using this class, then this is the directory to
        which the QCFFPI outputs will be written.
    
    nstart: `int`
        *Index* of the first frame of interest.
    
    nend: `int`
        *Index* of the last frame of interest (inclusive).
    
    step: `int`
        Spacing between indices of consecutive sampled frames.

    The main idea behind this class is to avoid loading all of the QCFFPI data in memory at once by 
    using generators to store each frame's energy and atomic positions/MO matrix. These generators 
    are obtained using the `fetch_energies` and `fetch_rMOs` functions, respectively.
    
    All other methods then use these generators to compute other quantum mechanical properties of
    the MAC structure for different frames of the MD trajectory.
    '''
    def __init__(self, QCFFPI_dir, nstart, nend, MD_timestep, step=1,ddprefix='frame-'):
        if path.isdir(QCFFPI_dir):
            self.QCFFPIdir = QCFFPI_dir
        else:
            raise FileNotFoundError(f"QCFFPI directory {QCFFPI_dir} not found. Cannot intialise trajectory")
        
        if nstart > nend:
            raise ValueError(f'Indexing of intial frame ({nstart}) should not be smaller than index of final frame ({nend}).')
        else:
            self.nstart = nstart
            self.nend = nend
        if step < 1:
            raise ValueError('Step size must be a positive integer')
        self.step = step
        self.frame_inds = range(nstart,nend,step)
        self.dt = MD_timestep
        self.ddprefix = ddprefix #prefix of subdirectories containing QCFFPI data; suffix is just 
                                 #the frame index
        
        self.Natoms = -1 # this will get updated once the energies or the MOs are obtained
        self.t = np.linspace(nstart*self.dt, nend*self.dt, step*self.dt)

        self.rMOs = None


    def fetch_energies(self,frames=None):
        '''
        This function loads the MO energies of the selcted frames into a *generator*.
        Parameters
        ----------
        frames: iterable containing `ints`
            List of frame indices for which we wish to extract QCFFPI data.
            The inds `self.frame_inds` are used by default.
            
        Returns
        -------
        energies: `generator` of `np.ndarray`s
            Each NumPy array contains the MO energies of a frame in `frames`.
        '''
        
        if frames == None:
            frames = self.frame_inds

        efiles = (path.join(self.QCFFPIdir,f'{self.ddprefix}{f}','orb_energy.dat') for f in frames)

        return map(qutils.read_energies, efiles)
    
    def fetch_rMOs(self, frames=None, MO_inds=None, cycle_save=False):
        '''
        This function loads the MO energies of the selcted frames into a *generator*.
        Parameters
        ----------
        frames: iterable containing `ints`
            List of frame indices for which we wish to extract QCFFPI data.
            The inds `self.frame_inds` are used by default.
            
        Returns
        -------
        energies: `generator` of `tuple`s of `np.ndarrays`s
            Each tuple contains: 
                * R: a (N,3) NumPy array containg the positions of the atoms in a frame.
                * M: a (N,N) NumPy array whose columns contain the MOs of a frame, represented in
                     AO space.
        '''
        
        if frames == None:
            frames = self.frame_inds
            
        MOfiles = (path.join(self.QCFFPIdir, f'{self.ddprefix}{f}', 'MO_coefs.dat') for f in frames)

        if self.Natoms == -1:
            N = qutils.get_Natoms(path.join(self.QCFFPIdir, f'{self.ddprefix}{frames[0]}', 'MO_coefs.dat'))
            self.Natoms = N
        else:
            N = self.Natoms

        # If rMOs will be re-used, bind it to `self`
        # cycle() allows one to iterate over rMOs multiple times without having to rebuild it
        if cycle_save: 
            self.rMOs = cycle(map(partial(qutils.read_MO_file, Natoms=N), MOfiles))
        
        else:
            return map(partial(qutils.read_MO_file, Natoms=N, MO_inds=MO_inds), MOfiles)

    
    def avg_energies(self, frames=None):
        '''Computes the MO energies averaged over the sampled frames indexed by `frames`.'''
        if frames:
            nsteps = len(frames)
        else:
            nsteps = self.t.size
        energies = self.fetch_energies(frames)
        return accumulate(energies)
        
    
    def MOtraj(self, MO_inds=None, frames=None, use_rMOs=True):
        '''Get trajectory of the center of masses (COMs) of MOs indexed by `MO_inds`, sampled at 
        times indexed by `frames`.'''
        
        if self.rMOs and use_rMOs:
            if frames:
                print('[WARNING - Trajectory.MOtraj] Using self.rMOs to construct MO trajectories.\
                Ignoring `frames` argument. ')
            return starmap(partial(qcM.MO_com, n=MO_inds),self.rMOs)
        else:
            rMOs = self.fetch_rMOs(frames)
            return starmap(partial(qcM.MO_com, n=MO_inds), rMOs)
        
    def approx_gammas(self, frames=None, MO_inds=None, use_rMOs=True, rotate90=True):
        '''Get time-dependence of the MO-lead couplings for the selected MOs and MD frames.
        Argument `rotate90` is set to `True` by default because we assume the leads are coupled to
        the top/bottom edges of the structure. For couplings to the left/right edges '''
        if frames == None:
            frames = self.frame_inds

        if self.rMOs and use_rMOs:
            if frames:
                print('[WARNING - Trajectory.approx_gammas] Using self.rMOs to compute approximate\
                couplings.\n Ignoring `frames` argument.')
            rMOs = self.rMOs    
        else:
            rMOs = self.fetch_rMOs(frames,MO_inds)
        return starmap(partial(MOGammas, n=MO_inds, rotate90=rotate90), rMOs)
    
    def avg_approx_gammas(self, frames=None, MO_inds=None, use_rMOs=True, rotate90=True):
        if frames:
            nsteps = len(frames)
        else:
            nsteps = self.t.size 
        gammas = self.approx_gammas(frames, MO_inds, use_rMOs, rotate90)
        return accumulate(gammas)/nsteps

    def fetch_MCOs(self, MO_inds=None, frames=None, use_rMOs=True, rotate90=True):
        '''
        Gets the MCOs and associated eigenvalues (sorted with increasing real part) of the selected
        frames.
        '''
        if frames == None:
            frames = self.frame_inds
        
        if use_rMOs:
            rMOs = self.rMOs
        else:
            rMOs = self.fetch_rMOs(frames, MO_inds)

    
