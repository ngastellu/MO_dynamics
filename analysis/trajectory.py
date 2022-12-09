from functools import partial
from os import path
import qcffpi_utils as qutils
from qchemMAC import AO_hamiltonian, AO_gammas, all_rgyrs

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
    '''
    def __init__(self, QCFFPI_dir, nstart, nend, step=1, MD_timestep):
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
        self.Natoms = -1 # this will get updated once the energies or the MOs are obtained


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

        efiles = [path.join(self.QCFFPIdir,f'frame-{f}','orb_energy.dat') for f in frames]

        return map(qutils.read_energies, efiles)
    
    def fetch_rMOs(self, frames=None):
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
            
        MOfiles = [path.join(self.QCFFPIdir, f'frame-{f}', 'MO_coefs.dat') for f in frames]

        N = qutils.get_Natoms(MOfiles[0])

        return map(partial(qutils.read_MO_file, Natoms=N), MOfiles)

    
    def MOtraj(mo_inds, frames=None):
        ''''''