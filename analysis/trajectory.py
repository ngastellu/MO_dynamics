from os import path
from MACmol import MACMolecule

class Trajectory:
    '''Class for easily obtaining and processing the QCFFPI data of the frames from a LAMMPS MD trajectory of MAC.

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
    def __init__(self, QCFFPI_dir, nstart, nend, step=1):
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
    
    def fetch_QCFFPI_data(self,frames=None):
        pass
        

