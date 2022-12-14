import numpy as np


# Set of functions useful to extract data from QCFFPI calculations of MAC structures.

def get_Natoms(infile):
    """Returns the number of atoms contained in a MAC structure whose MO coefficients
       are stored in `infile`."""
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


def read_MO_file(infile, Natoms=None, MO_inds=None):
    """Reads MO coefs output file from QCFFPI and returns a list of atomic positions and a AO -> MO
     transformation matrix with elements M_ij = <AO_i|MO_j>. If `MO_inds` is specified, then only
     the columns indexed by `MO_inds` (corresponding to specific MOs) are returned, rather thanthe
     full MO matrix."""
    
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

    if MO_inds:
        return positions, MO_matrix[:,MO_inds]
    else:
        return positions, MO_matrix


def read_energies(orb_file,Natoms=-1):
    """Reads energies from QCCFPI output file `orb_file` and returns them in an array.
    
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

    return np.array(list(map(float,[l.split()[1] for l in lines[:nlines_to_read]])))

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
