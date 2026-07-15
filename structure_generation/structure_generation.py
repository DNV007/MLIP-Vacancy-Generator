import copy # for generate_rattled_atoms_list

def generate_rattled_atoms_list(atoms, seed=42, iterations=10, stdev=0.001):
    '''
    Generates a list of rattled atoms objects from a parent atoms objects, 
    the rattled objects will be reused to generate a new object.

    Parameters
    ----------
    atoms : atoms
        Input structure.
    seed : INT, optional
        Seed for the structure rattling. The default is 42.
    iterations : INT, optional
        The amount of iterations to rattle for. The default is 10.
    stdev : FLOAT, optional
        stdev of the rattle. The default is 0.001.

    Returns
    -------
    TYPE
        DESCRIPTION.
    '''
    rattled_atoms_list = []
    rattled_atoms = atoms.copy()
    rattled_atoms.rattle(seed=seed)
    rattled_atoms_list.append(rattled_atoms)
    for i in range(iterations-1):
        rattled_atoms = rattled_atoms.copy()
        rattled_atoms.rattle(seed=seed, stdev=stdev)
        rattled_atoms_list.append(rattled_atoms)
        pass    
    return rattled_atoms_list

