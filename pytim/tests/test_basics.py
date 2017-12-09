# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: test_basics
    ===================
"""


class TestBasics():

    """
    This is a collection of basic tests to check
    that code is running -- no test on the correctness
    of the output is performed here.


    >>> # TEST:1 basic functionality
    >>> import MDAnalysis as mda
    >>> import pytim
    >>> from pytim import  *
    >>> from pytim.datafiles import *
    >>> u         = mda.Universe(WATER_GRO)
    >>> oxygens   = u.select_atoms("name OW")
    >>> interface = pytim.ITIM(u, alpha=1.5, max_layers=4)
    >>> print len(interface.layers[0,0])
    786
    >>> del interface
    >>> interface = pytim.ITIM(u, alpha=1.5, max_layers=4, multiproc=False)
    >>> print len(interface.layers[0,0])
    786
    >>> del interface

    >>> # TEST:2 basic functionality
    >>> u=None
    >>> interface = pytim.GITIM(u)
    Traceback (most recent call last):
        ...
    Exception: Wrong Universe passed to ITIM class


    >>> interface = pytim.ITIM(u)
    Traceback (most recent call last):
        ...
    Exception: Wrong Universe passed to ITIM class

    >>> # TEST:3 large probe sphere radius
    >>> u         = mda.Universe(WATER_GRO)
    >>> interface = pytim.ITIM(u, alpha=100000.0, max_layers=1,multiproc=False)
    Traceback (most recent call last):
        ...
    ValueError: parameter alpha must be smaller than the smaller box side

    >>> # TEST:4 interchangeability of Universe/AtomGroup
    >>> u         = mda.Universe(WATER_GRO)
    >>> oxygens   = u.select_atoms("name OW")
    >>> interface = pytim.ITIM(u, alpha=1.5,group=oxygens, max_layers=1,multiproc=False,molecular=False)
    >>> print len(interface.layers[0,0])
    262
    >>> interface = pytim.ITIM(oxygens, alpha=1.5,max_layers=1,multiproc=False,molecular=False)
    >>> print len(interface.layers[0,0])
    262


    >>> # PDB FILE FORMAT
    >>> import MDAnalysis as mda
    >>> import pytim
    >>> from pytim.datafiles import WATER_GRO
    >>> u         = mda.Universe(WATER_GRO)
    >>> oxygens   = u.select_atoms("name OW")
    >>> interface = pytim.ITIM(u, alpha=1.5, max_layers=4,molecular=True)
    >>> interface.writepdb('test.pdb',centered=False)
    >>> PDB =open('test.pdb','r').readlines()
    >>> line = filter(lambda l: 'ATOM     19 ' in l, PDB)[0]
    >>> beta = line[62:66] # PDB file format is fixed
    >>> print beta
    4.00


    >>> # mdtraj 
    >>> try:
    ...     import mdtraj
    ...     try:
    ...         import numpy as np 
    ...         import MDAnalysis as mda
    ...         import pytim
    ...         from pytim.datafiles import WATER_GRO,WATER_XTC
    ...         from pytim.datafiles import pytim_data,G43A1_TOP
    ...         # MDAnalysis 
    ...         u = mda.Universe(WATER_GRO,WATER_XTC)
    ...         ref = pytim.ITIM(u)
    ...         # mdtraj
    ...         t = mdtraj.load_xtc(WATER_XTC,top=WATER_GRO)
    ...         # mdtraj manipulates the name of atoms, we need to set the 
    ...         # radii by hand
    ...         _dict = { 'O':pytim_data.vdwradii(G43A1_TOP)['OW'],'H':0.0}
    ...         inter = pytim.ITIM(t, radii_dict=_dict)
    ...         ids_mda = []
    ...         ids_mdtraj = []
    ...         for ts in u.trajectory[0:2]:
    ...             ids_mda.append(ref.atoms.ids)
    ...         for ts in t[0:2]:
    ...             ids_mdtraj.append(inter.atoms.ids)
    ...         for fr in [0,1]:   
    ...             if not np.all(ids_mda[fr] == ids_mdtraj[fr]):
    ...                 print "MDAnalysis and mdtraj surface atoms do not coincide"
    ...         _a = u.trajectory[1] # we make sure we load the second frame
    ...         _b = t[1]
    ...         if not np.all(np.isclose(inter.atoms.positions[0], ref.atoms.positions[0])):
    ...             print "MDAnalysis and mdtraj atomic positions do not coincide"
    ...     except:
    ...         raise RuntimeError("mdtraj available, but a general exception happened")
    ... except:
    ...     pass


    >>> # check that using the biggest_cluster_only option without setting cluster_cut
    >>> # throws a warning and resets to biggest_cluster_only == False
    >>> import MDAnalysis as mda
    >>> import pytim
    >>> from   pytim.datafiles import GLUCOSE_PDB
    >>>
    >>> u       = mda.Universe(GLUCOSE_PDB)
    >>> solvent = u.select_atoms('name OW')
    >>> inter = pytim.GITIM(u, group=solvent, biggest_cluster_only=True)
    Warning: the option biggest_cluster_only has no effect without setting cluster_cut, ignoring it

    >>> print inter.biggest_cluster_only
    False
    """

    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
