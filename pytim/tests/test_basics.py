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
    >>> interface = pytim.ITIM(u, alpha=2.0, max_layers=4)
    >>> print len(interface.layers[0,0])
    780
    >>> del interface
    >>> interface = pytim.ITIM(u, alpha=2.0, max_layers=4, multiproc=False)
    >>> print len(interface.layers[0,0])
    780
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

    >>> # PDB FILE FORMAT
    >>> import MDAnalysis as mda
    >>> import pytim
    >>> from pytim.datafiles import *
    >>> u         = mda.Universe(WATER_GRO)
    >>> oxygens   = u.select_atoms("name OW")
    >>> interface = pytim.ITIM(u, alpha=2.0, max_layers=4,molecular=True)
    >>> interface.writepdb('test.pdb',centered=False)
    >>> PDB =open('test.pdb','r').readlines()
    >>> line = filter(lambda l: 'ATOM     19 ' in l, PDB)[0]
    >>> beta = line[62:66] # PDB file format is fixed
    >>> print beta
    4.00


    """

    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
