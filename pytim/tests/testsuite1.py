""" Module: testsuite1
    ==================
"""

class testsuite1():
    """
    This is a collection of basic tests to check
    that code is running -- no test on the correctness
    of the output is performed here.
    
    
    >>> # TEST:1 check that MDAnalysis can be imported
    >>> import MDAnalysis as mda
    
    >>> # TEST:2 chech that pytim can be imported
    >>> import pytim 
    
    >>> # TEST:3 check datafile module
    >>> from pytim.datafiles import *
    
    >>> # TEST:4 check integrity of WATER_GRO datafile
    >>> u         = mda.Universe(WATER_GRO)
    
    >>> # TEST:5 test initialization
    >>> oxygens   = u.select_atoms("name OW") 
    >>> interface = pytim.ITIM(u, alpha=2.0, max_layers=4)
    
    >>> # TEST:6 assign layer
    >>> interface.assign_layers()
    >>> del interface
     
    >>> # TEST:7 FAILS large probe sphere radius 
    >>> #interface = pytim.ITIM(u, alpha=100000.0, max_layers=1,multiproc=False)
    >>> #interface.assign_layers()

    """

    pass


