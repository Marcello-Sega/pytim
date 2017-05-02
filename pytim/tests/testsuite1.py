# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: testsuite1
    ==================
"""

class Testsuite1():

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
    >>> del interface

    >>> # TEST:2 large probe sphere radius
    >>> interface = pytim.ITIM(u, alpha=100000.0, max_layers=1,multiproc=False)
    Traceback (most recent call last):
        ...
    AssertionError: parameter alpha must be smaller than the smaller box side


    >>> # OBSERVABLES TEST: 1
    >>> u = mda.Universe(_TEST_ORIENTATION_GRO)
    >>> o = observables.MolecularOrientation(u)
    >>> print(o.compute(u.atoms).flatten())
    [ 1.          0.          0.          0.          1.          0.          0.
     -0.70710677 -0.70710677]

    >>> # OBSERVABLES TEST: 2
    >>> u=mda.Universe(_TEST_PROFILE_GRO)
    >>> o=observables.Number()
    >>> p=observables.Profile(u.atoms,direction='x',observable=o)
    >>> p.sample()
    >>> low,up,avg =  p.get_values(binwidth=1.0)
    >>> print(low[0:3])
    [ 0.  1.  2.]
    >>> print(avg[0:3])
    [ 0.01  0.02  0.03]


    """

    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
