# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: test_observables
    ========================
"""

class TestObservables():

    """
    This is a collection of basic tests to check
    that the observables are yelding the expected
    result.

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
