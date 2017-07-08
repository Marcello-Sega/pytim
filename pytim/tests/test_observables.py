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

    >>> # CORRELATOR TEST
    >>> from pytim.observables import correlate
    >>> a = np.array([1.,0.,1.,0.,1.])
    >>> b = np.array([0.,2.,0.,1.,0.])
    >>> corr = correlate(b,a)
    >>> ['{:.2f}'.format(i) for i in corr]
    ['-0.00', '0.75', '0.00', '0.75', '0.00']


    >>> corr = correlate(b)
    >>> ['{:.2f}'.format(i) for i in corr]
    ['1.00', '0.00', '0.67', '-0.00', '0.00']

    >>> # PROFILE EXTENDED TEST: checks trajectory averaging 
    >>> # and consistency in summing up layers  contributions
    >>> import numpy as np
    >>> import MDAnalysis as mda
    >>> import pytim
    >>> from   pytim.datafiles import *
    >>> from   pytim.observables import Profile
    >>> u = mda.Universe(WATER_GRO,WATER_XTC)
    >>> g=u.select_atoms('name OW')
    >>> inter = pytim.ITIM(u,group=g,max_layers=4,centered=True)
    >>> 
    >>> Layers=[]
    >>> AIL = inter.atoms.in_layers
    >>> 
    >>> Layers.append(Profile(u.atoms))
    >>> for n in np.arange(4):
    ...     Layers.append(Profile(AIL[::,n]))
    >>> Val=[]
    >>> for ts in u.trajectory[:4]:
    ...     for L in Layers:
    ...         L.sample()
    >>> for L in Layers:
    ...     Val.append(L.get_values(binwidth=2.0)[2])
    >>> 

    >>> print Val[0].sum()
    2.432

    >>> # the sum of the layers' contribution is expected to add up only close
    >>> # to the surface
    >>> print not np.sum(np.abs(np.sum(Val[1:],axis=0)[47:] - Val[0][47:])>1e-15)
    True


    """

    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
