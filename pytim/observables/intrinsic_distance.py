# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: IntrinsicDistance
    =========================
"""
from __future__ import print_function
from . import Observable


class IntrinsicDistance(Observable):
    """Initialize the intrinsic distance calculation.

    :param PYTIM interface: compute the intrinsic distance with respect
                            to this interface
    :param str symmetry: force calculation using this symmetry, if
                            availabe (e.g. 'generic', 'planar', 'spherical')
                            If 'default', uses the symmetry selected in
                            the PYTIM interface instance.

    Example:

    >>> import MDAnalysis as mda
    >>> import pytim
    >>> import numpy as np
    >>> from pytim import observables
    >>> from pytim.datafiles import MICELLE_PDB
    >>> u = mda.Universe(MICELLE_PDB)
    >>> micelle = u.select_atoms('resname DPC')
    >>> waterox = u.select_atoms('type O and resname SOL')
    >>> inter = pytim.GITIM(u,group=micelle, molecular=False, alpha=2.0)
    >>> dist  = observables.IntrinsicDistance(interface=inter)
    >>> d = dist.compute(waterox)
    >>> np.set_printoptions(precision=3,threshold=10)
    >>> print(d)
    [25.733  8.579  8.852 ... 18.566 13.709  9.876]

    >>> np.set_printoptions(precision=None,threshold=None)
    """

    def __init__(self, interface, symmetry='default', mode='default'):
        Observable.__init__(self, interface.universe)
        self.interface = interface
        self.mode = mode
        if symmetry == 'default':
            self.symmetry = self.interface.symmetry
        else:
            self.symmetry = symmetry

    def compute(self, inp, kargs=None):
        """Compute the intrinsic distance of a set of points from the first
        layers.

        :param ndarray positions: compute the intrinsic distance for this set
                                  of points
        """
        # see pytim/surface.py
        return self.interface._surfaces[0].distance(inp, self.symmetry, mode=self.mode)
