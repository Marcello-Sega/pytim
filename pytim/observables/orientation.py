# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: Orientation
    ===================
"""

from __future__ import print_function
from . import Observable
import numpy as np
from MDAnalysis.core.groups import Atom, AtomGroup


class Orientation(Observable):
    """Orientation of a group of points.

    :param str options: optional string. If `normal` is passed, the
                        orientation of the normal vectors is computed
                        If the option 'molecular' is passed at initialization
                        the coordinates of the second and third atoms are
                        folded around those of the first.

    """

    def __init__(self, universe, options=''):
        self.u = universe
        self.options = options

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param ndarray inp:  the input atom group. The length be a multiple
                             of three
        :returns: the orientation vectors

        For each triplet of positions A1,A2,A3, computes the unit vector
        between A2-A1 and  A3-A1 or, if the option 'normal' is passed at
        initialization, the unit vector normal to the plane spanned by the
        three vectors.


        """

        if isinstance(inp, AtomGroup) and len(inp) != 3 * len(inp.residues):
            inp = inp.residues
        if 'molecular' in self.options:
            pos = self.fold_around_first_atom_in_residue(inp)
        else:
            pos = inp.positions
        flat = pos.flatten()
        pos = flat.reshape(len(flat) // 3, 3)
        a = pos[1::3] - pos[0::3]
        b = pos[2::3] - pos[0::3]

        if 'normal' in self.options:
            v = np.cross(a, b)
        else:
            v = np.array(a + b)
        v = np.array([el / np.sqrt(np.dot(el, el)) for el in v])
        return v
