# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: basic_observables
    =========================
"""
from __future__ import print_function
import numpy as np
from scipy import stats
from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup

from .observable import Observable


class Number(Observable):
    """The number of atoms.
    """

    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: one, for each atom in the group

        """
        return np.ones(len(inp))


class Mass(Observable):
    """Atomic masses"""

    def __init__(self, *arg, **kwarg):
        """
            >>> import pytim
            >>> import MDAnalysis as mda
            >>> from pytim.datafiles import WATERSMALL_GRO
            >>> from pytim.observables import Mass
            >>> u = mda.Universe(WATERSMALL_GRO)
            >>> obs = Mass()
            >>> print (obs.compute(u.atoms)[0])
            15.999

        """
        Observable.__init__(self, None)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: an array of masses for each atom in the group

        """
        return inp.masses


class Charge(Observable):
    """Atomic charges"""

    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: an array of charges for each atom in the group

        """
        try:
            return inp.charges
        except AttributeError:
            print("Error, the passed Atomgroup has no charges attribute")


class Vector(Observable):
    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)
        self.select_direction(arg)


class Position(Vector):
    """Atomic positions"""

    def __init__(self, *arg, **kwarg):
        Vector.__init__(self, None)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic positions

        """
        inp = self._to_atomgroup(inp)
        return inp.positions[:, self.dirmask]


class Velocity(Vector):
    """Atomic velocities"""

    def __init__(self, *arg, **kwarg):
        Vector.__init__(self, *arg)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic velocities

        """

        inp = self._to_atomgroup(inp)
        return inp.velocities[:, self.dirmask]


class Force(Vector):
    """Atomic forces"""

    def __init__(self, *arg, **kwarg):
        Vector.__init__(self, None)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic forces

        """
        inp = self._to_atomgroup(inp)
        return inp.forces[:, self.dirmask]


class NumberOfResidues(Observable):
    """The number of residues.

    Instead of associating 1 to the center of mass of the residue, we
    associate 1/(number of atoms in residue) to each atom. In an
    homogeneous system, these two definitions are (on average)
    equivalent. If the system is not homogeneous, this is not true
    anymore.

    """

    def __init__(self, *arg, **karg):
        Observable.__init__(self, None)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: one, for each residue in the group

        """
        residue_names = np.unique(inp.atoms.resnames)
        tmp = np.zeros(len(inp))

        for name in residue_names:
            atom_id = np.where(inp.resnames == name)[0][0]
            tmp[inp.resnames == name] = 1. / len(inp[atom_id].residue.atoms)

        return tmp
