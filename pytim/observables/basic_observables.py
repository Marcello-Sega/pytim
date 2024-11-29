# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: basic_observables
    =========================
"""
from __future__ import print_function
import numpy as np
from scipy import stats
from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup
from MDAnalysis.lib import distances

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
        self.select_direction(arg[0])


class ReferenceFrame(Vector):
    """ Given a set of triplets of atoms or positions, return the
        corresponding set of reference frame units vectors, calculated
        using the modified Gram-Schmidt (orientation preserving) or
        using the faster QR decomposition (does not preserve orientation)
    """

    def __init__(self, QR=False, *arg, **kwarg):
        # TODO: does it need extension to 2- or 1- dimensional frames?
        Vector.__init__(self, 'xyz', kwarg)

    def compute(self, inp):
        try:
            pos = inp.atoms.positions
            pos = pos.flatten()
        except AttributeError:
            try:
                pos = inp.flatten()
            except AttributeError:
                raise ValueError('must pass an atom group or a ndarray')

        n = pos.shape[0]
        if np.mod(n, 9) > 0:  # 3 atoms x 3 coordinates
            raise ValueError('number of atoms in group not a multiple of 3')

        n = n // 3
        pos = pos.reshape(n, 3, 3)
        if self.QR is True:
            return np.asarray([np.linalg.qr(A)[0] for A in pos])
        else:
            return np.asarray([self._modified_GS(A) for A in pos])

    def _modified_GS(self, A):

        Q = np.zeros(A.shape)
        for i in np.arange(A.shape[1]):
            q = A[:, i]
            for j in range(i):
                q = q - np.dot(q, Q[:, j]) * Q[:, j]

            Q[:, i] = q / la.norm(q)
        return Q


class Distance(Observable):
    """ Distance between atoms in two groups. 

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> u = mda.Universe(pytim.datafiles.WATER_GRO)
        >>> dist = pytim.observables.Distance()
        >>> with np.printoptions(precision=3):
        ...     print(dist.compute(u.atoms[:2],u.atoms[:2]) )
        [0.    0.996 0.996 0.   ]

        The distance of the projections of the atoms position on 
        a plane can be computed by initializing the observable
        in the same way as the Position one, e.g.:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> u = mda.Universe(pytim.datafiles.WATER_GRO)
        >>> dist = pytim.observables.Distance('xy')
        >>> with np.printoptions(precision=3):
        ...     print(dist.compute(u.atoms[:2],u.atoms[:2]) )
        [0.    0.502 0.502 0.   ]


        Notice that the Distance observable is equivalent to using 
        RelativePosition passing a reference group and choosing to return
        the radial component of the spherical coordinates. The latter
        is, however, much slower than the former.

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> u = mda.Universe(pytim.datafiles.WATER_GRO)
        >>> d1 = pytim.observables.Distance().compute(u.atoms[:9],u.atoms[:9])
        >>> d2 = pytim.observables.RelativePosition(spherical=True).compute(u.atoms[:9],u.atoms[:9])[:,0]
        >>> all(np.isclose(d1,d2))
        True

        >>> d1 = pytim.observables.Distance('xy').compute(u.atoms[:9],u.atoms[:9])
        >>> d2 = pytim.observables.RelativePosition('xy',spherical=True).compute(u.atoms[:9],u.atoms[:9])[:,0]
        >>> all(np.isclose(d1,d2))
        True

    """

    def __init__(self, arg='xyz', **kwarg):
        Observable.__init__(self, None)
        self.select_direction(arg)

    def compute(self, inp, *args, **kwarg):
        # TODO generalise to set of points?
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic positions

        """
        inp = self._to_atomgroup(inp)
        inp2 = self._to_atomgroup(args[0])
        mask = np.asarray([0., 0., 0])
        mask[self.dirmask] = 1.0
        return distances.distance_array(
            inp.positions * mask,
            inp2.positions * mask,
            box=inp.universe.dimensions).ravel()


class Position(Vector):
    """ Atomic positions

        Example: compute the projection on the xz plane of the first water molecule

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> u = mda.Universe(pytim.datafiles.WATER_GRO)
        >>> proj = pytim.observables.Position('xz')
        >>> with np.printoptions(precision=3):
        ...     print(proj.compute(u.residues[0].atoms) )
        [[28.62 11.37]
         [28.42 10.51]
         [29.61 11.51]]

    """
    # TODO: add a flag to prevent recomputing the reference frame, in case
    # it's constant

    def __init__(self, arg='xyz', **kwarg):
        Vector.__init__(self, arg, kwarg)
        try:
            self.folded = kwarg['folded']
        except KeyError:
            self.folded = True
        try:
            self.cartesian = kwarg['cartesian']
        except KeyError:
            self.cartesian = True
        try:
            self.spherical = kwarg['spherical']
        except KeyError:
            self.spherical = False

        if self.spherical is True:
            self.cartesian = False

    def _cartesian_to_spherical(self, cartesian):
        spherical = np.empty(cartesian.shape)
        spherical[:, 0] = np.linalg.norm(cartesian, axis=1)
        if len(self.dirmask) > 1:  # 2d or 3d
            spherical[:, 1] = np.arctan2(cartesian[:, 1], cartesian[:, 0])
        if len(self.dirmask) == 3:  # 3d
            xy = np.sum(cartesian[:, [0, 1]]**2, axis=1)
            spherical[:, 2] = np.arctan2(np.sqrt(xy), cartesian[:, 2])

        return spherical

    def compute(self, inp, **kwarg):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic positions

        """
        inp = self._to_atomgroup(inp)
        box = inp.universe.dimensions[:3]
        if self.folded:
            cartesian = inp.atoms.pack_into_box()
           # return cartesian[:,self.dirmask]
        else:
            cartesian = inp.positions

        if self.cartesian:
            return cartesian[:, self.dirmask]
        if self.spherical:
            return self._cartesian_to_spherical(cartesian)[:, self.dirmask]


class RelativePosition(Position):

    def __init__(self, arg='xyz', **kwarg):
        return Position.__init__(self, arg, **kwarg)

    def compute(self, inp1, inp2, reference_frame=None):
        # by adding a new axis we allow take the difference of positions
        # of the N inp atoms to the M reference_group ones.
        # pos here has dimension (N,M,3)

        inp1 = self._to_atomgroup(inp1)
        inp2 = self._to_atomgroup(inp2)
        pos = inp1.positions[:, np.newaxis] - inp2.positions
        if reference_frame is True:
            ref = ReferenceFrame().compute(inp2)
            pos = 0
        elif reference_frame is not None:
            try:
                ref = ReferenceFrame().compute(reference_frame)
                pos = 0
            except:
                raise ValueError(
                    'reference_frame can be None,True, or an atom group ')

        # let's get back to dimension (NxM,dimension)
        pos = pos[:, :, self.dirmask]
        dimension = len(self.dirmask)
        pos = pos.reshape((np.prod(pos.shape[:-1]), dimension))
        box = inp1.universe.dimensions[self.dirmask]
        if self.folded:
            cond = np.where(pos > box / 2.)
            pos[cond] -= box[cond[1]]
            cond = np.where(pos < -box / 2.)
            pos[cond] += box[cond[1]]

        if self.cartesian:
            return pos
        if self.spherical:
            return self._cartesian_to_spherical(pos)


class Velocity(Vector):
    """Atomic velocities"""

    def __init__(self, arg='xyz', **kwarg):
        Vector.__init__(self, arg)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic velocities

        """

        inp = self._to_atomgroup(inp)
        return inp.velocities[:, self.dirmask]


class Force(Vector):
    """Atomic forces"""

    def __init__(self, arg='xyz', **kwarg):
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
