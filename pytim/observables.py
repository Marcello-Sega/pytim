# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: observables
    ===================
"""
from __future__ import print_function

# here go all and only those classes derived from Observable
from abc import ABCMeta, abstractmethod
import numpy as np

from pytim import utilities
# check also imports at the end

# we try here to have no options passed
# to the observables, so that classes are
# not becoming behemoths that do everything.
# Simple preprocessing (filtering,...)
# should be done by the user beforehand,
# or implemented in specific classes.

from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup


class Observable(object):
    """ Instantiate an observable.

        This is a metaclass: it can be used to define observables that
        will behave "properly" with other classes/function in this module (e.g.
        by being called from an RDF object). A simple example is:

        >>> import pytim
        >>> import MDAnalysis as mda
        >>> from pytim import observables
        >>> from pytim.datafiles import WATER_GRO
        >>>
        >>> class TotalNumberOfParticles(observables.Observable):
        ...     def compute(self):
        ...         return len(self.u.atoms)
        >>>
        >>> u = mda.Universe(WATER_GRO)
        >>> o = TotalNumberOfParticles(u)
        >>> print (o.compute())
        12000

    """

    __metaclass__ = ABCMeta

    def __init__(self, universe, options=''):
        self.u = universe
        self.options = options

    # TODO: add proper whole-molecule reconstruction
    def fold_atom_around_first_atom_in_residue(self, atom):
        """ remove pbcs and puts an atom back close to the first one
            in the same residue. Does not check connectivity (works only
            for molecules smaller than box/2
        """

        # let's take the first atom in the residue as the origin
        box = self.u.trajectory.ts.dimensions[0:3]
        pos = atom.position - atom.residue.atoms[0].position
        pos[pos >= box / 2.] -= box[pos >= box / 2.]
        pos[pos < -box / 2.] += box[pos < -box / 2.]
        return pos

    def fold_around_first_atom_in_residue(self, inp):
        """ same as fold_atom_around_first_atom_in_residue()
            but for groups of atoms.
        """

        pos = []

        if isinstance(inp, Atom):
            pos.append(self.fold_atom_around_first_atom_in_residue(inp))
        elif isinstance(inp, (AtomGroup, Residue, ResidueGroup)):
            for atom in inp.atoms:
                pos.append(self.fold_atom_around_first_atom_in_residue(atom))
        else:
            raise Exception(
                "input not valid for fold_around_first_atom_in_residue()")
        return np.array(pos)

    @abstractmethod
    def compute(self, inp=None, kargs=None):
        kargs = kargs or {}
        pass


class LayerTriangulation(Observable):
    """ Computes the triangulation of the surface and some associated
        quantities. Notice that this forces the interface to be centered
        in the box.

        :param Universe universe: the MDAnalysis universe
        :param ITIM interface:    compute the triangulation with respect to it
        :param int  layer:        (default: 1) compute the triangulation with
                                  respect to this layer of the interface
        :param bool return_triangulation: (default: True) return the Delaunay
                                  triangulation used for the interpolation
        :param bool return_statistics: (default: True) return the Delaunay
                                  triangulation used for the interpolation

        :return:                  Observable LayerTriangulation

        Example:

        >>> import pytim
        >>> import MDAnalysis as mda
        >>> from pytim.datafiles import WATER_GRO

        >>> interface = pytim.ITIM(mda.Universe(WATER_GRO),molecular=False)
        >>> surface   = pytim.observables.LayerTriangulation(\
                            interface,return_triangulation=False)
        >>> stats     = surface.compute()
        >>> print ("Surface= {:04.0f} A^2".format(stats[0]))
        Surface= 6328 A^2

    """

    def __init__(self,
                 interface,
                 layer=1,
                 return_triangulation=True,
                 return_statistics=True):

        Observable.__init__(self, interface.universe)
        self.interface = interface
        self.layer = layer
        self.return_triangulation = return_triangulation
        self.return_statistics = return_statistics

    def compute(self, inp=None, kargs=None):
        stats = []
        layer_stats = [None, None]

        if self.interface.do_center is False:
            oldpos = np.copy(self.interface.universe.atoms.positions)
            self.interface.center()

        surface = self.interface._surfaces[0]
        surface.triangulation()

        if self.return_statistics is True:
            for side in [0, 1]:
                layer_stats[side] = utilities.triangulated_surface_stats(
                    surface.trimmed_surf_triangs[side],
                    surface.triangulation_points[side])
            # this average depends on what is in the stats, it can't be done
            # automatically
            stats.append(layer_stats[0][0] + layer_stats[1][0])
            # add here new stats other than total area

        if self.interface.do_center is False:
            self.interface.universe.positions = np.copy(oldpos)

        if self.return_triangulation is False:
            return stats
        else:
            return [
                stats, surface.surf_triang, surface.triangulation_points,
                surface.trimmed_surf_triangs
            ]


class IntrinsicDistance(Observable):
    """Initialize the intrinsic distance calculation.

    :param Universe universe: the MDAnalysis universe
    :param ITIM    interface: compute the intrinsic distance with respect
                              to this interface
    :param int     layer:     (default: 1) compute the intrinsic distance
                              with respect to this layer of the interface

    Example: TODO

    """

    def __init__(self, interface, layer=1):
        Observable.__init__(self, interface.universe)
        self.interface = interface
        self.layer = layer

    def compute(self, inp, kargs=None):
        """Compute the intrinsic distance of a set of points from the first
        layers.

        :param ndarray positions: compute the intrinsic distance for this set
                                  of points
        """
        return self.interface._surfaces[0].distance(inp)


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


class Position(Observable):
    """Atomic positions"""

    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic positions

        """
        return inp.positions


class Velocity(Observable):
    """Atomic velocities"""

    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic velocities

        """
        return inp.velocities


class Force(Observable):
    """Atomic forces"""

    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic forces

        """
        return inp.forces


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
        beteeen A2-A1 and  A3-A1 or, if the option 'normal' is passed at
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
        pos = flat.reshape(len(flat) / 3, 3)
        a = pos[1::3] - pos[0::3]
        b = pos[2::3] - pos[0::3]

        if 'normal' in self.options:
            v = np.cross(a, b)
        else:
            v = np.array(a + b)
        v = np.array([el / np.sqrt(np.dot(el, el)) for el in v])
        return v


from profile import Profile
from rdf import RDF, RDF2D
from free_volume import FreeVolume
from correlator import Correlator
