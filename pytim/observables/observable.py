# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function

from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup
from abc import ABCMeta, abstractmethod
import numpy as np


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

    def select_direction(self, arg):
        _dirs = {'x': 0, 'y': 1, 'z': 2}

        def _inarg(string, inp):
            return np.any([string in e for e in inp])

        directions = np.array([True, True, True])
        if len(arg) > 0:
            if not _inarg('x', arg) or not _inarg('y', arg) or not _inarg(
                    'z', arg):
                RuntimeError(
                    "this observable accepts as argument a string like" +
                    " 'xy', 'z', ... to select components")
            directions = np.array([False, False, False])
            for key in _dirs.keys():
                if _inarg(key, arg):
                    directions[_dirs[key]] = True

        self.dirmask = np.where(directions)[0]

    @staticmethod
    def _to_atomgroup(inp):
        if isinstance(inp, Atom):
            ind = inp.index
            inp = inp.universe.atoms[ind:ind + 1]
        return inp

    @abstractmethod
    def compute(self, inp=None, kargs=None):
        kargs = kargs or {}
        pass

    @staticmethod
    def _():
        """
        This is a collection of basic tests to check
        that the observables are yelding the expected
        result.

        >>> # OBSERVABLES TEST: 1
        >>> import MDAnalysis as mda
        >>> import pytim
        >>> pytim.observables.Observable._() ; # coverage
        >>> from pytim import observables
        >>> from pytim.datafiles import *
        >>> import numpy as np
        >>> u = mda.Universe(_TEST_ORIENTATION_GRO)
        >>> o = observables.Orientation(u,options='molecular')
        >>> np.set_printoptions(precision=3,threshold=10)

        >>> print(o.compute(u.atoms).flatten())
        [ 1.     0.     0.     0.     1.     0.     0.    -0.707 -0.707]

        >>> np.set_printoptions()

        >>> # OBSERVABLES TEST: 2
        >>> u=mda.Universe(_TEST_PROFILE_GRO)
        >>> o=observables.Number()
        >>> p=observables.Profile(direction='x',observable=o)
        >>> p.sample(u.atoms)
        >>> low,up,avg =  p.get_values(binwidth=1.0)
        >>> print(low[0:3])
        [0. 1. 2.]
        >>> print(avg[0:3])
        [0.01 0.02 0.03]

        >>> # CORRELATOR TEST
        >>> from pytim.utilities import correlate
        >>> a = np.array([1.,0.,1.,0.,1.])
        >>> b = np.array([0.,2.,0.,1.,0.])
        >>> corr = correlate(b,a)
        >>> corr[np.abs(corr)<1e-12] = 0.0
        >>> ['{:.2f}'.format(i) for i in corr]
        ['0.00', '0.75', '0.00', '0.75', '0.00']


        >>> corr = correlate(b)
        >>> corr[np.abs(corr)<1e-12] = 0.0
        >>> ['{:.2f}'.format(i) for i in corr]
        ['1.00', '0.00', '0.67', '0.00', '0.00']

        >>> # PROFILE EXTENDED TEST: checks trajectory averaging
        >>> # and consistency in summing up layers  contributions
        >>> import numpy as np
        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from   pytim.datafiles import *
        >>> from   pytim.observables import Profile
        >>> u = mda.Universe(WATER_GRO,WATER_XTC)
        >>> g=u.select_atoms('name OW')
        >>> inter = pytim.ITIM(u,group=g,max_layers=4,centered=True, molecular=False)
        >>>
        >>> Layers=[]
        >>>
        >>> for n in np.arange(0,5):
        ...     Layers.append(Profile())
        >>> Val=[]
        >>> for ts in u.trajectory[:4]:
        ...     for n in range(len(Layers)):
        ...         if n == 0:
        ...             group = g
        ...         else:
        ...             group = u.atoms[u.atoms.layers == n]
        ...         Layers[n].sample(group)
        >>> for L in Layers:
        ...     Val.append(L.get_values(binwidth=2.0)[2])
        >>>

        >>> print (np.round(np.sum(np.array(Val[0]) * np.prod(u.dimensions[:3])) / len(Val[0]),decimals=0))
        4000.0

        >>> # the sum of the layers' contribution is expected to add up only close
        >>> # to the surface
        >>> print (not np.sum(np.abs(np.sum(Val[1:],axis=0)[47:] - Val[0][47:])>1e-15))
        True

        """
        pass
