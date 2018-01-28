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
