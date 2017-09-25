# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-

# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: observables
    ===================
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import stats

from MDAnalysis.lib import distances
from scipy.spatial import cKDTree
from itertools import chain
from pytim import utilities

# we try here to have no options passed
# to the observables, so that classes are
# not becoming behemoths that do everything.
# Simple preprocessing (filtering,...)
# should be done by the user beforehand,
# or implemented in specific classes.

try:  # MDA >=0.16
    from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup
except BaseException:
    from MDAnalysis.core.AtomGroup import Atom, AtomGroup, Residue,\
        ResidueGroup


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
        >>> print o.compute()
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

        def _inarg(string, arg):
            return np.any([string in e for e in arg])

        directions = np.array([True, True, True])
        if len(arg) > 0:
            if not _inarg('x', arg) or not _inarg('y', arg) or not _inarg('z', arg):
                RuntimeError(
                    "Velocity accepts as argument a string like 'xy', 'z', ... to select components")
            directions = np.array([False, False, False])
            if _inarg('x', arg):
                directions[0] = True
            if _inarg('y', arg):
                directions[1] = True
            if _inarg('z', arg):
                directions[2] = True

        self.dirmask = np.where(directions == True)[0]

    @abstractmethod
    def compute(self, inp, kargs={}):
        pass


class RDF(object):
    """Calculates a radial distribution function of some observable from two
    groups.

    The two functions must return an array (of scalars or of vectors)
    having the same size of the group. The scalar product between the
    two functions is used to weight the distriution function.

    .. math::

          g(r) = \\frac{1}{N}\left\langle \sum_{i\\neq j} \delta(r-|r_i-r_j|)\
            f_1(r_i,v_i)\cdot f_2(r_j,v_j) \\right\\rangle


    :param double max_radius:  compute the rdf up to this distance. If 'full' is \
                          supplied (default) computes it up to half of the  \
                          smallest box side.
    :param int nbins:     number of bins
    :param Observable observable:  observable for first group
    :param Observable observable2: observable for second group

    Example:

    >>> import MDAnalysis as mda
    >>> import numpy as np
    >>> import pytim
    >>> from pytim import observables
    >>> from pytim.datafiles import *
    >>>
    >>> u = mda.Universe(WATER_GRO,WATER_XTC)
    >>> oxygens = u.select_atoms("name OW")
    >>>
    >>> nres = observables.NumberOfResidues()
    >>>
    >>> rdf = observables.RDF(u,nbins=120,\
        observable=nres,observable2=nres)
    >>>
    >>> interface = pytim.ITIM(u,alpha=2.,group=oxygens,\
        cluster_cut=3.5,molecular=False)
    >>>
    >>> for ts in u.trajectory[::50]:
    ...     layer=interface.layers[0,0]
    ...     rdf.sample(layer,layer)
    >>> rdf.count[0]=0
    >>> np.savetxt('RDF3D.dat', np.column_stack((rdf.bins,rdf.rdf)))


    Note that one needs to specify neither both groups, not both observables.
    If only the first group (observable) is specified, the second is assumed
    to be the same as the first, as in the following example:

    >>> rdf1 = observables.RDF(u,observable=nres)
    >>> rdf2 = observables.RDF(u,observable=nres)
    >>> rdf3 = observables.RDF(u,observable=nres,observable2=nres)
    >>>
    >>> rdf1.sample(layer)
    >>> rdf2.sample(layer,layer)
    >>> rdf3.sample(layer,layer)
    >>> print np.all(rdf1.rdf[:]==rdf2.rdf[:]),np.all(rdf1.rdf[:]==rdf3.rdf[:])
    True True

    """

    def __init__(self, universe,
                 nbins=75, max_radius='full',
                 start=None, stop=None, step=None,
                 observable=None, observable2=None, kargs1={}, kargs2={}):
        if max_radius is 'full':
            self.max_radius = np.min(universe.dimensions[:3])
        else:
            self.max_radius = max_radius
        self.rdf_settings = {'bins': nbins,
                             'range': (0., self.max_radius)}
        self.universe = universe
        self.nsamples = 0
        self.observable = observable
        self.kargs1 = kargs1
        self.kargs2 = kargs2
        if observable2 is None:
            self.observable2 = observable
        else:
            self.observable2 = observable2

        self.n_frames = 0
        self.volume = 0.0
        self.n_squared = 0
        count, edges = np.histogram([-1, -1], **self.rdf_settings)
        self.count = count * 0.0
        self.edges = edges
        self.bins = 0.5 * (edges[:-1] + edges[1:])
        self.g1 = self.universe.atoms
        self.g2 = None
        self._rdf = self.count

    def sample(self, g1=None, g2=None, kargs1={}, kargs2={}):
        self.n_frames += 1
        self.g2 = g2
        if g1 is not None:
            self.g1 = g1
        if g2 is None:
            self.g2 = self.g1  # all atoms by default (see __init__)
        ka1 = self.kargs1.copy()
        ka1.update(kargs1)
        ka2 = self.kargs2.copy()
        ka2.update(kargs2)
        if self.observable is not None:
            # determine weights, otherwise assumes number of atoms (default)
            try:
                fg1 = self.observable.compute(self.g1, ka1)
            except:
                fg1 = self.observable.compute(self.g1)

            if (self.g1 == self.g2 and self.observable == self.observable2):
                fg2 = fg1
            else:
                try:
                    fg2 = self.observable2.compute(self.g2, ka2)
                except:
                    fg2 = self.observable2.compute(self.g2)

            try:
                error = (
                    fg1.shape[0] != self.g1.n_atoms or fg2.shape[0] != self.g2.n_atoms)
            except:
                error = True

            if error == True:
                raise Exception(
                    "Error, the observable passed to RDF should output "
                    "an array (of scalar or vectors) the same size of "
                    "the group")

            # both are (arrays of) scalars
            if len(fg1.shape) == 1 and len(fg2.shape) == 1:
                _weights = np.outer(fg1, fg2)
            # both are (arrays of) vectors
            elif len(fg1.shape) == 2 and len(fg2.shape) == 2:
                # TODO: tests on the second dimension...
                _weights = np.dot(fg1, fg2.T)
            else:
                raise Exception(
                    "Error, shape of the observable output not handled"
                    "in RDF")
            # numpy.histogram accepts negative weights
            self.rdf_settings['weights'] = _weights

        # This still uses MDA's distance_array. Pro: works also in triclinic
        # boxes. Con: could be faster (?)
        _distances = np.zeros((len(self.g1), len(self.g2)), dtype=np.float64)
        distances.distance_array(
            self.g1.positions,
            self.g2.positions,
            box=self.universe.dimensions,
            result=_distances)

        count = np.histogram(_distances, **self.rdf_settings)[0]
        self.count += count

        try:
            self.volume += self.universe.trajectory.ts.volume
        except BaseException:
            self.volume += self.universe.dimensions[0] * \
                self.universe.dimensions[1] * self.universe.dimensions[2]
        self.nsamples += 1
        self.n_squared += len(self.g1) * len(self.g2)

    @property
    def rdf(self):
        # Volume in each radial shell
        dr = (self.edges[1:] - self.edges[:-1])
        avr = (self.edges[1:] + self.edges[:-1]) / 2.
        vol = 4. * np.pi * avr ** 2 * dr

        # normalization
        density = self.n_squared / self.volume

        self._rdf = self.count / (density * vol * self.n_frames)

        return self._rdf


class RDF2D(RDF):
    """Calculates a radial distribution function of some observable from two
    groups, projected on a plane.

    The two functions must return an array (of scalars or of vectors)
    having the same size of the group. The scalar product between the
    two functions is used to weight the distriution function.

    :param int nbins:         number of bins
    :param char excluded_dir: project position vectors onto the plane\
                              orthogonal to 'z','y' or 'z'
    :param Observable observable:   observable for group 1
    :param Observable observable2:  observable for group 2

    Example:

    >>> import MDAnalysis as mda
    >>> import numpy as np
    >>> import pytim
    >>> from pytim import *
    >>> from pytim.datafiles import *
    >>>
    >>> u = mda.Universe(WATER_GRO,WATER_XTC)
    >>> oxygens = u.select_atoms("name OW")
    >>> interface = pytim.ITIM(u,alpha=2.,group=oxygens,\
        cluster_cut=3.5,molecular=False)
    >>> rdf = observables.RDF2D(u,nbins=250)
    >>>
    >>> for ts in u.trajectory[::50] :
    ...     layer=interface.layers[0,0]
    ...     rdf.sample(layer,layer)
    >>> rdf.count[0]=0
    >>> np.savetxt('RDF.dat', np.column_stack((rdf.bins,rdf.rdf)))


    This results in the following RDF (sampling more frequently):

    .. plot::

        import MDAnalysis as mda
        import numpy as np
        import pytim
        import matplotlib.pyplot as plt
        from   pytim.datafiles import *
        u = mda.Universe(WATER_GRO,WATER_XTC)
        oxygens = u.select_atoms("name OW")
        interface = pytim.ITIM(u,alpha=2.,group=oxygens,\
                               cluster_cut=3.5, molecular=False)
        rdf=pytim.observables.RDF2D(u,nbins=250)
        for ts in u.trajectory[::] :
            layer=interface.layers[0,0]
            rdf.sample(layer,layer)
        rdf.count[0]=0

        plt.plot(rdf.bins, rdf.rdf)

        plt.gca().set_xlim([0,7])

        plt.show()

    """

    def __init__(self, universe,
                 nbins=75, max_radius='full',
                 start=None, stop=None, step=None, excluded_dir='auto',
                 true2D=False, observable=None, kargs1={}, kargs2={}):
        RDF.__init__(self, universe, nbins=nbins, max_radius=max_radius,
                     start=start, stop=stop, step=step,
                     observable=observable, kargs1=kargs1, kargs2=kargs2)
        _dir = {'x': 0, 'y': 1, 'z': 2}
        self.true2D = true2D
        if excluded_dir == 'auto':
            try:
                self.excluded_dir = self.universe.interface.normal
            except:
                self.excluded_dir = 2
        else:
            self.excluded_dir = _dir[excluded_dir]

    def sample(self, g1=None, g2=None):
        # this uses RDF.sample(), only changes in normalization/distance
        # calculation are handled here
        _ts = self.universe.trajectory.ts
        excl = self.excluded_dir
        if g2 is None:
            g2 = g1
        if self.true2D:
            p1 = g1.positions
            p2 = g2.positions
            _p1 = np.copy(p1)
            _p2 = np.copy(p2)
            p1[:, excl] = 0
            p2[:, excl] = 0
        RDF.sample(self, g1, g2)
        if self.true2D:
            self.g1.positions = np.copy(_p1)
            self.g1.positions = np.copy(_p2)
        # we subtract the volume added for the 3d case,
        # and we add the surface
        self.volume += _ts.volume * (1. / _ts.dimensions[excl] - 1.)

    @property
    def rdf(self):
        # Volume in each radial shell
        dr = (self.edges[1:] - self.edges[:-1])
        avr = (self.edges[1:] + self.edges[:-1]) / 2.
        vol = 2. * np.pi * avr * dr

        # normalization
        density = self.n_squared / self.volume

        self._rdf = self.count / (density * vol * self.n_frames)

        return self._rdf


class LayerTriangulation(Observable):
    """ Computes the triangulation of the surface and some associated
        quantities. Notice that this forces the interface to be centered
        in the box.

        :param Universe universe: the MDAnalysis universe
        :param ITIM    interface: compute the triangulation with respect to it
        :param int     layer: (default: 1) compute the triangulation with respect\
                       to this layer of the interface
        :param bool    return_triangulation: (default: True) return the Delaunay\
                       triangulation used for the interpolation
        :param bool    return_statistics: (default: True) return the Delaunay\
                       triangulation used for the interpolation

        :returns Observable LayerTriangulation:


        Example:

        >>> import pytim
        >>> import MDAnalysis as mda
        >>> from pytim.datafiles import WATER_GRO

        >>> interface = pytim.ITIM(mda.Universe(WATER_GRO),molecular=False)
        >>> surface   = pytim.observables.LayerTriangulation(\
                            interface,return_triangulation=False)
        >>> stats     = surface.compute()
        >>> print ("Surface= {:04.0f} A^2".format(stats[0]))
        Surface= 6751 A^2

    """

    def __init__(
            self,
            interface,
            layer=1,
            return_triangulation=True,
            return_statistics=True):

        Observable.__init__(self, interface.universe)
        self.interface = interface
        self.layer = layer
        self.return_triangulation = return_triangulation
        self.return_statistics = return_statistics

    def compute(self, inp=None):
        stats = []
        layer_stats = [None, None]

        if self.interface.do_center == False:
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

        if self.interface.do_center == False:
            self.interface.universe.positions = np.copy(oldpos)

        if self.return_triangulation is False:
            return stats
        else:
            return [
                stats,
                surface.surf_triang,
                surface.triangulation_points,
                surface.trimmed_surf_triangs
            ]


class IntrinsicDistance(Observable):
    """Initialize the intrinsic distance calculation.

    :param Universe universe: the MDAnalysis universe
    :param ITIM    interface: compute the intrinsic distance with respect\
                              to this interface
    :param int     layer: (default: 1) compute the intrinsic distance\
                          with respect to this layer of the interface

    Example: TODO

    """

    def __init__(self, interface, layer=1):
        Observable.__init__(self, interface.universe)
        self.interface = interface
        self.layer = layer

    def compute(self, inp):
        """Compute the intrinsic distance of a set of points from the first
        layers.

        :param ndarray positions: compute the intrinsic distance for this set\
                                  of points

        """
        return self.interface._surfaces[0].distance(inp)


class Number(Observable):
    """The number of atoms."""

    def __init__(self, *arg, **kwarg):
        """ No need to pass a universe for this observable. We accept
            extra arguments not to fail if they are passed anyway by mistake.
        """
        Observable.__init__(self, None)

    def compute(self, inp):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: one, for each atom in the group

        """
        return np.ones(len(inp))


class NumberOfResidues(Observable):
    """The number of residues.

    Instead of associating 1 to the center of mass of the residue, we
    associate 1/(number of atoms in residue) to each atom. In an
    homogeneous system, these two definitions are (on average)
    equivalent. If the system is not homogeneous, this is not true
    anymore.

    """

    def __init__(self, *arg, **karg):
        """ No need to pass a universe for this observable. We accept
            extra arguments not to fail if they are passed anyway by mistake.
        """
        Observable.__init__(self, None)

    def compute(self, inp):
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
        """ No need to pass a universe for this observable. We accept
            extra arguments not to fail if they are passed anyway by mistake.
        """
        Observable.__init__(self, None)
        self.select_direction(arg)

    def compute(self, inp):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic positions

        """
        return inp.positions[:, self.dirmask]


class Velocity(Observable):
    """Atomic velocities"""

    def __init__(self, *arg, **kwarg):
        """ No need to pass a universe for this observable.
            :parameter *arg str :  'x','y','xy',... to select components
        """
        Observable.__init__(self, None)
        self.select_direction(arg)

    def compute(self, inp):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic positions

        """
        return inp.velocities[:, self.dirmask]


class Force(Observable):
    """Atomic velocities"""

    def __init__(self, *arg, **kwarg):
        """ No need to pass a universe for this observable. We accept
            extra arguments not to fail if they are passed anyway by mistake.
        """
        Observable.__init__(self, None)
        self.select_direction(arg)

    def compute(self, inp):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group
        :returns: atomic positions

        """
        return inp.forces[:, self.dirmask]


class Orientation(Observable):
    """Orientation of a group of points.

    :param str options: optional string. If `normal` is passed, the\
                        orientation of the normal vectors is computed
                        If the option 'molecular' is passed at initialization, \
                        coordinates of the second and third atoms are \
                        folded around those of the first.

    """

    def __init__(self, universe, options=''):
        self.u = universe
        self.options = options

    def compute(self, inp):
        """Compute the observable.

        :param ndarray inp:  the input atom group. The length be a multiple\
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


class Profile(object):
    """Calculates the profile (normal, or intrinsic) of a given observable
    across the simulation box.

    :param AtomGroup  group:        calculate the profile based on\
                                    this group
    :param str        direction:    'x','y', or 'z' : calculate the\
                                    profile along this direction
    :param Observable observable:   calculate the profile of this\
                                    quantity. If None is supplied, it \
                                    defaults to the number density
    :param ITIM       interface:    if provided, calculate the intrinsic\
                                    profile with respect to the first\
                                    layers
    :param AtomGroup  center_group: if `interface` is not provided, this\
                                    optional group can be supplied to\
                                    center the system

    Example (non-intrinsic, total profile + first 4 layers ):

    >>> import numpy as np
    >>> import MDAnalysis as mda
    >>> import pytim
    >>> from   pytim.datafiles import *
    >>> from   pytim.observables import Profile
    >>> from matplotlib import pyplot as plt
    >>>
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
    >>>
    >>> for ts in u.trajectory[::50]:
    ...     for L in Layers:
    ...         L.sample()
    >>>
    >>> density=[]
    >>> for L in Layers:
    ...     low,up,avg = L.get_values(binwidth=0.5)
    ...     density.append(avg)
    >>>
    >>> np.savetxt('profile.dat',list(zip(low,up,density[0],density[1],density[2],density[3],density[4])))

    This results in the following profile (zooming close to the interface border)

    .. plot::

        import numpy as np
        import MDAnalysis as mda
        import pytim
        from   pytim.datafiles import *
        from   pytim.observables import Profile
        from matplotlib import pyplot as plt

        u = mda.Universe(WATER_GRO,WATER_XTC)
        g=u.select_atoms('name OW')
        inter = pytim.ITIM(u,group=g,max_layers=4,centered=True)

        Layers=[]
        AIL = inter.atoms.in_layers

        Layers.append(Profile(u.atoms))
        for n in np.arange(4):
            Layers.append(Profile(AIL[::,n]))

        for ts in u.trajectory[::]:
            for L in Layers:
                L.sample()

        density=[]
        for L in Layers:
            low,up,avg = L.get_values(binwidth=0.5)
            density.append(avg)


        for dens in density:
            plt.plot(low,dens)

        plt.gca().set_xlim([80,120])
        plt.show()


    Example (intrinsic):

    >>> from pytim import observables

    >>> u       = mda.Universe(WATER_GRO,WATER_XTC)
    >>> oxygens = u.select_atoms("name OW")
    >>>
    >>> obs     = observables.Number()
    >>>
    >>> interface = pytim.ITIM(u, alpha=2.0, max_layers=1,cluster_cut=3.5,centered=True,molecular=False)
    >>> profile = observables.Profile(group=oxygens,observable=obs,interface=interface)
    >>>
    >>> for ts in u.trajectory[::50]:
    ...     profile.sample()
    >>>
    >>> low, up, avg = profile.get_values(binwidth=0.1)
    >>> np.savetxt('profile.dat',list(zip(low,up,avg)))


    This results in the following profile:

    .. plot::

        import MDAnalysis as mda
        import numpy as np
        import pytim
        import matplotlib.pyplot as plt
        from   pytim.datafiles   import *

        u       = mda.Universe(WATER_GRO,WATER_XTC)
        oxygens = u.select_atoms("name OW")

        obs     = pytim.observables.Number()

        interface = pytim.ITIM(u, alpha=2.0, max_layers=1,cluster_cut=3.5,centered=True,molecular=False)
        profile = pytim.observables.Profile(group=oxygens,observable=obs,interface=interface)

        for ts in u.trajectory[::]:
            profile.sample()

        low, up, avg = profile.get_values(binwidth=0.2)
        z = (low+up)/2.
        plt.plot(z, avg)
        axes = plt.gca()
        axes.set_xlim([-20,20])
        axes.set_ylim([0,0.1])
        plt.show()




    """

    def __init__(
            self,
            group,
            direction='z',
            observable=None,
            interface=None,
            center_group=None):
        # TODO: the directions are handled differently, fix it in the code
        _dir = {'x': 0, 'y': 1, 'z': 2}
        self._dir = _dir[direction]
        self.universe = group.universe
        self.group = group
        self.interface = interface
        self.center_group = center_group
        self.box = self.universe.dimensions[:3]

        self._range = [0., self.box[self._dir]]

        if self.interface is not None:
            self._range -= self.box[self._dir] / 2.

        if not isinstance(group, AtomGroup):
            raise TypeError("The first argument passed to "
                            "Profile() must be an AtomGroup.")
        if observable is None:
            self.observable = Number()
        else:
            self.observable = observable
        self.binsize = 0.1  # this is used for internal calculations, the
        # output binsize can be specified in
        # self.get_values()

        self.sampled_values = []
        self.sampled_bins = []
        self.pos = [utilities.get_x, utilities.get_y, utilities.get_z]

    def sample(self):
        # TODO: implement progressive averaging to handle very long trajs
        # TODO: implement memory cleanup
        self.box = self.universe.dimensions[:3]

        if self.interface is None:
            pos = self.group.positions[::, self._dir]
        else:
            pos = IntrinsicDistance(self.interface).compute(self.group)

        values = self.observable.compute(self.group)
        nbins = int(
            self.universe.trajectory.ts.dimensions[self._dir] / self.binsize)
        # we need to make sure that the number of bins is odd, so that the
        # central one encompasses zero (to make the delta-function
        # contribution appear always in this bin)
        if(nbins % 2 > 0):
            nbins += 1
        avg, bins, binnumber = stats.binned_statistic(
            pos, values, range=self._range, statistic='sum', bins=nbins)
        avg[np.isnan(avg)] = 0.0
        self.sampled_values.append(avg)
        # these are the bins midpoints
        self.sampled_bins.append(bins[1:] - self.binsize / 2.)

    def get_values(self, binwidth=None, nbins=None):
        if not self.sampled_values:
            raise UserWarning("No profile sampled so far")
        # we use the largest box (largest number of bins) as reference.
        # Statistics will be poor at the boundaries, but like that we don't
        # loose information
        max_bins = np.max(map(len, self.sampled_bins))
        max_size = max_bins * self.binsize

        if binwidth is not None:  # overrides nbins
            nbins = max_size / binwidth
        if nbins is None:  # means also binwidth must be none
            nbins = max_bins

        if(nbins % 2 > 0):
            nbins += 1

        avg, bins, _ = stats.binned_statistic(
            list(
                chain.from_iterable(
                    self.sampled_bins)), list(
                chain.from_iterable(
                    self.sampled_values)), range=self._range, statistic='sum',
            bins=nbins)
        avg[np.isnan(avg)] = 0.0
        vol = np.prod(self.box) / nbins
        return [bins[0:-1], bins[1:], avg / len(self.sampled_values) / vol]


class Correlator(object):
    """ Computes the (self) correlation of an observable (scalar or vector)

    :param Observable observable: compute the autocorrelation of this observable. If observable is None and reference is \
                                  not, the survival probability in the  
    :param bool reduced: when the observable is a vector, average over all spatial direction if reduced==True (default)
    :param bool normalize: normalize the correlation to 1 at t=0
    :param AtomGroup reference: if the group passed to the sample() function changes its composition along the trajectory \
                                (such as a layer group), a reference group that includes all atoms that could appear in the \
                                variable group must be passed, in order to provide a proper normalization. See\ the example \
                                below.
    :param double memory_warn: if not None, print a warning once this threshold of memory (in Mb) is passed.


    Example:

    >>> import pytim
    >>> import MDAnalysis as mda
    >>> import numpy as np
    >>> from pytim.datafiles import WATERSMALL_GRO
    >>> from pytim.utilities import lap
    >>> #  tmpdir here is specified only for travis
    >>> WATERSMALL_TRR = pytim.datafiles.pytim_data.fetch('WATERSMALL_LONG_TRR',tmpdir='./') # doctest:+ELLIPSIS
    checking presence of a cached copy ...

    >>> u = mda.Universe(WATERSMALL_GRO,WATERSMALL_TRR)
    >>> g = u.select_atoms('name OW')

    >>> velocity = pytim.observables.Velocity()
    >>> corr = pytim.observables.Correlator(observable=velocity)
    >>> for t in u.trajectory[1:]:
    ...     corr.sample(g)
    >>> vacf = corr.correlation()

    This produces the following (steps of 1 fs):

    .. plot::

        import pytim
        import MDAnalysis as mda
        import numpy as np
        from pytim.datafiles import WATERSMALL_GRO
        from pytim.utilities import lap
        WATERSMALL_TRR=pytim.datafiles.pytim_data.fetch('WATERSMALL_LONG_TRR')

        u = mda.Universe(WATERSMALL_GRO,WATERSMALL_TRR)
        g = u.select_atoms('name OW')

        velocity = pytim.observables.Velocity()
        corr = pytim.observables.Correlator(observable=velocity)
        for t in u.trajectory[1:]:
            corr.sample(g)

        vacf = corr.correlation()

        from matplotlib import pyplot as plt
        plt.plot(vacf[:1000])
        plt.plot([0]*1000)
        plt.show()


    In order to compute the correlation for variable groups, one should proceed
    as follows:

    >>> corr = pytim.observables.Correlator(observable=velocity,reference=g)
    >>> # notice the molecular=False switch, in order for the
    >>> # layer group to be made of oxygen atoms only and match
    >>> # the reference group
    >>> inter = pytim.ITIM(u,group=g,alpha=2.0,molecular=False)
    >>> # example only: sample longer for smooth results
    >>> for t in u.trajectory[1:10]: 
    ...     corr.sample(inter.atoms)
    >>> layer_vacf = corr.correlation()


    In order to compute the survival probability of some atoms in a layer, it is possible
    to pass observable=None together with the reference group:

    >>> corr = pytim.observables.Correlator(observable=None, reference = g)
    >>> inter = pytim.ITIM(u,group=g,alpha=2.0, molecular=False)
    >>>  # example only: sample longer for smooth results
    >>> for t in u.trajectory[1:10]:
    ...     corr.sample(inter.atoms)
    >>> survival = corr.correlation()

    """

    def __init__(self, universe=None, observable=None, reference=None, memory_warn=None):
        self.name = self.__class__.__name__
        self.observable = observable
        self.reference = reference
        self.timeseries = []
        self.maskseries = []
        self.shape = None
        self.mem_usage = 0.0
        self.warned = False
        self.nsamples = 0

        if self.reference is not None and self.observable is not None:
            self.masked = True
        else:
            self.masked = False

        if memory_warn is None:
            self.warned = True
            self.memory_warn = 0.0
        else:
            self.memory_warn = memory_warn
        if reference is not None:
            if observable is not None:
                self.reference_obs = observable.compute(reference) * 0.0
            else:
                self.reference_obs = np.zeros(len(reference), dtype=np.double)
            if len(self.reference_obs.shape) > 2:
                raise RuntimeError(
                    self.name + ' works only with scalar and vectors')
        else:
            if observable is None:
                raise RuntimeError(
                    self.name + ': at least the observable or the reference must be specified')

    def sample(self, group):
        """ Sample the timeseries for the autocorrelation function

            :parameter AtomGroup group: compute the observable on the atoms of this group

        """
        self.nsamples += 1
        if self.reference is not None:  # could be intermittent or continuous:
                                        # we need to collect also the residence
                                        # function
            # the residence function (1 if in the reference group, 0 otherwise)
            mask = np.isin(self.reference, group)
            # append the residence function to its timeseries
            self.maskseries.append(list(mask))
            if self.observable is not None:
                # this copies a vector of zeros with the correct shape
                sampled = self.reference_obs.copy()
                obs = self.observable.compute(group)
                sampled[np.where(mask)] = obs
                self.timeseries.append(list(sampled.flatten()))
            else:
                self.timeseries = self.maskseries
                if self.shape is None:
                    self.shape = (1,)
                sampled = mask
        else:
            if self.observable is None:
                RuntimeError(
                    'Cannot compute the survival probability without a reference group')
            sampled = self.observable.compute(group)
            self.timeseries.append(list(sampled.flatten()))

        self.mem_usage += sampled.nbytes / 1024.0 / 1024.0  # in Mb
        if self.mem_usage > self.memory_warn and self.warned == False:
            print "Warning: warning threshold of",
            print self.memory_warn, "Mb exceeded"
            self.warned = True

        if self.shape is None:
            self.shape = sampled.shape

    def correlation(self, reduced=True, normalized=True, continuous=True):
        """ Calculate the autocorrelation from the sampled data

            :parameter bool reduced: if True (default) average over particles and spatial directions.
                                        if False, no average is performed. If you want to compute the
                                        correlations only along some spatial direction, specify them in
                                        the observable, e.g., observable=pytim.observables.Velocity('xy')
            :parameter bool normalized: normalize the correlation function to: its zero-time value for
                                        regular correlations; to the average of the characteristic function
                                        for the survival probability.
            :parameter bool continuous: applies only when a reference group has been specified: if True 
                                        (default) the contribution of a particle at time lag $\\tau=t_1-t_0$
                                        is considered only if the particle did not leave the reference group
                                        between $t_0$ and $t_1$. If False, the intermittent correlation is
                                        calculated, and the above restriction is released.

            Example:

            >>> # We build a fake trajectory to test the various options:
            >>> import MDAnalysis as mda
            >>> import pytim
            >>> import numpy as np
            >>> from pytim.datafiles import WATER_GRO
            >>> from pytim.observables import Correlator, Velocity
            >>> np.set_printoptions(suppress=True,precision=3)
            >>> 
            >>> u = mda.Universe(WATER_GRO)
            >>> g = u.atoms[0:2]
            >>> g.velocities*=0.0
            >>> g.velocities+=1.0
            >>>
            >>> vv = Correlator(observable=Velocity('x'), reference=g) # velocity autocorrelation along x, variable group
            >>> nn = Correlator(reference=g) # survival probability in group g
            >>>
            >>> for c in [vv,nn]:
            ...     c.sample(g)     # t=0
            ...     c.sample(g[:1]) # t=1, exclude the second particle
            ...     g.velocities /= 2. # from now on v=0.5
            ...     c.sample(g)     # t=2
            >>>

            The timeseries sampled can be accessed using:

            >>> print vv.timeseries # rows refer to time, columns to particle 
            [[1.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
            >>>
            >>> print nn.timeseries
            [[True, True], [True, False], [True, True]]
            >>>

            Note that the average of  the characteristic function $h(t)$ is done over all
            trajectories, including those that start with h=0. When normalizing, the correlation
            $< h(t) h(0) >$ is divided by the average $<h>$ computed over all trajectores that
            extend up to a time lag $t$. This behavior is different from the calculation of the 
            autocorrelation function (see below).

            >>> # reduced, normalized, continuous
            >>> corr = nn.correlation()
            >>> print np.allclose(corr, [ 5./5   ,  2./3 ,  1./2 ])
            True
            >>> # not reduced, normalized, continuous
            >>> corr = nn.correlation(reduced=False)
            >>> print np.allclose(corr, [ [ 1. , 1.],  [ 1. , 0. ] , [ 1. , 0. ] ])
            True
            >>> # reduced, normalized, intermittent 
            >>> corr = nn.correlation(continuous=False)
            >>> print np.allclose(corr, [ 5./5   ,  2./3 ,  2./2 ])
            True
            >>> # not reduced, not normalized, intermittent
            >>> corr = nn.correlation(reduced=False,normalized=False,continuous=False)
            >>> print np.allclose(corr, [ [ 3./3 , 2./3],  [ 2./2 , 0. ] , [ 1. , 1. ] ])
            True

            The autocorrelation functions are calculated by taking into account in the average
            only those trajectory that start with $h=1$ (i.e., which start within the reference
            group). The normalization is done by dividing the correlation at time lag $t$ by its 
            value at time lag 0 computed over all trajectories that extend up to time lag $t$ and 
            do not start with $h=0$. Note that in this way the `reduce=True` option won't give the same 
            result as plain averaging the `reduce=False` case.

            >>> # reduced, not normalizd, intermittent
            >>> corr = vv.correlation(normalized=False,continuous=False)
            >>> print np.allclose(corr, [ (1+1+0.25+1+0.25)/5   ,  (1+0.5+0.)/3 ,  (0.5+0.5)/2])
            True
            >>> # check normalization
            >>> np.all(vv.correlation(continuous=False) == corr/corr[0])
            True
            >>> # not reduced, not normalized, intermittent
            >>> corr = vv.correlation(reduced=False,normalized=False,continuous=False)
            >>> print np.allclose(corr, [ [ (1+1+0.25)/3 , (1+0.25)/2.],  [ (1+0.5)/2 , 0. ] , [ 0.5 , 0.5 ] ])
            True
            >>> # check normalization
            >>> print np.all(vv.correlation(reduced=False,continuous=False) == corr/corr[0])
            True
            >>> # reduced, not normalizd, continuous
            >>> corr = vv.correlation(normalized=False,continuous=True)
            >>> print np.allclose(corr, [ (1+1+0.25+1+0.25)/5   ,  (1+0.5+0.)/3 ,  (0.5+0.)/2])
            True
            >>> # check normalization
            >>> np.all(vv.correlation(continuous=True) == corr/corr[0])
            True
            >>> # not reduced, not normalized, continuous
            >>> corr = vv.correlation(reduced=False,normalized=False,continuous=True)
            >>> print np.allclose(corr, [ [ (1+1+0.25)/3 , (1+0.25)/2.],  [ (1+0.5)/2 , 0./1. ] , [ 0.5/1 , 0./1 ] ])
            True
            >>> # check normalization
            >>> print np.all(vv.correlation(reduced=False,continuous=True) == corr/corr[0])
            True


        """
        intermittent = not continuous
        self.dim = self._determine_dimension()

        # the standard correlation
        if self.reference is None:
            ts = np.asarray(self.timeseries)
            corr = utilities.correlate(ts)
            norm = corr[:, 0]
            if reduced == True:
                corr = np.average(corr, axis=1)
                norm = corr[0]
            if normalized == True:
                corr /= norm
            return corr

        # prepare the mask for the intermittent/continuous cases
        if intermittent == True:
            ms = np.asarray(self.maskseries, dtype=np.double)
        else:  # we add Falses at the begining and at the end to ease the splitting in sub-trajectories
            falses = [[False] * len(self.maskseries[0])]
            ms = np.asarray(falses + self.maskseries + falses)

        # compute the survival probabily
        if self.observable is None:
            return self._survival_probability(ms, normalized, reduced, intermittent)
        # compute the autocorrelation function
        else:
            ts = np.asarray(self.timeseries)
            return self._autocorrelation(ts, ms, normalized, reduced, intermittent)

    def _autocorrelation(self, ts, ms, normalized, reduced, intermittent):

        if intermittent == True:
            corr, weight = self._autocorrelation_intermittent(ts, ms)
        else:
            corr, weight = self._autocorrelation_continuous(ts, ms)
        if reduced == True:
            corr = np.average(corr, axis=1, weights=weight)
        if normalized == True:
            corr /= corr[0]

        return corr

    def _determine_dimension(self):
        self.nseries = max(len(self.timeseries), len(self.maskseries))

        if len(self.shape) == 1:
            shape = (self.nseries, self.shape[0], 1)
            dim = 1
        elif len(self.shape) == 2:
            shape = (self.nseries, self.shape[0], self.shape[1])
            dim = self.shape[1]
        else:
            raise RuntimeError(
                "Correlations of tensorial quantites not allowed in " + self.name)
        return dim

    def _survival_probability(self, ms, normalized, reduced, intermittent):
        if intermittent == True:
            corr, norm = self._survival_intermittent(ms, normalized)
        else:
            corr, norm = self._survival_continuous(ms, normalized)

        if reduced == True:
            corr = np.average(corr, axis=1)
            if normalized == True:
                corr /= np.average(norm, axis=1)
        elif normalized == True:
            corr /= norm

        return corr

    def _survival_intermittent(self, ms, normalized):
        norm = None
        corr = utilities.correlate(ms)
        if normalized == True:
            norm = np.cumsum(ms, axis=0)[::-1]
            norm /= (1. + np.arange(norm.shape[0])
                     [::-1]).reshape(norm.shape[0], 1)

        return corr, norm

    def _survival_continuous(self, ms, normalized):
        norm = None
        n_part = len(ms[0])
        corr = np.zeros((self.nseries, n_part))
        norm = corr.copy()
        counting = (1. + np.arange(len(self.timeseries)))

        for part in range(n_part):
            edges = np.where(ms[::, part][:-1] != ms[::, part][1:])[0]
            deltat = edges[1::2] - edges[0::2]
            # for each of the disconnected segments:
            for n, dt in enumerate(deltat):
                # no need to compute the correlation, we know what it is
                corr[0:dt, part] += dt * 1. / len(self.timeseries)

            norm = np.cumsum(ms[1:-1], axis=0)[::-1]
            norm = norm / counting.reshape(counting.shape[0], 1)[::-1]

        return corr, norm

    def _autocorrelation_intermittent(self, ts, ms):

        dim = self.dim

        maskcorr = utilities.correlate(ms)
        cond = np.where(maskcorr > 1e-9)
        corr = ts.copy()
        weight = ts.copy()
        w = np.cumsum(ms, axis=0)[::-1]
        for xyz in range(dim):
            corr[:, xyz::dim] = utilities.correlate(ts[:, xyz::dim] * ms)
            corr[:, xyz::dim][cond] /= maskcorr[cond]
            weight[:, xyz::dim] = w

        return corr, weight

    def _autocorrelation_continuous(self, ts, ms):

        dim = self.dim
        norm = None
        n_part = len(ms[0])
        corr = np.zeros(ts.shape)
        weight = corr.copy()

        for xyz in range(dim):
            weight[:, xyz::dim] = np.cumsum(ms[1:-1], axis=0)[::-1]

        for part in range(n_part):
            edges = np.where(ms[::, part][:-1] != ms[::, part][1:])[0]
            deltat = edges[1::2] - edges[0::2]
            for n, dt in enumerate(deltat):  # for each of the disconnected segments
                t1 = edges[2 * n]
                t2 = edges[2 * n + 1]
                i1 = dim * part
                i2 = dim * (part + 1)
                corr[0:dt,
                     i1:i2] += utilities.correlate(ts[t1:t2, i1:i2]) / len(deltat)
                # print "dt=",dt,"index=",i1,i2,"corr+=",
                # utilities.correlate(ts[t1:t2,i1:i2]),"will be divided by
                # ",len(deltat)

        return corr, weight


class VoronoiTessellation(Observable):

    def __init__(self, universe):
        self.quantity = {'density': self.localdensity, 'volume': self.localvolume,
                         'surface': self.localsurface, 'vertices': self.vertices, 'facets': self.facets}
        self.defaultkargs = {'stored': False, 'quantity': 'volume'}
        self.u = universe

    def compute(self, group, kargs={}):
        r"""
            Computes the Voronoi tessellation and returns a related observable:

            :param AtomGroup group: tessellate the atomic positions of this group
            :param \**kwargs: see below

            :Keyword Arguments:
                * str *quantity*: 'density','volume','surface'. By default returns the volume.
                * bool *stored*: recompute the tessellation if stored==False, otherwise not.

            Example:

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import WATER_GRO
            >>> from pytim.observables import VoronoiTessellation
            >>> 
            >>> u = mda.Universe(WATER_GRO)
            >>> vor = VoronoiTessellation(u)
            >>> print vor.compute(quantity='volume')
            >>> print vor.compute(stored=True,quantity='surface')

        """
        t = self.u.trajectory.ts
        box = t.dimensions[:3]
        ka = self.defaultkargs.copy()
        ka.update(kargs)
        if ka['stored'] == False:
            pos, _ = pytim.utilities.generate_periodic_border(
                group.positions, box, box / 2.)
            vor = Voronoi(pos)
            self.hulls = []
            N = len(group.positions)
            for ireg in vor.point_region[:N]:
                # Convex hull of one Voronoi cell:
                ivert = vor.regions[ireg]
                points = vor.vertices[ivert]
                hull = ConvexHull(points)
                self.hulls.append(hull)
        return self.quantity[ka['quantity']]()

    def vertices(self):
        return np.array([1. * hull.vertices.shape[0] for hull in self.hulls])

    def facets(self):
        return np.array([1. * hull.nsimplex for hull in self.hulls])

    def localsurface(self):
        return np.array([hull.area for hull in self.hulls])

    def localvolume(self):
        return np.array([hull.volume for hull in self.hulls])

    def localdensity(self):
        return np.array([1. / hull.volume for hull in self.hulls])


class FreeVolume(object):
    """ Calculates the fraction of free volume in the system, or its profile.

        Note that this does not fit in the usual observable class as i it can not be 
        expressed as a property of particles, and needs some kind of gridding to be calculated.

        :param Universe universe:  the universe
        :param int npoints: number of Monte Carlo sampling points (default: 10x the number of atoms in the universe)

        Example:
        >>> import MDAnalysis as mda
        >>> import numpy as np
        >>> import pytim
        >>> from pytim.datafiles import CCL4_WATER_GRO

        >>> u = mda.Universe(CCL4_WATER_GRO)
        >>> inter = pytim.ITIM(u) # just to make sure all radii are set 
        >>> np.random.seed(0) # ensure reproducibility of test
        >>> FV = pytim.observables.FreeVolume(u)
        >>> bins, prof ,err = FV.compute_profile()

        >>> free, err = FV.compute()
        >>> print '{0:.3f}'.format(free)+' +/- {0:.3f}'.format(err)
        0.707 +/- 0.001

    """

    def __init__(self, universe, npoints=None):
        self.u = universe
        if npoints is None:
            npoints = 10 * len(universe.atoms)
        self.npoints = npoints

    def _compute(self, inp=None):
        res = np.array(0)
        box = self.u.dimensions[:3].copy()
        tree = cKDTree(np.random.random((self.npoints, 3)) * box)
        if inp is None:
            inp = self.u.atoms
        if not isinstance(inp, AtomGroup):
            raise RuntimeError(self.__class__.__name__ +
                               'compute needs AtomGroup as an input')
        # np.unique here avoids counting contributions from overlapping spheres
        radii = np.unique(inp.radii)

        for radius in radii:
            where = np.where(np.isclose(inp.radii,  radius))
            lst = [e for l in tree.query_ball_point(
                inp.positions[where], radius) for e in l]
            res = np.append(res, lst)
        return np.unique(res), tree.data

    def compute_profile(self, inp=None, nbins=30, direction=2):
        """ Compute a profile of the free volume fraction

            :param AtomGroup inp:  compute the volume fraction of this group, None selects the complete universe
            :param int nbins: number of bins, by default 30
            :param int direction: direction along wich to compute the the profile, z (2) by default

            :returns bins,fraction,error: the left limit of the bins, the free volume fraction in each bin, the associated std deviation
        """
        nbins += 1
        box = self.u.dimensions[:3].copy()

        slabwidth = box[direction] / nbins
        slabvol = self.u.trajectory.ts.volume / nbins

        bins = np.arange(nbins) * slabwidth

        histo = []
        error = []
        res, data = self._compute(inp)
        for i in range(nbins - 1):
            condition = np.logical_and(
                data[:, direction] > bins[i], data[:, direction] < bins[i + 1])
            in_slab = np.where(condition)
            n_in_slab = np.sum(condition * 1.0)
            if n_in_slab == 0:
                histo.append(0.0)
                error.append(0.0)
            else:
                ratio = np.sum(np.isin(res, in_slab) * 1.0) / n_in_slab
                histo.append(1. - ratio)
                error.append(np.sqrt(ratio * (1. - ratio) / n_in_slab))
        return bins, np.array(histo), np.array(error)

    def compute(self, inp=None):
        """ Compute the total free volume fraction in the simulation box

            :param AtomGroup inp:  compute the volume fraction of this group, None selects the complete universe
            :param int nbins: number of bins, by default 30

            :returns fraction, error: the free volume fraction and associated error

        """
        _, free, err = self.compute_profile(inp, nbins=1)
        return free[0], err[0]


#
