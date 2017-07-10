# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: observables
    ===================
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import stats

from MDAnalysis.lib import distances
from itertools import chain
from pytim import utilities
from pytim.correlator import correlate
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
        >>>
        >>> class TotalNumberOfParticles(observables.Observable):
        >>>     def compute(self):
        >>>         return len(self.u.atoms)
        >>>
        >>> u = mda.Universe(WATER_GRO)
        >>> o = TotalNumberOfParticles(u)
        >>> print o.compute()
        12000

    """

    __metaclass__ = ABCMeta

    def __init__(self, universe, options='',):
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
    def compute(self, inp):
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
                 observable=None, observable2=None):
        if max_radius is 'full':
            self.max_radius = np.min(universe.dimensions[:3])
        else:
            self.max_radius = max_radius
        self.rdf_settings = {'bins': nbins,
                             'range': (0., self.max_radius)}
        self.universe = universe
        self.nsamples = 0
        self.observable = observable
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

    def sample(self, g1=None, g2=None):
        self.n_frames += 1
        self.g2 = g2
        if g1 is not None:
            self.g1 = g1
        if g2 is None:
            self.g2 = self.g1  # all atoms by default (see __init__)

        if self.observable is not None:
            # determine weights, otherwise assumes number of atoms (default)
            fg1 = self.observable.compute(self.g1)
            if (self.g1 == self.g2 and self.observable == self.observable2):
                fg2 = fg1
            else:
                fg2 = self.observable2.compute(self.g2)

            if len(fg1) != len(self.g1) or len(fg2) != len(self.g2):
                raise Exception(
                    "Error, the observable passed to RDF should output"
                    "an array (of scalar or vectors) the same size of"
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
                 true2D=False, observable=None):
        RDF.__init__(self, universe, nbins=nbins, max_radius=max_radius,
                     start=start, stop=stop, step=step,
                     observable=observable)
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
    """Computes the triangulation of the surface and some associated
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

       >>> interface = pytim.ITIM(mda.Universe(WATER_GRO),molecular=False)
       >>> surface   = observables.LayerTriangulation(\
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


class Orientation(Observable):
    """Orientation of a group of points.

    :param str options: optional string. If `normal` is passed, the\
                        orientation of the normal vectors is computed

    This observable does not take into account boudary conditions.
    See :class:`~pytim.observables.MolecularOrientation`

    """

    def __init__(self, options=''):
        Observable.__init__(self, None, options=options)

    def compute(self, pos):
        """Compute the observable.

        :param ndarray inp:  the input atom group. The length be a multiple\
                             of three
        :returns: the orientation vectors

        For each triplet of positions A1,A2,A3, computes the unit vector
        beteeen A2-A1 and  A3-A1 or, if the option 'normal' is passed at
        initialization, the unit vector normal to the plane spanned by the
        three vectors

        """

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


class MolecularOrientation(Observable):
    """Molecular orientation vector of a set of molecules."""

    def compute(self, inp):
        """Compute the observable.

        :param variable inp: an AtomGroup or a ResidueGroup

        TODO: document and check if the function needs to be modified

        """
        if isinstance(inp, AtomGroup) and len(inp) != 3 * len(inp.residues):
            inp = inp.residues
        pos = self.fold_around_first_atom_in_residue(inp)
        return Orientation().compute(pos)


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

#
