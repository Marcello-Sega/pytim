# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: rdf
    ===========
"""
from __future__ import print_function
import numpy as np
from MDAnalysis.lib import distances


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
    >>> print (np.all(rdf1.rdf[:]==rdf2.rdf[:]))
    True
    >>> print (np.all(rdf1.rdf[:]==rdf3.rdf[:]))
    True

    """

    def __init__(self, universe,
                 nbins=75, max_radius='full',
                 start=None, stop=None, step=None,
                 observable=None, observable2=None, kargs1=None, kargs2=None):

        kargs1 = kargs1 or {}
        kargs2 = kargs2 or {}
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
        for ts in u.trajectory[::50] :
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
                 true2D=False, observable=None, kargs1=None, kargs2=None):

        RDF.__init__(self, universe, nbins=nbins, max_radius=max_radius,
                     start=start, stop=stop, step=step,
                     observable=observable, kargs1=kargs1, kargs2=kargs2)
        _dir={'x': 0, 'y': 1, 'z': 2}
        self.true2D = true2D
        if excluded_dir == 'auto':
            try:
                self.excluded_dir = self.universe.interface.normal
            except AttributeError:
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
            self.g2.positions = np.copy(_p2)
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
