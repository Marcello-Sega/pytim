# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: RDF
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


    :param double max_radius:       compute the rdf up to this distance.
                                    If 'full' is supplied (default) computes
                                    it up to half of the smallest box side.
    :param int nbins:               number of bins
    :param Observable observable:   observable for the first group
    :param Observable observable2:  observable for the second group

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

    def __init__(self,
                 universe,
                 nbins=75,
                 max_radius='full',
                 start=None,
                 stop=None,
                 step=None,
                 observable=None,
                 observable2=None,
                 kargs1=None,
                 kargs2=None):

        kargs1 = kargs1 or {}
        kargs2 = kargs2 or {}
        if max_radius is 'full':
            self.max_radius = np.min(universe.dimensions[:3])
        else:
            self.max_radius = max_radius
        self.rdf_settings = {'bins': nbins, 'range': (0., self.max_radius)}
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

    def _compute_observable(self, ka1, ka2):
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
            error = (fg1.shape[0] != self.g1.n_atoms
                     or fg2.shape[0] != self.g2.n_atoms)
        except:
            error = True
        return fg1, fg2, error

    def _determine_weights(self, fg1, fg2):
        # both are (arrays of) scalars
        if len(fg1.shape) == 1 and len(fg2.shape) == 1:
            weights = np.outer(fg1, fg2)
        # both are (arrays of) vectors
        elif len(fg1.shape) == 2 and len(fg2.shape) == 2:
            # TODO: tests on the second dimension...
            weights = np.dot(fg1, fg2.T)
        else:
            raise Exception("Error, shape of the observable output not handled"
                            "in RDF")
        return weights

    def sample(self, g1=None, g2=None, kargs1=None, kargs2=None):
        kargs1 = kargs1 or {}
        kargs2 = kargs2 or {}
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
            fg1, fg2, error = self._compute_observable(ka1, ka2)

            if error is True:
                raise Exception(
                    "Error, the observable passed to RDF should output "
                    "an array (of scalar or vectors) the same size of "
                    "the group")

            # numpy.histogram accepts negative weights
            self.rdf_settings['weights'] = self._determine_weights(fg1, fg2)

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

        box = self.universe.dimensions
        self.volume += np.product(box[:3])
        self.nsamples += 1
        self.n_squared += len(self.g1) * len(self.g2)

    @property
    def rdf(self):
        # Volume in each radial shell
        dr = (self.edges[1:] - self.edges[:-1])
        avr = (self.edges[1:] + self.edges[:-1]) / 2.
        vol = 4. * np.pi * avr**2 * dr

        # normalization
        density = self.n_squared / self.volume

        self._rdf = self.count / (density * vol * self.n_frames)

        return self._rdf
