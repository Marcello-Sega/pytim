# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: RDF
    ===========
"""
from __future__ import print_function
import numpy as np
from .distributionfunction import DistributionFunction
from MDAnalysis.lib import distances
from . import Distance


class RDF(DistributionFunction):
    r"""Calculates a radial distribution function of some observable from two
    groups.

    The two functions must return an array (of scalars or of vectors)
    having the same size of the group. The scalar product between the
    two functions is used to weight the distriution function.

    .. math::

          g(r) = \frac{1}{N}\left\langle \sum_{i\neq j} \delta(r-|r_i-r_j|)\
            f_1(r_i,v_i)\cdot f_2(r_j,v_j) \right\rangle


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
    >>> rdf = observables.RDF(u,nbins=120,observable=nres,observable2=nres)
    >>>
    >>> interface = pytim.ITIM(u,alpha=2.,group=oxygens,cluster_cut=3.5,molecular=False)
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
                 max_distance='full',
                 nbins=75,
                 start=None,
                 stop=None,
                 step=None,
                 observable=None,
                 observable2=None,
                 **kargs):
        try:
            max_distance = kargs['max_radius']
        except:
            pass

        if max_distance == 'full':
            max_distance = np.min(universe.dimensions[:3]) / 2.

        self._distance = Distance()

        DistributionFunction.__init__(
            self,
            universe,
            order=2,
            nbins=nbins,
            start=start,
            stop=stop,
            step=step,
            generalized_coordinate=self._distance,
            generalized_coordinate2=self._distance,
            observable=observable,
            observable2=observable2,
            max_distance=max_distance,
            coords_in=['x', 'y', 'z'],
            coords_out=['r'],
            kargs1=None,
            kargs2=None)

    def sample(self, g1=None, g2=None, kargs1=None, kargs2=None):
        if g2 is None:
            g2 = g1
        DistributionFunction.sample(
            self,
            g1=g1,
            g2=g2,
            kargs1=kargs1,
            kargs2=kargs2)

    @property
    def bins(self):
        return 0.5 * (self.edges[0][:-1] + self.edges[0][1:])

    @property
    def rdf(self):
        # Volume in each radial shell
        dr = (self.edges[0][1:] - self.edges[0][:-1])
        avr = (self.edges[0][1:] + self.edges[0][:-1]) / 2.
        vol = 4. * np.pi * avr**2 * dr

        # normalization
        density = self.n_normalize / self.volume

        self._rdf = self.count / (density * vol * self.n_frames)

        return self._rdf



