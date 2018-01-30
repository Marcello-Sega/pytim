# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: RDF2D
    =============
"""
from __future__ import print_function
import numpy as np
from MDAnalysis.lib import distances
from . import RDF


class RDF2D(RDF):
    """Calculates a radial distribution function of some observable from two
    groups, projected on a plane.

    The two functions must return an array (of scalars or of vectors)
    having the same size of the group. The scalar product between the
    two functions is used to weight the distriution function.

    :param int nbins:               number of bins
    :param char excluded_dir:       project position vectors onto the plane
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

    def __init__(self,
                 universe,
                 nbins=75,
                 max_radius='full',
                 start=None,
                 stop=None,
                 step=None,
                 excluded_dir='auto',
                 true2D=False,
                 observable=None,
                 kargs1=None,
                 kargs2=None):

        RDF.__init__(
            self,
            universe,
            nbins=nbins,
            max_radius=max_radius,
            start=start,
            stop=stop,
            step=step,
            observable=observable,
            kargs1=kargs1,
            kargs2=kargs2)
        _dir = {'x': 0, 'y': 1, 'z': 2}
        self.true2D = true2D
        if excluded_dir == 'auto':
            try:
                self.excluded_dir = self.universe.interface.normal
            except AttributeError:
                self.excluded_dir = 2
        else:
            self.excluded_dir = _dir[excluded_dir]

    def sample(self, g1=None, g2=None, kargs1=None, kargs2=None):
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
