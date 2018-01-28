# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: LayerTriangulation
    ==========================
"""

from __future__ import print_function
from . import Observable
import numpy as np
from pytim import utilities


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
                 return_triangulation=True,
                 return_statistics=True):

        Observable.__init__(self, interface.universe)
        self.interface = interface
        self.return_triangulation = return_triangulation
        self.return_statistics = return_statistics

    def _return_stats(self, stats, surface):
        if self.return_triangulation is False:
            return stats
        else:
            return [
                stats, surface.surf_triang, surface.triangulation_points,
                surface.trimmed_surf_triangs
            ]

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

        return self._return_stats(stats, surface)
