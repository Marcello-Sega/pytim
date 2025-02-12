# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: Orientation
    ===================
"""

from . import Observable
import numpy as np
from MDAnalysis.core.groups import Atom, AtomGroup
from scipy import spatial
from ..utilities_pbc import generate_periodic_border
import itertools

class Voronoi(Observable):
    """ Voronoi polyhedra volumes and surfaces
    """

    def __init__(self, universe, options=''):
        self.u = universe
        self.options = options

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group. 
        :returns: 
          volumes: ndarray of Voronoi polyhedra volumes
          areas  : ndarray of Voronoi polyhedra surface areas

        """

        points, box  = inp.positions, inp.dimensions[:3]
        xpoints, ids = generate_periodic_border(points, box, box / 2, method='3d')
        
        # Compute Voronoi tessellation
        self.voronoi = spatial.Voronoi(xpoints)

        volumes = []
        areas = []

        for region in self.voronoi.point_region[:len(points)]:
            vertices = np.asarray(self.voronoi.vertices[self.voronoi.regions[region]])
            convex_hull = spatial.ConvexHull(vertices)
            volumes.append(convex_hull.volume)
            areas.append(convex_hull.area)

        return np.asarray(volumes), np.asarray(areas)


