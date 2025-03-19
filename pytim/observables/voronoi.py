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

    def compute(self, inp, volume=True, area=True, facets=False,  projections=False):
        """Compute the observable.

        :param AtomGroup inp:  the input atom group.
        :param bool volume     :  compute the volumes
        :param bool area       :  compute the areas
        :param bool facets     :  compute facets areas and normals
        :param bool projections:  compute projected areas and volumes
        :returns:
          a tuple with volumes (ndarray), areas (ndarray) and facets (dictionary) and projections (dictionary)
          as selected by the corresponding options.

          The arrays volumes and areas have shape equal to len(inp)

          The facets dictionary has keys 'facet_areas' and 'facet_normals', and the corresponding values are
          lists of length len(inp), each one containing a list of variable length, depending on the number of
          facet associated to the point.

          The projections dictionary has keys 'projected_areas' and 'projected_volumes', and the corresponding values
          are ndarrays of shape (len(inp),3 ) for each of the three Cartesian directions x,y,z.

       """

        points, box  = inp.positions, inp.dimensions[:3]
        xpoints, ids = generate_periodic_border(points, box, box, method='3d')

        # Compute Voronoi tessellation
        self.voronoi = spatial.Voronoi(xpoints)

        volumes = []
        areas = []
        if facets or projections : dictionary = {'facet_areas':[],'facet_normals':[], 'projected_areas':[],'projected_volumes':[]}
        else: dictionary = []

        # Keep only the regions in the
        for p,region_id in enumerate(self.voronoi.point_region[:len(points)]):
            region = self.voronoi.regions[region_id]
            if -1 in region: raise ValueError('There are open boundaries in the voronoi diagram. Choose a larger skin for the inclusion of periodic copies.')
            vertices = np.asarray(self.voronoi.vertices[region])
            hull = spatial.ConvexHull(vertices)
            if volume: volumes.append(hull.volume) # total volume of polyhedron associated to point p
            if area:   areas.append(hull.area)     # total area
            if facets or projections:
                pts = hull.points[hull.simplices]
                d1 = pts[:, 1, :] - pts[:, 0, :]
                d2 = pts[:, 2, :] - pts[:, 0, :]
                fareas = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=1)
                centroid =  np.mean(hull.points, axis=0)
                fnormals = hull.equations[:,:-1]
                if facets:
                    dictionary['facet_areas'].append(fareas.tolist())     # area of each facet
                    dictionary['facet_normals'].append(fnormals.tolist()) # normal of each facet
                if projections:
                    fdistances = np.abs(np.sum(centroid*fnormals,axis=1) + hull.equations[:,-1])
                    pvolume = [float(np.sum(fareas* fdistances * fnormals[:,i]**2)/3.) for i in [0,1,2]]
                    parea = [float(np.sum(fareas* fnormals[:,i]**2)) for i in [0,1,2]]
                    dictionary['projected_areas'].append(parea)
                    dictionary['projected_volumes'].append(pvolume)
        try:
            dictionary['projected_areas'] = np.array(dictionary['projected_areas'])
            dictionary['projected_volumes'] = np.array(dictionary['projected_volumes'])
        except:
            pass
        return [val for val in [volumes, areas, dictionary] if len(val) > 0 ]

