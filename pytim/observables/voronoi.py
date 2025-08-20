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
            tuple (volumes, areas, info)

            - **volumes** (ndarray): total volume(s) of the polyhedron(s).
            - **areas** (ndarray): total surface area(s).
            - **info** (dict, optional): contains additional per-facet and per-axis data:

                * ``facet_areas`` – list of facet areas :math:`A_f`
                * ``facet_normals`` – list of outward unit normals :math:`n_f`
                * ``projected_areas`` – per-axis contributions

                  .. math::

                     A_i = \sum_f A_f (n_f \cdot e_i)^2

                  where :math:`e_i` is the Cartesian unit vector in direction
                  :math:`i \in \{x, y, z\}`

                * ``projected_volumes`` – per-axis contributions

                  .. math::

                     V_i = \sum_f \frac{A_f h_f}{3} (n_f \cdot e_i)^2

                  with :math:`h_f` the height of the pyramid defined by facet
                  :math:`f` and the chosen reference point.

        Notes
        -----
        - The projected area/volume decomposition uses the squared direction cosine
          of each facet normal, so that the three components sum to the total area
          or volume.


        Example:

        >>> import numpy as np
        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import WATER_GRO
        >>> from pytim.observables import Voronoi
        >>> def rprint(x,n=3): print(np.around(x,n))
        >>> 
        >>> u = mda.Universe(WATER_GRO)
        >>> ox = u.select_atoms('name OW')
        >>> # just a slice to compute the observable faster
        >>> ox = ox [ox.positions[:,2] < u.dimensions[2]/10]
        >>> voro = Voronoi(u)
        >>> volumes, areas, dic  = voro.compute(ox, facets=True,projections=True)
        >>> rprint([volumes[0],areas[0]])
        [28.708 54.081]

        >>> print(dic.keys())
        dict_keys(['facet_areas', 'facet_normals', 'projected_areas', 'projected_volumes'])

        >>> # The first atom's polyhedra has 48 facets.
        >>> # The projected areas and volumes are the
        >>> # amount of area pointing along x,y,z
        >>> print([len(dic[k][0]) for k in dic.keys()])
        [48, 48, 3, 3]

        >>> rprint(dic['projected_areas'][0])
        [17.962 18.882 17.237]

        >>> rprint(dic['projected_volumes'][0])
        [9.803 9.877 9.029]

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

