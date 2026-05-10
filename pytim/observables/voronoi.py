# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: Voronoi
    ===============
"""

from . import Observable
import numpy as np
from scipy import spatial

from ..utilities_pbc import generate_periodic_border

try:
    import freud
    from freud.locality import Voronoi as FreudVoronoi
except ImportError:  # pragma: no cover - depends on optional dependency
    freud = None
    FreudVoronoi = None


class Voronoi(Observable):
    """Voronoi polyhedra volumes and surfaces.

    If ``freud`` is installed, the Voronoi tessellation is computed with
    ``freud.locality.Voronoi``, which handles periodic boundary conditions
    directly. If ``freud`` is not installed, the SciPy Voroni implementation
    is used, including explicit periodic border copies.
    """

    def __init__(self, universe, options=''):
        self.u = universe
        self.options = options
        self.backend = 'freud' if FreudVoronoi is not None else 'scipy'

    def _freud_box_from_mdanalysis_timestep(self, inp):
        """Return a freud Box from the current MDAnalysis timestep.

        MDAnalysis already exposes the unit cell as triclinic box vectors via
        ``Timestep.triclinic_dimensions``. Freud can consume this representation
        directly with ``freud.box.Box.from_matrix``.
        """
        try:
            matrix = inp.universe.trajectory.ts.triclinic_dimensions
        except AttributeError:
            matrix = self.u.trajectory.ts.triclinic_dimensions

        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (3, 3) or np.isclose(np.linalg.det(matrix), 0.0):
            raise ValueError("Voronoi requires a valid periodic simulation box.")

        return freud.box.Box.from_matrix(matrix)

    @staticmethod
    def _initial_info(facets, projections):
        """Create the optional information dictionary."""
        if facets or projections:
            return {
                'facet_areas': [],
                'facet_normals': [],
                'projected_areas': [],
                'projected_volumes': [],
            }
        return []

    @staticmethod
    def _finalize_info(info):
        """Match the original return type for projected quantities."""
        if isinstance(info, dict):
            try:
                info['projected_areas'] = np.array(info['projected_areas'])
                info['projected_volumes'] = np.array(info['projected_volumes'])
            except Exception:
                pass
        return info

    @staticmethod
    def _append_facet_and_projection_info(info, hull, facets, projections):
        """Append per-facet and projected data from a ConvexHull object.

        This is shared by the freud and SciPy backends. The hull simplices are
        used exactly as in the original implementation, so non-triangular cell
        faces may be represented by multiple coplanar triangular simplices.
        """
        pts = hull.points[hull.simplices]
        d1 = pts[:, 1, :] - pts[:, 0, :]
        d2 = pts[:, 2, :] - pts[:, 0, :]
        fareas = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=1)

        centroid = np.mean(hull.points, axis=0)
        fnormals = hull.equations[:, :-1]

        if facets:
            info['facet_areas'].append(fareas.tolist())
            info['facet_normals'].append(fnormals.tolist())

        if projections:
            fdistances = np.abs(
                np.sum(centroid * fnormals, axis=1) + hull.equations[:, -1]
            )
            parea = [
                float(np.sum(fareas * fnormals[:, i] ** 2))
                for i in [0, 1, 2]
            ]
            pvolume = [
                float(np.sum(fareas * fdistances * fnormals[:, i] ** 2) / 3.0)
                for i in [0, 1, 2]
            ]
            info['projected_areas'].append(parea)
            info['projected_volumes'].append(pvolume)

    def compute(self, inp, volume=True, area=True, facets=False, projections=False):
        r"""Compute the observable.

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
        - If ``freud`` is available, periodic Voronoi cells are computed without
          explicitly generating border copies.
        - If ``freud`` is not available, the original SciPy implementation is
          used as a fallback.
        - The optional facet/projection information is still computed through
          ``scipy.spatial.ConvexHull`` in both backends.
        - The projected area/volume decomposition uses the squared direction
          cosine of each facet normal, so that the three components sum to the
          total area or volume.
        """
        if self.backend == 'freud':
                return self._compute_freud(inp, volume, area, facets, projections)
        if self.backend == 'scipy':
            return self._compute_scipy(inp, volume, area, facets, projections)

        raise ValueError("Unknown Voronoi backend {!r}".format(self.backend))

    def _compute_freud(self, inp, volume=True, area=True, facets=False, projections=False):
        """Compute Voronoi cells using freud.locality.Voronoi."""
        fbox = self._freud_box_from_mdanalysis_timestep(inp)

        # freud boxes are centered at the origin. Wrap the coordinates into
        # Freud's conventional half-open box before tessellating.
        points = fbox.wrap(np.asarray(inp.positions, dtype=np.float32))

        self.voronoi = FreudVoronoi()
        self.voronoi.compute((fbox, points))

        volumes = []
        areas = []
        info = self._initial_info(facets, projections)

        if volume:
            volumes = self.voronoi.volumes.tolist()

        if area:
            nlist = self.voronoi.nlist
            areas = np.bincount(
                nlist.query_point_indices,
                weights=nlist.weights,
                minlength=len(points),
            ).tolist()

        if facets or projections:
            for vertices in self.voronoi.polytopes:
                hull = spatial.ConvexHull(np.asarray(vertices, dtype=float))
                self._append_facet_and_projection_info(
                    info, hull, facets, projections
                )

        info = self._finalize_info(info)
        return [val for val in [volumes, areas, info] if len(val) > 0]

    def _compute_scipy(self, inp, volume=True, area=True, facets=False, projections=False):
        """SciPy implementation with explicit periodic border copies."""
        points, box = inp.positions, inp.dimensions[:3]
        xpoints, ids = generate_periodic_border(points, box, box, method='3d')

        # Compute Voronoi tessellation.
        self.voronoi = spatial.Voronoi(xpoints)

        volumes = []
        areas = []
        info = self._initial_info(facets, projections)

        # Keep only the regions belonging to the original input points.
        for region_id in self.voronoi.point_region[:len(points)]:
            region = self.voronoi.regions[region_id]
            if -1 in region:
                raise ValueError(
                    'There are open boundaries in the voronoi diagram. '
                    'Choose a larger skin for the inclusion of periodic copies.'
                )

            vertices = np.asarray(self.voronoi.vertices[region])
            hull = spatial.ConvexHull(vertices)

            if volume:
                volumes.append(hull.volume)
            if area:
                areas.append(hull.area)
            if facets or projections:
                self._append_facet_and_projection_info(
                    info, hull, facets, projections
                )

        info = self._finalize_info(info)
        return [val for val in [volumes, areas, info] if len(val) > 0]
