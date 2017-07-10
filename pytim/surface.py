# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from pytim import utilities


class Surface(object):
    """ Everything about the continuum description of surfaces.

        Any implementation must provide the following methods:

        distance()      -> returns an array of relative positions with respect to the interface
        triangulation() -> a triangulated surface, if available
        regular_grid()  -> the surface elevation on a regular grid, if available
        dump()          -> save to disk in format of choice
    """
    __metaclass__ = ABCMeta
    # TODO: so far includes only methods for CT. Gather also other ones here

    def __init__(self, interface, options=None):
        self.interface = interface
        self.normal = interface.normal
        self.alpha = interface.alpha
        self.options = options
        try:
            self._layer = options['layer']
        except:
            self._layer = 0

    # TODO: documentation
    @abstractproperty
    def interpolation(self, inp):
        """ returns interpolated position on the surface """
        positions = utilities.extract_positions(inp)

    @abstractproperty
    def distance(self, inp):
        """ returns distance from the surface """
        positions = utilities.extract_positions(inp)
        distance_array = positions[::, 2]
        return distance_array

    @abstractproperty
    def triangulation(self):
        """ return a scipy.spatial.qhull.Delaunay triangulation
            of the surface
        """
        return triangulation

    @abstractproperty
    def regular_grid(self):
        """ returns the points defining the regular grid in 2 dimensions, and
            the elevation values evaluated at the grid points
            (like the input of scipy.interpolate.RegularGridInterpolator )
        """
        return (x, y), elevations

    @abstractproperty
    def dump(self):
        """ save the surface to file """
        return True

    def update_q_vectors(self, box):
        try:
            if np.any(box != self.box):
                self._compute_q_vectors(box)
        except:
            self._compute_q_vectors(box)

    def _compute_q_vectors(self, box):
        self.box = np.roll(box, 2 - self.normal)
        nmax = map(int, np.ceil(self.box[0:2] / self.alpha))
        q_indices = np.mgrid[0:nmax[0], 0:nmax[1]]
        self.q_vectors = q_indices * 1.0
        self.q_vectors[0] *= 2. * np.pi / box[0]
        self.q_vectors[1] *= 2. * np.pi / box[1]
        self.modes_shape = self.q_vectors[0].shape
        qx = self.q_vectors[0][::, 0]
        qy = self.q_vectors[1][0]
        Qx = np.repeat(qx, len(qy))
        Qy = np.tile(qy, len(qx))
        self.Qxy = np.vstack((Qx, Qy)).T
        self.Q = np.sqrt(np.sum(self.Qxy * self.Qxy, axis=1)[1:])

    @staticmethod
    def _surface_from_modes(points, q_vectors, modes):
        elevation = []
        for point in points:
            dotp = q_vectors[0] * point[0] + q_vectors[1] * point[1]
            phase = np.cos(dotp) + 1.j * np.sin(dotp)
            elevation.append(np.sum((phase * modes).real))
        return np.array(elevation)

    def surface_from_modes(self, points, modes):
        return self._surface_from_modes(points, self.q_vectors, modes)

    def surface_modes(self, points):
        QR = np.dot(self.Qxy, points[::, 0:2].T).T
        # ph[0] is are the phases associated to each of the ~ n^2 modes for
        # particle 0.
        # We exclude the zero mode.
        ph = (np.cos(QR) + 1.j * np.sin(QR)).T[1:].T
        z = points[::, 2]
        az = np.mean(z)
        z = z - az
        A = (ph / self.Q)
        z = z
        s = np.dot(np.linalg.pinv(A), z)
        return np.append(az + 0.j, s / self.Q)

    def triangulate_layer_flat(self, layer=0):
        """Triangulate a layer.

        :param int layer:  (default: 1) triangulate this layer (on both sides\
                           of the interface)
        :return list triangulations:  a list of two Delaunay triangulations,\
                           which are also stored in self.surf_triang
        """
        if layer > len(self.interface._layers[0]):
            raise ValueError(self.UNDEFINED_LAYER)

        box = self.interface.universe.dimensions[:3]

        upper = self.interface._layers[0][layer]
        lower = self.interface._layers[1][layer]
        delta = self.interface.alpha * 4.0 + 1e-6
        upperpos = utilities.generate_periodic_border(upper.positions, box,
                                                      delta, method='2d')[0]
        lowerpos = utilities.generate_periodic_border(lower.positions, box,
                                                      delta, method='2d')[0]

        self.surf_triang = [None, None]
        self.trimmed_surf_triangs = [None, None]
        self.triangulation_points = [None, None]
        self.surf_triang[0] = Delaunay(upperpos[:, 0:2])
        self.surf_triang[1] = Delaunay(lowerpos[:, 0:2])
        self.triangulation_points[0] = upperpos[:]
        self.triangulation_points[1] = lowerpos[:]
        self.trimmed_surf_triangs[0] = utilities.trim_triangulated_surface(
            self.surf_triang[0], box
        )
        self.trimmed_surf_triangs[1] = utilities.trim_triangulated_surface(
            self.surf_triang[1], box
        )
        return self.surf_triang

    def _distance_flat(self, positions):
        box = self.interface.universe.dimensions[:3]

        pos = np.copy(positions)

        cond=np.where(pos[:,0:2]>box[0:2])
        pos[cond]-=box[cond[1]]
        cond=np.where(pos[:,0:2]<0*box[0:2])
        pos[cond]+=box[cond[1]]

        elevation = self.interpolation(pos)
        if not (np.sum(np.isnan(elevation)) == 0):
            raise Warning("Internal error: a point has fallen outside"
                          "the convex hull")
        # positive values are outside the surface, negative inside
        distance = (pos[:, 2] - elevation) * np.sign(pos[:, 2])
        return distance

    def _initialize_distance_interpolator_flat(self, layer):
        self._layer = layer
        self.triangulate_layer_flat(layer=self._layer)

        self._interpolator = [None, None]
        for side in [0, 1]:
            self._interpolator[side] = LinearNDInterpolator(
                self.surf_triang[layer],
                self.triangulation_points[layer][:, 2])
