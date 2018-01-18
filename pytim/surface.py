# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

from __future__ import print_function
from abc import ABCMeta, abstractproperty
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import LinearNDInterpolator
from pytim import utilities
from . import messages


class Surface(object):
    """ Everything about the continuum description of surfaces.

        Any implementation must provide the following methods:

        distance()      -> returns an array of relative positions with respect
                           to the interface
        triangulation() -> a triangulated surface, if available
        regular_grid()  -> the surface elevation on a regular grid, if
                           available
        dump()          -> save to disk in format of choice
    """
    __metaclass__ = ABCMeta

    def __init__(self, interface, options=None):
        self.interface = interface
        self.normal = interface.normal
        self.alpha = interface.alpha
        self.options = options
        try:
            self._layer = options['layer']
        except (TypeError, KeyError):
            self._layer = 0

    @abstractproperty
    def interpolation(self, inp):
        """ returns interpolated position on the surface """
        return utilities.extract_positions(inp)

    @abstractproperty
    def distance(self, inp, *kargs):
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
        except BaseException:
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

        :param int layer:  (default: 0) triangulate this layer (on both sides\
                           of the interface)
        :return list triangulations:  a list of two Delaunay triangulations,\
                           which are also stored in self.surf_triang
        """
        if layer > len(self.interface._layers[0]):
            raise ValueError(messages.UNDEFINED_LAYER)

        box = self.interface.universe.dimensions[:3]

        upper = self.interface._layers[0][layer]
        lower = self.interface._layers[1][layer]
        delta = self.interface.alpha * 4.0 + 1e-6
        upperpos = utilities.generate_periodic_border(
            upper.positions, box, delta, method='2d')[0]
        lowerpos = utilities.generate_periodic_border(
            lower.positions, box, delta, method='2d')[0]

        self.surf_triang = [None, None]
        self.trimmed_surf_triangs = [None, None]
        self.triangulation_points = [None, None]
        self.surf_triang[0] = Delaunay(upperpos[:, 0:2])
        self.surf_triang[1] = Delaunay(lowerpos[:, 0:2])
        self.triangulation_points[0] = upperpos[:]
        self.triangulation_points[1] = lowerpos[:]
        self.trimmed_surf_triangs[0] = utilities.trim_triangulated_surface(
            self.surf_triang[0], box)
        self.trimmed_surf_triangs[1] = utilities.trim_triangulated_surface(
            self.surf_triang[1], box)
        return self.surf_triang

    @staticmethod
    def local_env_com(positions, reference_pos, box, nneigh):
        tree = cKDTree(positions, boxsize=box)
        # local_env are the positions of the atoms next to pos
        _, local_env_indices = tree.query(reference_pos, k=nneigh)
        local_env = positions[local_env_indices].copy()
        for k in range(nneigh):
            local_env[:, k, :] = utilities.pbc_compact(local_env[:, k, :],
                                                       reference_pos, box)
        return np.average(local_env, axis=1)

    def _distance_generic(self, positions, symmetry):

        intr = self.interface
        pos = positions
        if len(pos) == 0:
            raise ValueError("empty group")
        box = intr.universe.dimensions[:3]
        tri = utilities.find_surface_triangulation(intr)
        # positions of the points in the triangulated surface
        l1pos = intr.triangulation.points[tri]
        # their baricenters
        l1centers = np.average(l1pos, axis=1)
        # tree of the surface triangles' centers
        tree = cKDTree(l1centers, boxsize=box)

        # distances and indices of the surface triangles' centers to pos[]
        dist, ind = tree.query(pos, k=1)
        # we need now the position of the nearest triangle's centers
        # selected above.
        l1centers = l1centers[ind]

        if symmetry == 'generic':
            return dist
        if symmetry == 'spherical':
            # tree of all the atoms in cluster_group
            COM = self.local_env_com(intr.cluster_group.positions, l1centers,
                                     box, 5)
            P1 = utilities.pbc_compact(pos, l1centers, box) - l1centers
            P2 = utilities.pbc_compact(COM, l1centers, box) - l1centers
            sign = -np.sign(np.sum(P1 * P2, axis=1))
            return sign * dist
        raise ValueError("Incorrect symmetry used for distance calculation")

    def _distance_spherical(self, positions):
        box = self.interface.universe.dimensions[:3]
        cond = self.interface.atoms.layers == 1
        tree = cKDTree(self.interface.atoms.positions[cond], boxsize=box)
        d, i = tree.query(positions, k=1)
        dd, surf_neighs = tree.query(self.interface.atoms.positions[cond][i],
                                     5)
        p = self.interface.atoms.positions[cond]
        center = np.mean(p)

        side = (np.sum((positions - center)**2, axis=1) > np.sum(
            (p - center)**2, axis=1)) * 2 - 1
        return d * side

    def _distance_flat(self, positions):
        box = self.interface.universe.dimensions[:3]

        pos = np.copy(positions)

        cond = np.where(pos[:, 0:2] > box[0:2])
        pos[cond] -= box[cond[1]]
        cond = np.where(pos[:, 0:2] < 0 * box[0:2])
        pos[cond] += box[cond[1]]

        elevation = self.interpolation(pos)
        if np.sum(np.isnan(elevation)) > 1 + int(0.01 * len(pos)):
            Warning("more than 1% of points have fallen outside"
                    "the convex hull while determining the"
                    "interpolated position on the surface."
                    "Something is wrong.")
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


class SurfaceFlatInterface(Surface):
    def distance(self, inp, *args):
        positions = utilities.extract_positions(inp)
        return self._distance_flat(positions)

    def interpolation(self, inp):
        positions = utilities.extract_positions(inp)
        upper_set = positions[positions[:, 2] >= 0]
        lower_set = positions[positions[:, 2] < 0]

        elevation = np.zeros(len(positions))

        try:
            self.options['from_modes']
            upper_interp = self.surface_from_modes(upper_set, self.modes[0])
            lower_interp = self.surface_from_modes(lower_set, self.modes[1])
        except (TypeError, KeyError):
            self._initialize_distance_interpolator_flat(layer=self._layer)
            upper_interp = self._interpolator[0](upper_set[:, 0:2])
            lower_interp = self._interpolator[1](lower_set[:, 0:2])

        elevation[np.where(positions[:, 2] >= 0)] = upper_interp
        elevation[np.where(positions[:, 2] < 0)] = lower_interp
        return elevation

    def dump(self):
        pass

    def regular_grid(self):
        pass

    def triangulation(self, layer=0):
        return self.triangulate_layer_flat(layer)


class SurfaceGenericInterface(Surface):
    def distance(self, inp, *args):
        symmetry = args[0]
        positions = utilities.extract_positions(inp)
        return self._distance_generic(positions, symmetry)

    def interpolation(self, inp):
        pass

    def dump(self):
        pass

    def regular_grid(self):
        pass

    def triangulation(self, layer=0):
        pass


#
