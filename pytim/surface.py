# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

from __future__ import print_function
from abc import ABCMeta, abstractproperty
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import LinearNDInterpolator
from . import utilities
from . import messages
from .observables import LocalReferenceFrame as LocalReferenceFrame


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
        self.z = self.normal
        try:
            self.xyz = np.roll(np.array([0, 1, 2]), 2 - self.z)
            self.xy = self.xyz[0:2]
        except:
            self.xyz, self.xy = None, None
        try:
            self._layer = options['layer']
        except (TypeError, KeyError):
            self._layer = 0

    @abstractproperty
    def interpolation(self, inp):
        """ returns interpolated position on the surface """
        return utilities.extract_positions(inp)

    @abstractproperty
    def distance(self, inp, *args, **kargs):
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
        """ Compute the q-vectors compatible with the current box dimensions.

            Calculated quantities:
            q_vectors : two 2D arrays forming the grid of q-values, similar
                        to a meshgrid
            Qxy       : array of the different q-vectors
            Q         : squared module of Qxy with the first element missing
                        (no Q = 0.0)
        """
        self.box = np.roll(box, 2 - self.normal)
        nmax = list(map(int, np.ceil(self.box[0:2] / self.alpha)))
        self.q_vectors = np.mgrid[0:nmax[0], 0:nmax[1]] * 1.0
        self.q_vectors[0] *= 2. * np.pi / box[0]
        self.q_vectors[1] *= 2. * np.pi / box[1]
        self.modes_shape = self.q_vectors[0].shape
        qx = self.q_vectors[0][:, 0]
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
        QR = np.dot(self.Qxy, points[:, 0:2].T).T
        # ph[0] are the phases associated to each of the ~ n^2 modes for
        # particle 0.
        # We exclude the zero mode.
        ph = (np.cos(QR) + 1.j * np.sin(QR))[:, 1:]
        z = points[:, 2]
        az = np.mean(z)
        z = z - az
        A = (ph / self.Q)
        z = z
        s = np.dot(np.linalg.pinv(A), z)
        # return the surface modes reshaped into an array
        return np.append(az + 0.j, s / self.Q).reshape(self.modes_shape)

    def triangulate_layer_flat(self, layer=0):
        """Triangulate a layer.

        :param int layer:  (default: 0) triangulate this layer (on both sides\
                           of the interface)
        :return list triangulations:  a list of two Delaunay triangulations,\
                           which are also stored in self.surf_triang
        """
        if layer > len(self.interface._layers[0]):
            raise ValueError(messages.UNDEFINED_LAYER)

        box = self.interface.universe.dimensions[:3][self.xyz]

        upper = (self.interface._layers[0][layer])
        lower = (self.interface._layers[1][layer])
        delta = self.interface.alpha * 4.0 + 1e-6
        # here we rotate the system to have normal along z
        upperpos = utilities.generate_periodic_border(
            upper.positions[:, self.xyz], box, delta, method='2d')[0]
        lowerpos = utilities.generate_periodic_border(
            lower.positions[:, self.xyz], box, delta, method='2d')[0]

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

    def _distance_generic(self, inp, symmetry):

        inter = self.interface
        pos = utilities.extract_positions(inp)

        if len(pos) == 0:
            raise ValueError("empty group")
        box = inter.universe.dimensions[:3]
        l1centers = inter.atoms.positions[inter.atoms.layers == 1]
        # tree of the surface triangles' centers
        tree = cKDTree(l1centers, boxsize=box)

        # distances and indices of the surface triangles' centers to pos[]
        dist, ind = tree.query(pos, k=1)

        if not isinstance(inp, np.ndarray):
            # a group has been passed, we know exactly which atoms are
            # surface ones.
            try:
                dist[inp.atoms.layers == 1] = 0.0
                # Warning, the corresponding values of 'ind' will be wrong.
                # If you change this code, check that it won't
                # depend on 'ind'
            except AttributeError:
                raise RuntimeError(
                    "Wrong parameter passed to _distance_generic")
        l1centers_ = l1centers[ind]
        buried = inter.is_buried(pos)
        sign = np.ones(dist.shape[0])
        sign[np.where(buried)[0]] = -1.0
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

        distance = self.interpolation(pos)
        if np.sum(np.isnan(distance)) > 1 + int(0.01 * len(pos)):
            Warning("more than 1% of points have fallen outside"
                    "the convex hull while determining the"
                    "interpolated position on the surface."
                    "Something is wrong.")
        # positive values are outside the surface, negative inside
        cond = np.where(np.isclose(distance, 0., atol=1e-2))
        distance[cond] = 0.0
        return distance

    def _initialize_distance_interpolator_flat(self, layer):
        self._layer = layer
        self.triangulate_layer_flat(layer=self._layer)

        self._interpolator = [None, None]
        for side in [0, 1]:
            self._interpolator[side] = LinearNDInterpolator(
                self.surf_triang[side], self.triangulation_points[side][:, 2])


class SurfaceFlatInterface(Surface):
    def distance(self, inp, *args, **kargs):
        positions = utilities.extract_positions(inp)
        return self._distance_flat(positions)

    def interpolation(self, inp):
        positions = utilities.extract_positions(inp)
        box = self.interface.universe.dimensions[self.z]
        try:
            self.options['from_modes']
            upper_interp = self.surface_from_modes(upper_set, self.modes[0])
            lower_interp = self.surface_from_modes(lower_set, self.modes[1])
        except (TypeError, KeyError):
            self._initialize_distance_interpolator_flat(layer=self._layer)
            upper_interp = self._interpolator[0](positions[:, self.xy])
            lower_interp = self._interpolator[1](positions[:, self.xy])

        d1 = upper_interp - positions[:, self.z]
        d1[np.where(d1 > box / 2.)] -= box
        d1[np.where(d1 < -box / 2.)] += box

        d2 = lower_interp - positions[:, self.z]
        d2[np.where(d2 > box / 2.)] -= box
        d2[np.where(d2 < -box / 2.)] += box

        cond = np.where(np.abs(d1) <= np.abs(d2))[0]
        distance = d2
        distance[cond] = -d1[cond]

        return distance

    def dump(self):
        pass

    def regular_grid(self):
        pass

    def triangulation(self, layer=0):
        return self.triangulate_layer_flat(layer)


class SurfaceGenericInterface(Surface):
    def distance(self, inp, *args, **kargs):
        symmetry = args[0]
        try:
            mode = kargs['mode']
        except:
            mode = 'default'

        if mode == 'default':
            return self._distance_generic(inp, symmetry)

    def interpolation(self, inp):
        pass

    def dump(self):
        pass

    def regular_grid(self):
        pass

    def triangulation(self, layer=0):
        pass


#
