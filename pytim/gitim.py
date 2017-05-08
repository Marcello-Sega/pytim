#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: gtim
    ============
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import distance
import itertools
from pytim import utilities
import pytim


class GITIM(pytim.PYTIM):
    """Identifies interfacial molecules at macroscopically flat interfaces.

        :param Universe universe:      the MDAnalysis universe
        :param float alpha:            the probe sphere radius
        :param AtomGroup itim_group:   identify the interfacial molecules\
                                       from this group
        :param dict radii_dict:        dictionary with the atomic radii of\
                                       the elements in the itim_group.
                                       If None is supplied, the default one\
                                       (from GROMOS 43a1) will be used.
        :param int max_layers:         the number of layers to be identified
        :param bool info:              print additional info
        :param bool multiproc:         parallel version (default: True. \
                                       Switch off for debugging)

        Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from   pytim.datafiles import *
        >>>
        >>> u       = mda.Universe(MICELLE_PDB)
        >>> g       = u.select_atoms('resname DPC')
        >>> radii=pytim_data.vdwradii(G43A1_TOP)
        >>>
        >>> interface =pytim.GITIM(u,itim_group=g,molecular=False,\
                symmetry='spherical',alpha=2.5)
        >>> layer = interface.layers[0]
        >>> interface.writepdb('gitim.pdb',centered=False)
        >>> print repr(layer)
        <AtomGroup with 558 atoms>

    """

    def __init__(
            self,
            universe,
            alpha=2.0,
            symmetry='spherical',
            normal='guess',
            itim_group=None,
            radii_dict=None,
            max_layers=1,
            cluster_cut=None,
            cluster_threshold_density=None,
            molecular=True,
            extra_cluster_groups=None,
            info=False,
            multiproc=True):

        self._basic_checks(universe)

        self.universe = universe
        self.cluster_threshold_density = cluster_threshold_density
        self.alpha = alpha
        self.max_layers = max_layers
        self._layers = np.empty([max_layers], dtype=type(universe.atoms))
        self.info = info
        self.normal = None
        self.PDB = {}
        self.molecular = molecular

        self.cluster_cut = cluster_cut
        self.extra_cluster_groups = extra_cluster_groups
        self.itim_group = itim_group

        self._define_groups()

        self._assign_symmetry(symmetry)

        if(self.symmetry == 'planar'):
            self._assign_normal(normal)

        self.assign_radii(radii_dict)
        self.sanity_checks()

        self.grid = None
        self.use_threads = False
        self.use_kdtree = True
        self.use_multiproc = multiproc

        pytim.PatchTrajectory(universe.trajectory, self)
        self._assign_layers()

    def _assign_symmetry(self, symmetry):
        if self.itim_group is None:
            raise TypeError(self.UNDEFINED_ITIM_GROUP)
        if symmetry == 'guess':
            raise ValueError( "symmetry 'guess' To be implemented")
        else:
            if not (symmetry in self.symmetry_dict):
                raise ValueError(self.WRONG_DIRECTION)
            self.symmetry = symmetry

    def sanity_checks(self):
        """ Basic checks to be performed after the initialization.

            We test them also here in the docstring:

            >>> import pytim 
            >>> import MDAnalysis as mda
            >>> u = mda.Universe(pytim.datafiles.WATER_GRO)
            >>>
            >>> pytim.GITIM(u,alpha=-1.0)
            Traceback (most recent call last):
            ...
            ValueError: parameter alpha must be positive

            >>> pytim.GITIM(u,alpha=-1000000)
            Traceback (most recent call last):
            ...
            ValueError: parameter alpha must be smaller than the smaller box side

        """
        if self.alpha < 0:
            raise ValueError(self.ALPHA_NEGATIVE)
        if self.alpha >= np.amin(self.universe.dimensions[:3]):
            raise ValueError(self.ALPHA_LARGE)

        if(self.cluster_cut is not None):
            elements = len(self.cluster_cut) 
            try:
                extraelements = len(self.extra_cluster_groups) 
            except TypeError:
                extraelements = -1
            if  not (elements == 1 or elements == 1 + extraelements):
                raise  StandardError(self.MISMATCH_CLUSTER_SEARCH)
        else:
            if self.extra_cluster_groups is not None:
                raise ValueError(self.UNDEFINED_CLUSTER_SEARCH)

    @staticmethod
    def alpha_prefilter(triangulation, alpha):
        t = triangulation
        threshold = 2. * alpha
        return t.simplices[[np.max(distance.cdist(t.points[simplex],
                                                  t.points[simplex],
                                                  'euclidean')) >=
                            threshold + 2. * np.min(t.radii[simplex])
                            for simplex in t.simplices]]

    def circumradius(self, simplex):

        points = self.triangulation.points
        radii = self.triangulation.radii

        R = []
        r_i = points[simplex]
        rad_i = radii[simplex]
        d = (rad_i[0] - rad_i)[1:]
        M = (r_i[0] - r_i)[1:]

        # r_i2 = np.sum(r_i**2, axis=1)
        # d_2 = d**2
        # s = ((r_i2[0] - r_i2[1:] - d_2[0] + d_2)) / 2.

        try:
            u = np.dot(np.linalg.inv(M), d)
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.message:
                print "Warning, singular matrix for ", r_i
                # TODO is this correct? The singular matrix most likely comes
                # out of points alinged in the plane
                return 0
            else:
                raise
        v = r_i[1] - r_i[0]

        A = - (rad_i[0] - np.dot(u, v))
        B = np.linalg.norm(rad_i[0] * u + v)
        C = 1 - np.sum(u**2)
        R.append((A + B) / C)
        R.append((A - B) / C)
        R = np.array(R)
        positive_rad = R[R >= 0]
        if positive_rad.size == 1:
            return np.min(positive_rad)
        else:
            return 0.0

    def alpha_shape(self, alpha):
        # print  utilities.lap()
        box = self.universe.dimensions[:3]
        delta = 2. * self.alpha + 1e-6
        points = self.cluster_group.positions[:]
        nrealpoints = len(points)
        extrapoints, extraids = utilities.generate_periodic_border_3d(
            points, box, delta
        )
        # add points at the vertices of the expanded (by 2 alpha) box
        for dim in range(8):
                # [0,0,0],[0,0,1],[0,1,0],...,[1,1,1]
            tmp = np.array(
                np.array(
                    list(
                        np.binary_repr(
                            dim,
                            width=3)),
                    dtype=np.int8),
                dtype=np.float)
            tmp *= (box + delta)
            # tmp += (np.random.random(3)-0.5)*box*1e-8 # the random gitter
            # (rescaled to be small wrt the box) is added to prevent coplanar
            # points
            tmp[tmp < box / 2.] -= delta
            tmp = np.reshape(tmp, (1, 3))
            extrapoints = np.append(extrapoints, tmp, axis=0)
            extraids = np.append(extraids, -1)

        # print utilities.lap()
        self.triangulation = Delaunay(extrapoints)
        self.triangulation.radii = np.append(
            self.cluster_group.radii[extraids[extraids >= 0]], np.zeros(8))
        # print utilities.lap()

        prefiltered = self.alpha_prefilter(self.triangulation, alpha)

        # print utilities.lap()
        a_shape = prefiltered[[self.circumradius(
            simplex) >= self.alpha for simplex in prefiltered]]

        # print utilities.lap()
        _ids = np.unique(a_shape.flatten())

        # remove the indices corresponding to the 8 additional points, which
        # have extraid==-1
        ids = _ids[np.logical_and(_ids >= 0, _ids < nrealpoints)]

        # print utilities.lap()
        return ids

    def _assign_layers(self):
        """Determine the GITIM layers."""
        # this can be used later to shift back to the original shift
        self.original_positions = np.copy(self.universe.atoms.positions[:])
        self.universe.atoms.pack_into_box()

        if(self.cluster_cut is not None):
            # groups have been checked already in sanity_checks()
            labels, counts, _ = utilities.do_cluster_analysis_DBSCAN(
                self.itim_group, self.cluster_cut[0],
                self.universe.dimensions[:6],
                self.cluster_threshold_density, self.molecular)
            labels = np.array(labels)
            # the label of atoms in the largest cluster
            label_max = np.argmax(counts)
            # the indices (within the group) of the
            ids_max = np.where(labels == label_max)[0]
            # atoms belonging to the largest cluster
            self.cluster_group = self.itim_group[ids_max]

        else:
            self.cluster_group = self.itim_group

        if self.symmetry == 'planar':
            utilities.centerbox(self.universe, center_direction=self.normal)
            self.center(self.cluster_group, self.normal)
            utilities.centerbox(self.universe, center_direction=self.normal)
        if self.symmetry == 'spherical':
            self.center(self.cluster_group, 'x', halfbox_shift=False)
            self.center(self.cluster_group, 'y', halfbox_shift=False)
            self.center(self.cluster_group, 'z', halfbox_shift=False)
            self.universe.atoms.pack_into_box(self.universe.dimensions[:3])

        # first we label all atoms in itim_group to be in the gas phase
        self.itim_group.atoms.bfactors = 0.5
        # then all atoms in the larges group are labelled as liquid-like
        self.cluster_group.atoms.bfactors = 0

        size = len(self.cluster_group.positions)
        self._seen = np.zeros(size, dtype=np.int8)

        alpha_ids = self.alpha_shape(self.alpha)

        # only the 1st layer is implemented in gitim so far
        if self.molecular:
            self._layers[0] = self.cluster_group[alpha_ids].residues.atoms
        else:
            self._layers[0] = self.cluster_group[alpha_ids]

        for layer in self._layers:
            layer.bfactors = 1

        # reset the interpolator
        self._interpolator = None

    @property
    def layers(self):
        """Access the layers as numpy arrays of AtomGroups.

        The object can be sliced as usual with numpy arrays.

        """
        return self._layers

#
