#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: gtim
    ============
"""

import numpy as np
from scipy.spatial import distance
from pytim import utilities
import pytim
from pytetgen import Delaunay

class GITIM(pytim.PYTIM):
    """Identifies interfacial molecules at macroscopically flat interfaces.

        :param Universe universe:      the MDAnalysis universe
        :param float alpha:            the probe sphere radius
        :param AtomGroup group:        identify the interfacial molecules\
                                       from this group
        :param dict radii_dict:        dictionary with the atomic radii of\
                                       the elements in the group.
                                       If None is supplied, the default one\
                                       (from GROMOS 43a1) will be used.
        :param int max_layers:         the number of layers to be identified
        :param bool info:              print additional info

        Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from   pytim.datafiles import *
        >>>
        >>> u       = mda.Universe(MICELLE_PDB)
        >>> g       = u.select_atoms('resname DPC')
        >>> radii=pytim_data.vdwradii(G43A1_TOP)
        >>>
        >>> interface =pytim.GITIM(u,group=g,molecular=False,\
                symmetry='spherical',alpha=2.5)
        >>> layer = interface.layers[0]
        >>> interface.writepdb('gitim.pdb',centered=False)
        >>> print repr(layer)
        <AtomGroup with 547 atoms>

    """
    _surface = None

    def __init__(
            self,
            universe,
            alpha=2.0,
            symmetry='spherical',
            normal='guess',
            group=None,
            radii_dict=None,
            max_layers=1,
            cluster_cut=None,
            cluster_threshold_density=None,
            molecular=True,
            extra_cluster_groups=None,
            info=False,
            centered=False,
            warnings=False,
            _noextrapoints=False,
            **kargs):

        # this is just for debugging/testing
        self._noextrapoints = _noextrapoints
        self.do_center = centered
        sanity = pytim.SanityCheck(self)
        sanity.assign_universe(
            universe, radii_dict=radii_dict, warnings=warnings)
        sanity.assign_alpha(alpha)

        self.cluster_threshold_density = cluster_threshold_density
        self.max_layers = max_layers
        self._layers = np.empty([max_layers], dtype=type(universe.atoms))
        self.info = info
        self.normal = None
        self.PDB = {}
        self.molecular = molecular
        sanity.assign_groups(group, cluster_cut, extra_cluster_groups)
        sanity.assign_radii()

        self._assign_symmetry(symmetry)

        if(self.symmetry == 'planar'):
            sanity.assign_normal(normal)

        pytim.PatchTrajectory(universe.trajectory, self)

        self._assign_layers()

        self._atoms = self.LayerAtomGroupFactory(
            self._layers[:].sum().indices, self.universe)

    def _sanity_checks(self):
        """ Basic checks to be performed after the initialization.

        """

    @staticmethod
    def alpha_prefilter(triangulation, alpha):
        t = triangulation
        threshold = 2. * alpha
        return t.simplices[ np.array([np.max(distance.cdist(t.points[simplex],
                                                  t.points[simplex],
                                                  'euclidean')) >=
                            threshold + 2. * np.min(t.radii[simplex])
                            for simplex in t.simplices])]

    def circumradius(self, simplex):

        points = self.triangulation.points
        radii = self.triangulation.radii
    
        R = []
        r_i = points[simplex]
        rad_i = radii[simplex]

        d = (rad_i[0] - rad_i[1:])
        M = (r_i[0] - r_i[1:])

        r_i2 = np.sum(r_i**2, axis=1)
        rad_i2 = rad_i**2
        s = (r_i2[0] - r_i2[1:] - rad_i2[0] + rad_i2[1:]) / 2.
        try:
            invM = np.linalg.inv(M)
            u = np.dot(invM, d)
            v = r_i[0] - np.dot(invM, s)
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.message:
                print "Warning, singular matrix for ", r_i
                # TODO is this correct? The singular matrix most likely comes
                # out of points alinged in the plane
                return 0
            else:
                raise RuntimeError(err.message)

        u2 = np.sum(u**2)
        v2 = np.sum(v**2)
        uv = np.sum(u * v)
        A = (rad_i[0] - uv)
        arg = (rad_i[0] - uv)**2 - (u2 - 1) * (v2 - rad_i2[0])
        if arg < 0:
            return 0.0
        B = np.sqrt(arg)
        C = u2 - 1
        R.append((A + B) / C)
        R.append((A - B) / C)
        r_i = np.roll(r_i, 1)
        rad_i = np.roll(rad_i, 1)

        R = np.array(R)
        if R[0] < 0 and R[1] < 0:
            return 0.0
        return np.min(R[R >= 0])

    def alpha_shape(self, alpha):
        # print  utilities.lap()
        box = self.universe.dimensions[:3]
        delta = 2. * self.alpha + 1e-6
        points = self.cluster_group.positions[:]
        nrealpoints = len(points)
        np.random.seed(0)  # pseudo-random for reproducibility
        gitter = (np.random.random(3 * 8).reshape(8, 3)) * 1e-9
        if self._noextrapoints == False:
            extrapoints, extraids = utilities.generate_periodic_border(
                points, box, delta, method='3d'
            )
        else:
            extrapoints = np.copy(points)
            extraids = np.arange(len(points), dtype=np.int)
        # add points at the vertices of the expanded (by 2 alpha) box
        cube_vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], 
                                  [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], 
                                  [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        if self._noextrapoints == False:
            for dim,vertex in enumerate(cube_vertices):
                vertex = vertex * box + delta + gitter[dim] # added to prevent coplanar points
                vertex [vertex < box / 2.] -= 2*delta
                vertex = np.reshape(vertex, (1, 3))
                extrapoints = np.append(extrapoints, vertex, axis=0)
                extraids = np.append(extraids, -1)
        # print utilities.lap()
        self.triangulation = Delaunay(extrapoints)
        self.triangulation.radii = np.append(
            self.cluster_group.radii[extraids[extraids >= 0]], np.zeros(8))
        # print utilities.lap()

        prefiltered = self.alpha_prefilter(self.triangulation, alpha)
        # print utilities.lap()

        a_shape = prefiltered[np.array([self.circumradius(
            simplex) >= self.alpha for simplex in prefiltered])]
        # print utilities.lap()
        _ids = np.unique(a_shape.flatten())
        # remove the indices corresponding to the 8 additional points, which
        # have extraid==-1
        ids = _ids[np.logical_and(_ids >= 0, _ids < nrealpoints)]

        return ids

    def _assign_layers(self):
        """Determine the GITIM layers."""
        # this can be used later to shift back to the original shift
        self.original_positions = np.copy(self.universe.atoms.positions[:])
        self.universe.atoms.pack_into_box()

        self._define_cluster_group()

        self.centered_positions = None
        if self.do_center:
            self.center()

        # first we label all atoms in group to be in the gas phase
        self.label_group(self.itim_group.atoms, 0.5)
        # then all atoms in the larges group are labelled as liquid-like
        self.label_group(self.cluster_group.atoms, 0.0)

        size = len(self.cluster_group.positions)

        alpha_ids = self.alpha_shape(self.alpha)

        # only the 1st layer is implemented in gitim so far
        if self.molecular:
            self._layers[0] = self.cluster_group[alpha_ids].residues.atoms
        else:
            self._layers[0] = self.cluster_group[alpha_ids]

        for layer in self._layers:
            self.label_group(layer, 1.0)

        # reset the interpolator
        self._interpolator = None

    @property
    def layers(self):
        """Access the layers as numpy arrays of AtomGroups.

        The object can be sliced as usual with numpy arrays.

        """
        return self._layers

#
