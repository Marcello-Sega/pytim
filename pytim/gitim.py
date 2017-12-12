#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: gitim
    =============
"""

import numpy as np
from scipy.spatial import distance
from pytim import utilities
import pytim
try:
    from pytetgen import Delaunay
except:
    from scipy.spatial import Delaunay

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
        >>>
        >>> interface =pytim.GITIM(u,group=g,molecular=False,\
                symmetry='spherical',alpha=2.5)
        >>> layer = interface.layers[0]
        >>> interface.writepdb('gitim.pdb',centered=False)
        >>> print repr(layer)
        <AtomGroup with 793 atoms>


        Successive layers can be identified with GITIM as well. In this example we
        identify two solvation shells of glucose


        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from   pytim.datafiles import *
        >>>
        >>> u       = mda.Universe(GLUCOSE_PDB)
        >>> g       = u.select_atoms('name OW')
        >>> # it is faster to consider only oxygens.
        >>> # Hydrogen atoms are anyway within Oxygen's radius,
        >>> # in SPC* models.
        >>> interface =pytim.GITIM(u, group=g, molecular=True, \
                    symmetry='spherical', alpha=2.0, max_layers=2)
        >>>
        >>> interface.writepdb('glucose_shells.pdb')
        >>> print repr(interface.layers[0]),repr(interface.layers[1])
        <AtomGroup with 54 atoms> <AtomGroup with 117 atoms>



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
            biggest_cluster_only = False,
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

        self.biggest_cluster_only = biggest_cluster_only
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
        sanity.check_multiple_layers_options()
        sanity.assign_radii()

        self._assign_symmetry(symmetry)

        if(self.symmetry == 'planar'):
            sanity.assign_normal(normal)

        pytim.PatchTrajectory(self.universe.trajectory, self)

        self._assign_layers()

    def _sanity_checks(self):
        """ Basic checks to be performed after the initialization.

        """

    @staticmethod
    def alpha_prefilter(triangulation, alpha):
        t = triangulation
        threshold = 2.0 * alpha
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

    def alpha_shape(self, alpha, group):
        # print  utilities.lap()
        box = self.universe.dimensions[:3]
        delta = 2.1 * self.alpha + 1e-6
        points = group.positions[:]
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
            group.radii[extraids[extraids >= 0]], np.zeros(8))
        # print utilities.lap()

        #prefiltered = self.alpha_prefilter(self.triangulation, alpha)
        prefiltered = self.triangulation.simplices # == skip prefiltering
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
        self.label_group(self.universe.atoms, beta=0.0, layer=-1, cluster=-1, side=-1)
        self.original_positions = np.copy(self.universe.atoms.positions[:])
        self.universe.atoms.pack_into_box()

        self._define_cluster_group()

        self.centered_positions = None
        if self.do_center:
            self.center()

        # first we label all atoms in group to be in the gas phase
        self.label_group(self.itim_group.atoms, beta=0.5)
        # then all atoms in the larges group are labelled as liquid-like
        self.label_group(self.cluster_group.atoms, beta=0.0)

        alpha_group = self.cluster_group[:]

        # TODO the successive layers analysis should be done by removing points from the triangulation
        # and updating the circumradius of the neighbors of the removed points only.
        for layer in range(0,self.max_layers):

            alpha_ids = self.alpha_shape(self.alpha,alpha_group)

            group = alpha_group[alpha_ids]

            if self.biggest_cluster_only == True: # apply the same clustering algorith as set at init
                l,c,_ = utilities.do_cluster_analysis_DBSCAN(group,self.cluster_cut[0], self.universe.dimensions[:],
                                                             threshold_density=self.cluster_threshold_density,
                                                             molecular=self.molecular)
                group = group [ np.where(np.array(l) == np.argmax(c))[0] ]

            alpha_group = alpha_group[:] - group[:]

            if self.molecular:
                self._layers[layer]  = group.residues.atoms
            else:
                self._layers[layer] = group

            self.label_group(self._layers[layer], beta = 1.*(layer+1), layer = (layer+1) )

        # reset the interpolator
        self._interpolator = None

    @property
    def layers(self):
        """Access the layers as numpy arrays of AtomGroups.

        The object can be sliced as usual with numpy arrays.

        """
        return self._layers

#
