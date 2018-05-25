#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: gitim
    =============
"""
from __future__ import print_function
import numpy as np
from scipy.spatial import distance

from . import utilities
from .sanity_check import SanityCheck
from .surface import SurfaceFlatInterface
from .surface import SurfaceGenericInterface
try:
    from pytetgen import Delaunay
except ImportError:
    from scipy.spatial import Delaunay

from .interface import Interface
from .patches import PatchTrajectory, PatchOpenMM, PatchMDTRAJ
from circumradius import circumradius


class GITIM(Interface):
    """ Identifies interfacial molecules at curved interfaces.

        *(Sega, M.; Kantorovich, S.; Jedlovszky, P.; Jorge, M., \
J. Chem. Phys. 138, 044110, 2013)*

        :param Object universe:   The MDAnalysis_ Universe, MDTraj_ trajectory
                                  or OpenMM_ Simulation objects.
        :param Object group:        An AtomGroup, or an array-like object with
                                    the indices of the atoms in the group. Will
                                    identify the interfacial molecules from
                                    this group
        :param float alpha:         The probe sphere radius
        :param str normal:          'x','y,'z' or 'guess'
                                    (for planar interfaces only)
        :param bool molecular:      Switches between search of interfacial
                                    molecules / atoms (default: True)
        :param int max_layers:      The number of layers to be identified
        :param dict radii_dict:     Dictionary with the atomic radii of the
                                    elements in the group. If None is supplied,
                                    the default one (from GROMOS 43a1) will be
                                    used.
        :param float cluster_cut:   Cutoff used for neighbors or density-based
                                    cluster search (default: None disables the
                                    cluster analysis)
        :param float cluster_threshold_density: Number density threshold for
                                    the density-based cluster search. 'auto'
                                    determines the threshold automatically.
                                    Default: None uses simple neighbors cluster
                                    search, if cluster_cut is not None
        :param Object extra_cluster_groups: Additional groups, to allow for
                                    mixed interfaces
        :param bool biggest_cluster_only: Tag as surface atoms/molecules only
                                    those in the largest cluster. Need to
                                    specify also a :py:obj:`cluster_cut` value.
        :param str symmetry:        Gives the code a hint about the topology
                                    of the interface: 'generic' (default)
                                    or  'planar'
        :param bool centered:       Center the  :py:obj:`group`
        :param bool info:           Print additional info
        :param bool warnings:       Print warnings
        :param bool autoassign:     If true (default) detect the interface
                                    every time a new frame is selected.

        Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from   pytim.datafiles import *
        >>>
        >>> u = mda.Universe(MICELLE_PDB)
        >>> g = u.select_atoms('resname DPC')
        >>>
        >>> interface =pytim.GITIM(u,group=g,molecular=False, alpha=2.0)
        >>> layer = interface.layers[0]
        >>> interface.writepdb('gitim.pdb',centered=False)
        >>> print (repr(layer))
        <AtomGroup with 909 atoms>


        Successive layers can be identified with :mod:`~pytim.gitim.GITIM`
        as well. In this example we identify two solvation shells of glucose:


        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from   pytim.datafiles import *
        >>>
        >>> u = mda.Universe(GLUCOSE_PDB)
        >>> g = u.select_atoms('name OW')
        >>> # it is faster to consider only oxygens.
        >>> # Hydrogen atoms are anyway within Oxygen's radius,
        >>> # in SPC* models.
        >>> interface =pytim.GITIM(u, group=g, alpha=2.0, max_layers=2)
        >>>
        >>> interface.writepdb('glucose_shells.pdb')
        >>> print (repr(interface.layers[0]))
        <AtomGroup with 54 atoms>
        >>> print (repr(interface.layers[1]))
        <AtomGroup with 117 atoms>

        .. _MDAnalysis: http://www.mdanalysis.org/
        .. _MDTraj: http://www.mdtraj.org/
        .. _OpenMM: http://www.openmm.org/
    """

    def __init__(self,
                 universe,
                 group=None,
                 alpha=2.0,
                 normal='guess',
                 molecular=True,
                 max_layers=1,
                 radii_dict=None,
                 cluster_cut=None,
                 cluster_threshold_density=None,
                 extra_cluster_groups=None,
                 biggest_cluster_only=False,
                 symmetry='generic',
                 centered=False,
                 info=False,
                 warnings=False,
                 autoassign=True,
                 _noextrapoints=False,
                 **kargs):

        # this is just for debugging/testing
        self._noextrapoints = _noextrapoints
        self.autoassign = autoassign

        self.do_center = centered

        self.biggest_cluster_only = biggest_cluster_only
        sanity = SanityCheck(self)
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
        try:
            self._buffer_factor = kargs['buffer_factor']
        except:
            self._buffer_factor = 3.5

        if (self.symmetry == 'planar'):
            sanity.assign_normal(normal)
            self._surfaces = np.empty(
                max_layers, dtype=type(SurfaceFlatInterface))
            for nlayer in range(max_layers):
                self._surfaces[nlayer] = SurfaceFlatInterface(
                    self, options={'layer': nlayer})
        else:  # generic
            self._surfaces = np.empty(
                max_layers, dtype=type(SurfaceGenericInterface))
            for nlayer in range(max_layers):
                self._surfaces[nlayer] = SurfaceGenericInterface(
                    self, options={'layer': nlayer})

        PatchTrajectory(self.universe.trajectory, self)

        self._assign_layers()

    def _sanity_checks(self):
        """ Basic checks to be performed after the initialization.

        """

    def alpha_shape(self, alpha, group, layer):
        box = self.universe.dimensions[:3]
        delta = np.array([self._buffer_factor * self.alpha] * 3)
        delta = np.min([delta, box / 2.], axis=0)

        points = group.positions[:]
        nrealpoints = len(points)
        state = np.random.get_state()
        np.random.seed(0)  # pseudo-random for reproducibility
        gitter = (np.random.random(3 * 8).reshape(8, 3)) * 1e-9
        if self._noextrapoints is False:
            extrapoints, extraids = utilities.generate_periodic_border(
                points, box, delta, method='3d')
        else:
            extrapoints = np.copy(points)
            extraids = np.arange(len(points), dtype=np.int)
        # add points at the vertices of the expanded (by 2 alpha) box
        cube_vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [
            0.0, 1.0, 0.0
        ], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]])
        if self._noextrapoints is False:
            n_cube = 8
            for dim, vertex in enumerate(cube_vertices):
                # added to prevent coplanar points
                cond = np.where(vertex < 0.5)[0]
                vertex = vertex * box + 2 * delta + gitter[dim]
                vertex[cond] -= 4 * delta[cond]
                vertex = np.reshape(vertex, (1, 3))
                extrapoints = np.append(extrapoints, vertex, axis=0)
                extraids = np.append(extraids, -1)
        else:
            n_cube = 0
        if layer == 0:
            self.triangulation = []
        self.triangulation.append(Delaunay(extrapoints))
        try:
            triangulation = self.triangulation[layer]
        except IndexError:
            raise IndexError("alpha_shape called using a wrong layer")
        triangulation.radii = np.append(group.radii[extraids[extraids >= 0]],
                                        np.zeros(8))

        simplices = triangulation.simplices

        try:
            _points = self.triangulation[layer].points
            radii = self.triangulation[layer].radii
        except IndexError:
            raise IndexError("alpha_shape called using a wrong layer")

        np.random.set_state(state)

        cr = circumradius(_points, radii, simplices)
        # we filter first according to the touching sphere radius
        a_shape = simplices[cr >= self.alpha]
        # then we remove all simplices involving the 8 outer points, if any
        cond = np.where(np.all(a_shape < len(_points) - n_cube, axis=1))[0]
        a_shape = a_shape[np.unique(cond)]
        # finally, we select only the ids of atoms in the basic cell.
        return np.unique(a_shape[np.where(a_shape < nrealpoints)])

    def _assign_layers_setup(self):
        self.reset_labels()
        # this can be used later to shift back to the original shift
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

        # TODO the successive layers analysis should be done by removing points
        # from the triangulation and updating the circumradius of the neighbors
        # of the removed points  only.

        dbs = utilities.do_cluster_analysis_dbscan
        return alpha_group, dbs

    def _assign_layers_postprocess(self, dbs, group, alpha_group, layer):
        if self.biggest_cluster_only is True:
            # apply the same clustering algorith as set at init
            l, c, _ = dbs(
                group,
                self.cluster_cut[0],
                threshold_density=self.cluster_threshold_density,
                molecular=self.molecular)
            group = group[np.where(np.array(l) == np.argmax(c))[0]]

        alpha_group = alpha_group[:] - group[:]
        if len(group) > 0:
            if self.molecular:
                self._layers[layer] = group.residues.atoms
            else:
                self._layers[layer] = group
        else:
            self._layers[layer] = group.universe.atoms[:0]

        self.label_group(
            self._layers[layer], beta=1. * (layer + 1), layer=(layer + 1))
        return alpha_group

    def _assign_layers(self):
        """Determine the GITIM layers."""

        alpha_group, dbs = self._assign_layers_setup()

        for layer in range(0, self.max_layers):

            alpha_ids = self.alpha_shape(self.alpha, alpha_group, layer)

            group = alpha_group[alpha_ids]

            alpha_group = self._assign_layers_postprocess(
                dbs, group, alpha_group, layer)

        # reset the interpolator
        self._interpolator = None

    @property
    def layers(self):
        """Access the layers as numpy arrays of AtomGroups.

        The object can be sliced as usual with numpy arrays.
        Differently from :mod:`~pytim.itim.ITIM`, there are no sides. Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import MICELLE_PDB
        >>>
        >>> u = mda.Universe(MICELLE_PDB)
        >>> micelle = u.select_atoms('resname DPC')
        >>> inter = pytim.GITIM(u, group=micelle, max_layers=3,molecular=False)
        >>> inter.layers  #all layers
        array([<AtomGroup with 909 atoms>, <AtomGroup with 301 atoms>,
               <AtomGroup with 164 atoms>], dtype=object)
        >>> inter.layers[0]  # first layer (0)
        <AtomGroup with 909 atoms>

        """
        return self._layers

    def _():
        """ additional tests

        >>> import numpy as np
        >>> from circumradius import circumradius
        >>> from scipy.spatial import Delaunay
        >>> p = [0.,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,0,1]
        >>> r = np.array(p).reshape(8,3)
        >>> tri = Delaunay(r)
        >>> radius = circumradius(r,np.ones(8)*0.5,tri.simplices)[0]
        >>> print(np.isclose(radius, (np.sqrt(3)-1.)/2))
        True

        >>> import pytim
        >>> import numpy as np
        >>> from pytim.datafiles import _TEST_BCC_GRO
        >>> import MDAnalysis as mda
        >>> u= mda.Universe(_TEST_BCC_GRO)
        >>> # we use a minimal system with one atom in the group
        >>> # this will represent a cubic lattice, and test also PBCs
        >>> u.atoms[0:1].positions = np.array([0., 0., 0.])
        >>> u.dimensions = np.array([1., 1., 1., 90., 90., 90.])
        >>> g = u.atoms[0:1]
        >>> # the maximum value is (np.sqrt(3)-1.)/2)  ~=  0.366025403
        >>> inter = pytim.GITIM(u,group=g,radii_dict={'A':0.5},alpha=0.3660254)
        >>> print(repr(inter.atoms))
        <AtomGroup with 1 atom>

        >>> # with alpha > (np.sqrt(3)-1.)/2) no atom is found as surface one
        >>> inter = pytim.GITIM(u,group=g,radii_dict={'A':0.5},alpha=0.3660255)
        >>> print(repr(inter.atoms))
        <AtomGroup with 0 atoms>


        """


#
