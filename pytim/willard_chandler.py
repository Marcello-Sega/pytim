#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: willard-chandler
    ========================
"""

from __future__ import print_function
from skimage import measure
import numpy as np

from . import messages
from . import utilities, cube, wavefront_obj
from .sanity_check import SanityCheck
from .vtk import Writevtk

from .interface import Interface
from .patches import patchTrajectory, patchOpenMM, patchMDTRAJ

np.set_printoptions(legacy=False)  # fixes problem with skimage

try:
    marching_cubes = measure.marching_cubes
except AttributeError:
    marching_cubes = measure.marching_cubes_lewiner


class WillardChandler(Interface):
    """ Identifies the dividing surface using the Willard-Chandler method
        NOTE that this method does *not* identify surface atoms

        *(Willard, A. P.; Chandler, D. J. Phys. Chem. B 2010, 114, 1954–1958)*

        :param Object universe:   The MDAnalysis_ Universe, MDTraj_ trajectory
                                  or OpenMM_ Simulation objects.
        :param Object group:      An AtomGroup, or an array-like object with
                                  the indices of the atoms in the group.
                                  Will identify the interfacial molecules from
                                  this group
        :param float alpha:       The width of the Gaussian kernel
        :param float mesh:        The grid spacing for the density calculation
        :param float density_cutoff: The density value used to define the
                                  isosurface. `None` (default) uses the average
                                  of the minimum and maximum density.
        :param AtomGroup group:   Compute the density using this group
        :param dict radii_dict:   Dictionary with the atomic radii of
                                  the elements in the group.
                                  If None is supplied, the default one
                                  (from GROMOS 43a1) will be used.
        :param float cluster_cut: Cutoff used for neighbors or density-based
                                  cluster search (default: None disables the
                                  cluster analysis)
        :param float cluster_threshold_density: Number density threshold for
                                  the density-based cluster search. 'auto'
                                  determines the threshold automatically.
                                  Default: None uses simple neighbors cluster
                                  search, if cluster_cut is not None
        :param Object extra_cluster_groups: Additional groups, to allow for
                                  mixed interfaces

        :param bool include_zero_radius: if false (default) exclude atoms with zero radius
                                  from the surface analysis (they are always included
                                  in the cluster search, if present in the relevant
                                  group) to avoid some artefacts.
        :param bool centered:     Center the  :py:obj:`group`
        :param bool warnings:     Print warnings

        Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import *
        >>>
        >>> u = mda.Universe(MICELLE_PDB)
        >>> g = u.select_atoms('resname DPC')
        >>>
        >>> radii = pytim_data.vdwradii(G43A1_TOP)
        >>>
        >>> inter= pytim.WillardChandler(u, group=g, alpha=3.0, fast=True)
        >>> R, _, _, _ = pytim.utilities.fit_sphere(inter.triangulated_surface[0])
        >>> print ("Radius={:.3f}".format(R))
        Radius=19.970


        .. _MDAnalysis: http://www.mdanalysis.org/
        .. _MDTraj: http://www.mdtraj.org/
        .. _OpenMM: http://www.openmm.org/
    """

    _surface = None

    @property
    def layers(self):
        """ The method does not identify layers.

        Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import *
        >>>
        >>> u = mda.Universe(MICELLE_PDB)
        >>> g = u.select_atoms('resname DPC')
        >>> inter= pytim.WillardChandler(u, group=g, alpha=3.0, fast=True)
        >>> inter.layers
        <AtomGroup with 0 atoms>
        """
        return self._layers

    def _sanity_checks(self):
        """ Basic checks to be performed after the initialization.

            >>> import pytest
            >>> with pytest.raises(Exception):
            ...     pytim.WillardChandler(u,mesh=-1)

        """

    def __init__(self,
                 universe,
                 group=None,
                 alpha=2.0,
                 radii_dict=None,
                 mesh=2.0,
                 symmetry='spherical',
                 cluster_cut=None,
                 include_zero_radius=False,
                 cluster_threshold_density=None,
                 extra_cluster_groups=None,
                 centered=False,
                 warnings=False,
                 autoassign=True,
                 density_cutoff=None,
                 **kargs):

        self.autoassign, self.do_center = autoassign, centered
        self.include_zero_radius = include_zero_radius
        self.density_cutoff = density_cutoff
        sanity = SanityCheck(self, warnings=warnings)
        sanity.assign_universe(universe, group)
        sanity.assign_alpha(alpha)

        if mesh <= 0:
            raise ValueError(messages.MESH_NEGATIVE)
        self.mesh, self.spacing, self.ngrid, self.PDB = mesh, None, None, {}

        sanity.assign_radii(radii_dict=radii_dict)

        sanity.assign_cluster_params(cluster_cut,
                                     cluster_threshold_density, extra_cluster_groups)

        self._assign_symmetry(symmetry)

        patchTrajectory(self.universe.trajectory, self)
        self._assign_layers()
        self._atoms = self._layers[:]  # this is an empty AtomGroup
        self.writevtk = Writevtk(self)

    def writecube(self, filename="pytim.cube", group=None, sequence=False):
        """ Write to cube files (sequences) the volumentric density and the
            atomic positions.

            :param str filename:  the file name
            :param bool sequence: if true writes a sequence of files adding
                                  the frame to the filename

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import MICELLE_PDB
            >>> u = mda.Universe(MICELLE_PDB)
            >>> g = u.select_atoms('resname DPC')
            >>> inter= pytim.WillardChandler(u, group=g, alpha=3.0, mesh=2.0)
            >>> inter.writecube('dens.cube') # writes on dens.cube
            >>> inter.writecube('dens.cube',group=g) # writes also  particles
            >>> inter.writecube('dens.cube',sequence=True) # dens.<frame>.cube
        """
        if sequence is True:
            filename = cube.consecutive_filename(self.universe, filename)
        # TODO handle optional atomic_numbers
        cube.write_file(
            filename,
            group,
            self.ngrid,
            self.spacing,
            self.density_field,
            atomic_numbers=None)

    def writeobj(self, filename="pytim.obj", sequence=False):
        """ Write to wavefront obj files (sequences) the triangulated surface

            :param str filename:  the file name
            :param bool sequence: if true writes a sequence of files adding
                                  the frame to the filename

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import MICELLE_PDB
            >>> u = mda.Universe(MICELLE_PDB)
            >>> g = u.select_atoms('resname DPC')
            >>> inter= pytim.WillardChandler(u, group=g, alpha=3.0, mesh=2.0)
            >>> inter.writeobj('surf.obj') # writes on surf.obj
            >>> inter.writeobj('surf.obj',sequence=True) # surf.<frame>.obj
        """

        if sequence is True:
            filename = wavefront_obj.consecutive_filename(
                self.universe, filename)

        vert, surf = list(self.triangulated_surface[0:2])
        wavefront_obj.write_file(filename, vert, surf)

    def _assign_layers(self):
        """ There are no layers in the Willard-Chandler method.

            This function identifies the dividing surface and stores the
            triangulated isosurface, the density and the particles.

        """
        self.reset_labels()
        # we assign an empty group for consistency
        self._layers, self.normal = self.universe.atoms[:0], None

        self.prepare_box()

        self._define_cluster_group()

        self.centered_positions = None
        if self.do_center is True:
            self.center()

        if self.include_zero_radius:
            pos = self.cluster_group.positions
        else:
            pos = self.cluster_group.positions[self.cluster_group.radii > 0.0]
        box = self.universe.dimensions[:3]

        ngrid, spacing = utilities.compute_compatible_mesh_params(
            self.mesh, box)
        self.spacing, self.ngrid = spacing, ngrid
        grid = utilities.generate_grid_in_box(box, ngrid, order='xyz')
        kernel, _ = utilities.density_map(pos, grid, self.alpha, box)

        kernel.pos = pos.copy()
        self.density_field = kernel.evaluate_pbc_fast(grid)

        # Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan
        # Tavares. Efficient implementation of Marching Cubes’ cases with
        # topological guarantees. Journal of Graphics Tools 8(2) pp. 1-15
        # (december 2003). DOI: 10.1080/10867651.2003.10487582
        volume = self.density_field.reshape(
            tuple(np.array(ngrid[::-1]).astype(int)))
        verts, faces, normals, values = marching_cubes(
            volume, self.density_cutoff, spacing=tuple(spacing))
        # note that len(normals) == len(verts): they are normals
        # at the vertices, and not normals of the faces
        # verts and normals have x and z flipped because skimage uses zyx ordering
        self.triangulated_surface = [np.fliplr(verts), faces, np.fliplr(normals)]
        self.surface_area = measure.mesh_surface_area(verts, faces)
        verts += spacing[::-1] / 2.
