#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

""" Module: willard-chandler
    ========================
"""

import numpy as np
from scipy.spatial import Delaunay
from skimage import measure
from pytim import utilities
import pytim


class WillardChandler(pytim.PYTIM):
    """Identifies the dividing surface using the Willard-Chandler method
       (A. P. Willard and D. Chandler, J. Phys. Chem. B 2010, 114, 1954–1958)

    :param Universe universe: the MDAnalysis universe
    :param float mesh:        the grid spacing for the density calculation
    :param float alpha:       the width of the Gaussian kernel
    :param AtomGroup itim_group: compute the density using this group
    :param dict radii_dict:   dictionary with the atomic radii of the\
                              elements in the itim_group. If None is\
                              supplied, the default one (from MDAnalysis)\
                              will be used.
    :param str surface_basename: if not None, save the triangulation on
                              an UNSTRUCTURED_GRID on\
                              vtk files named <surface_basename>.<n>.vtk,\
                              where <n> is the frame number.
    :param str particle_basename: if not None, save the particles (along\
                              with radii and color scheme for the atom\
                              types using POLYDATA on vtk files.\
                              Uses the same file name convetion as\
                              surface_basename.
    :param str density_basename: if not None, save the full density\
                              (along with radii and color scheme for the\
                              atom types) using STRUCTURED_POINTS on vtk\
                              files. Uses the same file name convetion as\
                              surface_basename.

    """

    @property
    def layers(self):
        """The method does not identify layers."""
        self.layers = None
        return None

    def writepdb(self,filename='layers.pdb',centered='no',multiframe=True):
        """The method does not identify layers."""
        pass

    def _sanity_checks(self):
        """ Basic checks to be performed after the initialization.

            We test them also here in the docstring:

            >>> import pytim 
            >>> import MDAnalysis as mda
            >>> u = mda.Universe(pytim.datafiles.WATER_GRO)
            >>>
            >>> pytim.WillardChandler(u,alpha=-1.0)
            Traceback (most recent call last):
            ...
            ValueError: parameter alpha must be positive

            >>> pytim.WillardChandler(u,alpha=-1000000)
            Traceback (most recent call last):
            ...
            ValueError: parameter alpha must be smaller than the smaller box side

            >>> pytim.WillardChandler(u,mesh=-1)
            Traceback (most recent call last):
            ...
            ValueError: parameter mesh must be positive
    
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
            if not(elements == 1 or elements == 1 + extraelements):
                raise StandardError(self.MISMATCH_CLUSTER_SEARCH)
        else:
            if self.extra_cluster_groups is not None:
                raise ValueError(self.UNDEFINED_CLUSTER_SEARCH)

    def __init__(self, universe, alpha=2.0, mesh=30, itim_group=None,
                 radii_dict=None, surface_basename=None,
                 particles_basename=None, density_basename=None):

        self._basic_checks(universe)
        self.cluster_cut = None
        # TODO make a uniform grid for non-cubic boxes, use part of _assign_mesh() from itim.py
        self.mesh = mesh
        self.extra_cluster_groups = None
        self.universe = universe
        self.alpha = alpha
        self.itim_group = itim_group
        self.density_basename = density_basename
        self.particles_basename = particles_basename
        self.surface_basename = surface_basename

        self.assign_radii(radii_dict)
        self._sanity_checks()

        self._define_groups()

        pytim.PatchTrajectory(universe.trajectory, self)
        self._assign_layers()

    def dump_density(self, densmap):
        """save the density on a vtk file named consecutively using the frame
        number."""
        filename = utilities.vtk_consecutive_filename(self.universe,
                                                      self.density_basename)
        spacing = self.universe.dimensions[:3] / self.mesh
        utilities.write_vtk_scalar_grid(filename, self.mesh, spacing,
                                        densmap)

    def dump_points(self, pos):
        """save the particles n a vtk file named consecutively using the frame
        number."""
        radii = self.itim_group.radii
        types = self.itim_group.types
        color = [utilities.colormap[element] for element in types]
        color = (np.array(color) / 256.).tolist()

        filename = utilities.vtk_consecutive_filename(self.universe,
                                                      self.particles_basename)
        utilities.write_vtk_points(filename, pos, color=color, radius=radii)

    def dump_triangulation(self, vertices, triangles):
        """save a triangulation on a vtk file named consecutively using the
        frame number."""
        filename = utilities.vtk_consecutive_filename(self.universe,
                                                      self.surface_basename)
        utilities.write_vtk_triangulation(filename, vertices, triangles)

    def _assign_layers(self):
        """There are no layers in the Willard-Chandler method.

        This function identifies the dividing surface and stores the
        triangulated isosurface, the density and the particles.

        """
        # this can be used later to shift back to the original shift
        self.original_positions = np.copy(self.universe.atoms.positions[:])
        self.universe.atoms.pack_into_box()

        pos = self.itim_group.positions
        box = self.universe.dimensions[:3]
        delta = 2. * self.alpha + 1e-6
        extrapoints, _ = utilities.generate_periodic_border_3d(
            pos, box, delta
        )
        grid = utilities.generate_grid_in_box(box, self.mesh)
        kernel, std = utilities.density_map(pos, grid, self.alpha)
        field = kernel(grid)

        # Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan
        # Tavares. Efficient implementation of Marching Cubes’ cases with
        # topological guarantees. Journal of Graphics Tools 8(2) pp. 1-15
        # (december 2003). DOI: 10.1080/10867651.2003.10487582
        volume = field.reshape((self.mesh, self.mesh, self.mesh))

        spacing = box / self.mesh
        verts, faces, normals, values = measure.marching_cubes(
            volume, None,
            spacing=tuple(spacing)
        )
        self.triangulated_surface = [verts, faces]
        self.surface_area = measure.mesh_surface_area(verts, faces)
        verts += spacing / 2.

        if self.density_basename is not None:
            self.dump_density(field)
        if self.particles_basename is not None:
            self.dump_points(pos)
        if self.surface_basename is not None:
            self.dump_triangulation(verts, faces)

#
