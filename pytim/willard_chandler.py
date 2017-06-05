#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

""" Module: willard-chandler
    ========================
"""

import numpy as np
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
    :param str particles_basename: if not None, save the particles (along\
                              with radii and color scheme for the atom\
                              types using POLYDATA on vtk files.\
                              Uses the same file name convetion as\
                              surface_basename.
    :param str density_basename: if not None, save the full density\
                              (along with radii and color scheme for the\
                              atom types) using STRUCTURED_POINTS on vtk\
                              files. Uses the same file name convetion as\
                              surface_basename.

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
    >>> interface = pytim.WillardChandler(u, itim_group=g, alpha=3.0,\
    ...                                   density_basename="dens",\
    ...                                   particles_basename="atoms",\
    ...                                   surface_basename="surf")
    >>> R, _, _, _ = pytim.utilities.fit_sphere(\
    ...                interface.triangulated_surface[0])
    >>> print "Radius={:.3f}".format(R)
    Radius=19.325

    """

    _surface = None

    @property
    def layers(self):
        """The method does not identify layers."""
        self.layers = None
        return None

    def writepdb(self, filename='layers.pdb', centered='no', multiframe=True):
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

    def __init__(self, universe, alpha=2.0, mesh=2.0,
                 itim_group=None,
                 radii_dict=None, surface_basename=None,
                 particles_basename=None, density_basename=None):

        sanity = pytim.SanityCheck(self)
        sanity.assign_universe(universe)
        sanity.assign_alpha(alpha)

        self.mesh = mesh
        self.spacing = None
        self.ngrid = None

        self.density_basename = density_basename
        self.particles_basename = particles_basename
        self.group_basename = particles_basename+'_group'
        self.surface_basename = surface_basename

        sanity.assign_radii(radii_dict)
        # TODO implement cluster group
        sanity.assign_groups(itim_group, None, None)

        pytim.PatchTrajectory(universe.trajectory, self)
        self._assign_layers()

    def dump_density(self, densmap):
        """save the density on a vtk file named consecutively using the frame
        number."""
        filename = utilities.vtk.consecutive_filename(self.universe,
                                                      self.density_basename)
        utilities.vtk.write_scalar_grid(filename, self.ngrid, self.spacing,
                                        densmap)

    def dump_group(self, group,groupfilename):
        """save the particles n a vtk file named consecutively using the frame
        number."""
        radii = group.radii
        types = group.types
        color = [utilities.colormap[element] for element in types]
        color = (np.array(color) / 256.).tolist()

        filename = utilities.vtk.consecutive_filename(self.universe,
                                                      groupfilename)
        utilities.vtk.write_points(filename, group.positions, color=color,
                                   radius=radii)

    def dump_triangulation(self, vertices, triangles, normals=None):
        """save a triangulation on a vtk file named consecutively using the
        frame number."""
        filename = utilities.vtk.consecutive_filename(self.universe,
                                                      self.surface_basename)
        utilities.vtk.write_triangulation(
            filename, vertices, triangles, normals)

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
        #extrapoints, _ = utilities.generate_periodic_border_3d(pos, box, delta)
        ngrid, spacing = utilities.compute_compatible_mesh_params(
            self.mesh, box
        )
        self.spacing = spacing
        self.ngrid = ngrid
        grid = utilities.generate_grid_in_box(box, ngrid)
        kernel, _ = utilities.density_map(pos , grid, self.alpha,box)

        field = kernel.evaluate_pbc(grid)

        # Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan
        # Tavares. Efficient implementation of Marching Cubes’ cases with
        # topological guarantees. Journal of Graphics Tools 8(2) pp. 1-15
        # (december 2003). DOI: 10.1080/10867651.2003.10487582
        volume = field.reshape(tuple(ngrid[::-1]))
        verts, faces, normals, values = measure.marching_cubes(
            volume, None,
            spacing=tuple(spacing)
        )
        # note that len(normals) == len(verts): they are normals
        # at the vertices, and not normals of the faces
        self.triangulated_surface = [verts, faces, normals]
        self.surface_area = measure.mesh_surface_area(verts, faces)
        verts += spacing[::-1]/2.

        if self.density_basename is not None:
            self.dump_density(field)
        if self.particles_basename is not None:
            self.dump_group(self.universe.atoms,self.particles_basename+'all')
            self.dump_group(self.itim_group,self.particles_basename+'itim')

        if self.surface_basename is not None:
            # not quite sure where the order (xyz) got reverted,
            # most likely in utilities.generate_grid_in_box() called above
            self.dump_triangulation(verts[::,::-1], faces, normals)

#
