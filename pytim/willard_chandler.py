#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

""" Module: willard-chandler
    ========================
"""

import numpy as np
from skimage import measure
from pytim import utilities,vtk,cube
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
    :param str basename:      if not None, prepend it to the output files
                              with volumetric, surface or particle data\
                              as  <basename>_<type>.<n>.extension,\
                              where <n> is the frame number.
    :param str output_format: 'vtk', 'cube', or None
    :param bool output_part:  if True and output_format is 'vtk', save the \
                              particles (along with radii and color scheme \
                              for the atom\ types using POLYDATA.\
                              Disregarded otherwise.
    :param bool output_dens:  if True and output_format is 'vtk' save the
                              volumetric density data using STRUCTURED_POINTS\
                              Disregarded otherwise.
    :param bool output_surf:  if True and output_format is 'vtk' save the
                              surface triangulation data using UNSTRUCTURED_GRID\
                              Disregarded otherwise.

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
                                          output_format='vtk',\
                                          output_dens=True,\
                                          output_part=True,\
                                          output_surf=True)
    >>> R, _, _, _ = pytim.utilities.fit_sphere(\
                       interface.triangulated_surface[0])
    >>> print "Radius={:.3f}".format(R)
    Radius=19.376

    """

    _surface = None

    @property
    def layers(self):
        """The method does not identify layers."""
        self.layers = None
        return None

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

    def __init__(self, universe, alpha=2.0, mesh=2.0, symmetry='spherical',
                 itim_group=None, radii_dict=None,
                 cluster_cut=None, cluster_threshold_density=None,
                 extra_cluster_groups=None,
                 output_format=None, output_surf=True, output_part=True,
                 output_dens=True, basename=None,**kargs):

        sanity = pytim.SanityCheck(self)
        sanity.assign_universe(universe)
        sanity.assign_alpha(alpha)

        self.mesh = mesh
        self.spacing = None
        self.ngrid = None
        self.output_format = output_format
        self.output={'dens':output_dens,'surf':output_surf,'part':output_part}
        if basename is None:
            self.basename=''
        self.density_basename = self.basename+'dens'
        self.particles_basename = self.basename+'part'
        self.surface_basename = self.basename+'surf'
        self.PDB = {}

        sanity.assign_radii(radii_dict)

        sanity.assign_groups(itim_group, cluster_cut, extra_cluster_groups)

        self._assign_symmetry(symmetry)

        if(self.symmetry == 'planar'):
            sanity.assign_normal(normal)

        pytim.PatchTrajectory(universe.trajectory, self)
        self._assign_layers()
        self._atoms = self._layers[:] # this is an empty AtomGroup

    def dump_density(self, densmap):
        """save the density on a vtk file named consecutively using the frame
        number."""
        filename = vtk.consecutive_filename(self.universe,
                                                      self.density_basename)
        vtk.write_scalar_grid(filename, self.ngrid, self.spacing,
                                        densmap)

    def dump_group(self, group,groupfilename):
        """save the particles n a vtk file named consecutively using the frame
        number."""
        radii = group.radii
        types = group.types
        color = [utilities.colormap[element] for element in types]
        color = (np.array(color) / 256.).tolist()

        filename = vtk.consecutive_filename(self.universe,
                                                      groupfilename)
        vtk.write_atomgroup(filename, group, color=color,
                                   radius=radii)

    def dump_triangulation(self, vertices, triangles, normals=None):
        """save a triangulation on a vtk file named consecutively using the
        frame number."""
        filename = vtk.consecutive_filename(self.universe,
                                                      self.surface_basename)
        vtk.write_triangulation(
            filename, vertices, triangles, normals)

    def _assign_layers(self):
        """There are no layers in the Willard-Chandler method.

        This function identifies the dividing surface and stores the
        triangulated isosurface, the density and the particles.

        """
        # we assign an empty group for consistency
        self._layers = self.universe.atoms[:0]

        # this can be used later to shift back to the original shift
        self.original_positions = np.copy(self.universe.atoms.positions[:])
        self.universe.atoms.pack_into_box()

        self._define_cluster_group()

        if self.symmetry == 'planar':
            utilities.centerbox(self.universe, center_direction=self.normal)
            self.center(self.cluster_group, self.normal)
            utilities.centerbox(self.universe, center_direction=self.normal)
        if self.symmetry == 'spherical':
            self.center(self.cluster_group, 'x', halfbox_shift=False)
            self.center(self.cluster_group, 'y', halfbox_shift=False)
            self.center(self.cluster_group, 'z', halfbox_shift=False)
            self.universe.atoms.pack_into_box(self.universe.dimensions[:3])

        pos = self.cluster_group.positions
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

        if self.output_format == 'vtk':
            if self.output['dens']:
                self.dump_density(field)
            if self.output['part']:
                # this can be extended to different groups if needed
                self.dump_group(self.universe.atoms,self.particles_basename)
            if self.output['surf']:
                # not quite sure where the order (xyz) got reverted,
                # most likely in utilities.generate_grid_in_box() called above
                self.dump_triangulation(verts[::,::-1], faces, normals)
        if self.output_format == 'cube':
                filename = cube.consecutive_filename(self.universe,
                                                     self.basename+'data')
                # field is stored in z,y,x, we need to flip it
                _field = field.reshape((ngrid[2],ngrid[1]*ngrid[0])).T.flatten()
                # TODO handle optional atomic_numbers
                cube.write_file(filename, self.universe.atoms, self.ngrid,
                                spacing,_field, atomic_numbers=None)

#
