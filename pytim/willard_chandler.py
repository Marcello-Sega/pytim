#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

""" Module: willard-chandler
    ========================
"""

import numpy as np
from skimage import measure
from pytim import utilities, vtk, cube, wavefront_obj
import pytim


class WillardChandler(pytim.PYTIM):
    """Identifies the dividing surface using the Willard-Chandler method
       (A. P. Willard and D. Chandler, J. Phys. Chem. B 2010, 114, 1954–1958)

    :param Universe universe: the MDAnalysis universe
    :param float mesh:        the grid spacing for the density calculation
    :param float alpha:       the width of the Gaussian kernel
    :param AtomGroup group:   compute the density using this group
    :param dict radii_dict:   dictionary with the atomic radii of the\
                              elements in the group. If None is\
                              supplied, the default one (from MDAnalysis)\
                              will be used.

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
    >>> interface = pytim.WillardChandler(u, group=g, alpha=3.0)
    >>> R, _, _, _ = pytim.utilities.fit_sphere(\
                       interface.triangulated_surface[0])
    >>> print "Radius={:.3f}".format(R)
    Radius=19.325

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
                 group=None, radii_dict=None,
                 cluster_cut=None, cluster_threshold_density=None,
                 extra_cluster_groups=None, centered=False, **kargs):

        self.do_center = centered
        sanity = pytim.SanityCheck(self)
        sanity.assign_universe(universe)
        sanity.assign_alpha(alpha)

        if mesh <= 0:
            raise ValueError(self.MESH_NEGATIVE)
        self.mesh = mesh
        self.spacing = None
        self.ngrid = None
        self.PDB = {}

        sanity.assign_radii(radii_dict)

        sanity.assign_groups(group, cluster_cut, extra_cluster_groups)

        self._assign_symmetry(symmetry)

        if(self.symmetry == 'planar'):
            sanity.assign_normal(normal)

        pytim.PatchTrajectory(universe.trajectory, self)
        self._assign_layers()
        self._atoms = self._layers[:]  # this is an empty AtomGroup
        self.writevtk = WillardChandler.Writevtk(self)

    class Writevtk(object):
        def __init__(self, interface):
            self.interface = interface

        def density(self, filename='pytim_dens.vtk', sequence=False):
            """ Write to vtk files the volumetric density:
                :param str filename: the file name
                :param bool sequence: if true writes a sequence of files adding the frame to the filename
                >>> interface.writevtk.density('dens.vtk') # writes on dens.vtk
                >>> interface.writevtk.density('dens.vtk',sequence=True) # writes on dens.<frame>.vtk
            """
            inter = self.interface
            if sequence == True:
                filename = vtk.consecutive_filename(inter.universe, filename)
            vtk.write_scalar_grid(filename, inter.ngrid,
                                  inter.spacing, inter.density_field)

        def particles(self, filename='pytim_part.vtk', group=None, sequence=False):
            """ Write to vtk files the particles in a group:
                :param str filename: the file name
                :param bool sequence: if true writes a sequence of files adding the frame to the filename
                :param AtomGroup group: if None, writes the whole universe
                >>> interface.writevtk.particles('part.vtk') # writes on part.vtk
                >>> interface.writevtk.particles('part.vtk',sequence=True) # writes on part.<frame>.vtk
            """
            inter = self.interface
            if sequence == True:
                filename = vtk.consecutive_filename(inter.universe, filename)
            if group == None:
                group = inter.universe.atoms
            self._dump_group(group, filename)

        def surface(self, filename='pytim_surf.vtk', sequence=False):
            """ Write to vtk files the triangulated surface:
                :param str filename: the file name
                :param bool sequence: if true writes a sequence of files adding the frame to the filename
                >>> interface.writevtk.surface('surf.vtk') # writes on surf.vtk
                >>> interface.writevtk.surface('surf.vtk',sequence=True) # writes on surf.<frame>.vtk
            """
            inter = self.interface
            vertices = inter.triangulated_surface[0]
            faces = inter.triangulated_surface[1]
            normals = inter.triangulated_surface[2]
            if sequence == True:
                filename = vtk.consecutive_filename(inter.universe, filename)
            vtk.write_triangulation(
                filename, vertices[::, ::-1], faces, normals)

    def writecube(self, filename="pytim.cube", group=None, sequence=False):
        """ Write to cube files (sequences) the volumentric density and the atomic positions.
            :param str filename: the file name
            :param bool sequence: if true writes a sequence of files adding the frame to the filename
            >>> interface.writecube('dens.cube') # writes on dens.cube
            >>> interface.writecube('dens.cube',sequence=True) # writes on dens.<frame>.cube
        """
        if sequence == True:
            filename = cube.consecutive_filename(self.universe, filename)
        # TODO handle optional atomic_numbers
        cube.write_file(filename, group, self.ngrid,
                        self.spacing, self.density_field, atomic_numbers=None)

    def writeobj(self, filename="pytim.obj", sequence=False):
        """ Write to wavefront obj files (sequences) the triangulated surface
            :param str filename: the file name
            :param bool sequence: if true writes a sequence of files adding the frame to the filename
            >>> interface.writeobj('surf.obj') # writes on surf.obj
            >>> interface.writeobj('surf.obj',sequence=True) # writes on surf.<frame>.obj
        """

        if sequence == True:
            filename = wavefront_obj.consecutive_filename(
                self.universe, filename)

        vert = self.triangulated_surface[0]
        surf = self.triangulated_surface[1]
        wavefront_obj.write_file(filename, vert, surf)

    def _dump_group(self, group, filename):
        """save the particles n a vtk file named consecutively using the frame
        number."""
        radii = group.radii
        types = group.types
        color = [utilities.colormap[element] for element in types]
        color = (np.array(color) / 256.).tolist()
        vtk.write_atomgroup(filename, group, color=color, radius=radii)

    def _assign_layers(self):
        """There are no layers in the Willard-Chandler method.

        This function identifies the dividing surface and stores the
        triangulated isosurface, the density and the particles.

        """
        # we assign an empty group for consistency
        self._layers = self.universe.atoms[:0]

        self.normal = None

        # this can be used later to shift back to the original shift
        self.original_positions = np.copy(self.universe.atoms.positions[:])
        self.universe.atoms.pack_into_box()

        self._define_cluster_group()

        self.centered_positions = None
        if self.do_center == True:
            self.center()

        pos = self.cluster_group.positions
        box = self.universe.dimensions[:3]
        delta = 2. * self.alpha + 1e-6
        #extrapoints, _ = utilities.generate_periodic_border_3d(pos, box, delta)
        ngrid, spacing = utilities.compute_compatible_mesh_params(
            self.mesh, box
        )
        self.spacing = spacing
        self.ngrid = ngrid
        grid = utilities.generate_grid_in_box(box, ngrid, order='zyx')
        kernel, _ = utilities.density_map(pos, grid, self.alpha, box)

        self.density_field = kernel.evaluate_pbc(grid)

        # Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan
        # Tavares. Efficient implementation of Marching Cubes’ cases with
        # topological guarantees. Journal of Graphics Tools 8(2) pp. 1-15
        # (december 2003). DOI: 10.1080/10867651.2003.10487582
        volume = self.density_field.reshape(tuple(ngrid[::-1]))
        verts, faces, normals, values = measure.marching_cubes(
            volume, None,
            spacing=tuple(spacing)
        )
        # note that len(normals) == len(verts): they are normals
        # at the vertices, and not normals of the faces
        self.triangulated_surface = [verts, faces, normals]
        self.surface_area = measure.mesh_surface_area(verts, faces)
        verts += spacing[::-1] / 2.


#
