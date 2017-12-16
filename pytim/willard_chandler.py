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
       (Willard, A. P.; Chandler, D. J. Phys. Chem. B 2010, 114, 1954–1958)

    :param Object universe:   the MDAnalysis Universe, MDTraj trajectory\
                              or OpenMM Simulation objects.
    :param Object group:      An AtomGroup, or an array-like object with\
                              the indices of the atoms in the group.\
                              Will dentify the interfacial molecules from\
                              this group
    :param float alpha:       the width of the Gaussian kernel
    :param float mesh:        the grid spacing for the density calculation
    :param AtomGroup group:   compute the density using this group
    :param dict radii_dict:   dictionary with the atomic radii of\
                              the elements in the group.
                              If None is supplied, the default one\
                              (from GROMOS 43a1) will be used.
    :param float cluster_cut: cutoff used for neighbors or density-based\
                              cluster search (default: None disables the\
                              cluster analysis)
    :param float cluster_threshold_density: Number density threshold for\
                              the density-based cluster search. 'auto'\
                              determines the threshold automatically.\
                              Default: None uses simple neighbors cluster\
                              search, if cluster_cut is not None
    :param Object extra_cluster_groups: Additional groups, to allow for mixed interfaces
    :param bool centered:     center the  :py:obj:`group`
    :param bool warnings:     print warnings 
    :param bool fast:         use a faster version with truncated Gaussians (default: True)


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
    >>> interface = pytim.WillardChandler(u, group=g, alpha=3.0, fast=False)
    >>> R, _, _, _ = pytim.utilities.fit_sphere(\
                       interface.triangulated_surface[0])
    >>> print "Radius={:.3f}".format(R)
    Radius=19.325

    >>> # the fast kernel gives a slightly (<0.1 Angstrom) different result
    >>> interface = pytim.WillardChandler(u, group=g, alpha=3.0, fast=True)
    >>> R, _, _, _ = pytim.utilities.fit_sphere(\
                       interface.triangulated_surface[0])
    >>> print "Radius={:.3f}".format(R)
    Radius=19.383

    """

    _surface = None

    @property
    def layers(self):
        """The method does not identify layers."""
        self.layers = None
        return None

    def _sanity_checks(self):
        """ Basic checks to be performed after the initialization.

            >>> import pytest
            >>> with pytest.raises(Exception):
            ...     pytim.WillardChandler(u,mesh=-1)

        """

    def __init__(self, universe, group=None, 
                 alpha=2.0, radii_dict=None, mesh=2.0, symmetry='spherical',
                 cluster_cut=None, cluster_threshold_density=None,
                 extra_cluster_groups=None, centered=False, warnings=False, fast=True, **kargs):

        self.do_center = centered
        sanity = pytim.SanityCheck(self)
        sanity.assign_universe(
            universe, radii_dict=radii_dict, warnings=warnings)
        sanity.assign_alpha(alpha)

        if mesh <= 0:
            raise ValueError(self.MESH_NEGATIVE)
        self.mesh = mesh
        self.spacing = None
        self.ngrid = None
        self.PDB = {}

        sanity.assign_radii()

        sanity.assign_groups(group, cluster_cut, extra_cluster_groups)

        self._assign_symmetry(symmetry)

        if(self.symmetry == 'planar'):
            sanity.assign_normal(normal)

        self.fast = fast

        pytim.PatchTrajectory(self.universe.trajectory, self)
        self._assign_layers()
        self._atoms = self._layers[:]  # this is an empty AtomGroup
        self.writevtk = pytim.willard_chandler.Writevtk(self)


    def writecube(self, filename="pytim.cube", group=None, sequence=False):
        """ Write to cube files (sequences) the volumentric density and the atomic positions.

            :param str filename: the file name
            :param bool sequence: if true writes a sequence of files adding the frame to the filename

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import MICELLE_PDB
            >>> u = mda.Universe(MICELLE_PDB)
            >>> g = u.select_atoms('resname DPC')
            >>> interface = pytim.WillardChandler(u, group=g, alpha=3.0, mesh=2.0)
            >>> interface.writecube('dens.cube') # writes on dens.cube
            >>> interface.writecube('dens.cube',group=g) # writes on dens.cube, including particles
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

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import MICELLE_PDB
            >>> u = mda.Universe(MICELLE_PDB)
            >>> g = u.select_atoms('resname DPC')
            >>> interface = pytim.WillardChandler(u, group=g, alpha=3.0, mesh=2.0)
            >>> interface.writeobj('surf.obj') # writes on surf.obj
            >>> interface.writeobj('surf.obj',sequence=True) # writes on surf.<frame>.obj
        """

        if sequence == True:
            filename = wavefront_obj.consecutive_filename(
                self.universe, filename)

        vert = self.triangulated_surface[0]
        surf = self.triangulated_surface[1]
        wavefront_obj.write_file(filename, vert, surf)

    def _assign_layers(self):
        """There are no layers in the Willard-Chandler method.

        This function identifies the dividing surface and stores the
        triangulated isosurface, the density and the particles.

        """
        self.label_group(self.universe.atoms, beta=0.0,
                         layer=-1, cluster=-1, side=-1)
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

        if self.fast == True:
            kernel.pos = pos.copy()
            self.density_field = kernel.evaluate_pbc_fast(grid)
        else:
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

class Writevtk(object):
    def __init__(self, interface):
        self.interface = interface

    def _dump_group(self, group, filename):
        """save the particles n a vtk file named consecutively using the frame
        number."""
        radii = group.radii
        types = group.types
        color = [utilities.colormap[element] for element in types]
        color = (np.array(color) / 256.).tolist()
        vtk.write_atomgroup(filename, group, color=color, radius=radii)

    def density(self, filename='pytim_dens.vtk', sequence=False):
        """ Write to vtk files the volumetric density:
            :param str filename: the file name
            :param bool sequence: if true writes a sequence of files adding the frame to the filename

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import MICELLE_PDB
            >>> u = mda.Universe(MICELLE_PDB)
            >>> g = u.select_atoms('resname DPC')
            >>> interface = pytim.WillardChandler(u, group=g, alpha=3.0, mesh=2.0)

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

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import MICELLE_PDB
            >>> u = mda.Universe(MICELLE_PDB)
            >>> g = u.select_atoms('resname DPC')
            >>> interface = pytim.WillardChandler(u, group=g, alpha=3.0, mesh=2.0)

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

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import MICELLE_PDB
            >>> u = mda.Universe(MICELLE_PDB)
            >>> g = u.select_atoms('resname DPC')
            >>> interface = pytim.WillardChandler(u, group=g, alpha=3.0, mesh=2.0)
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


#
