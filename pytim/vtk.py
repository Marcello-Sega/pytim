# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
from . import utilities
import numpy as np


class Writevtk(object):
    def __init__(self, interface):
        self.interface = interface

    def _dump_group(self, group, filename):
        """ Save the particles n a vtk file named consecutively using the frame
            number.
        """
        radii, types, color = group.radii, group.types, []
        for element in types:
            try:
                c = (utilities.atoms_maps[element])['color']
            except KeyError:  # defaults to Carbon
                c = (utilities.atoms_maps['C'])['color']
            color.append(c)
        color = (np.array(color) / 256.).tolist()
        write_atomgroup(filename, group, color=color, radius=radii)

    def density(self, filename='pytim_dens.vtk', sequence=False):
        """ Write to vtk files the volumetric density.

            :param str filename:  the file name
            :param bool sequence: if true writes a sequence of files adding
                                  the frame to the filename

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import MICELLE_PDB
            >>> u = mda.Universe(MICELLE_PDB)
            >>> g = u.select_atoms('resname DPC')
            >>> inter= pytim.WillardChandler(u, group=g, alpha=3.0, mesh=2.0)

            >>> inter.writevtk.density('dens.vtk') # writes on dens.vtk
            >>> inter.writevtk.density('dens.vtk',sequence=True) # dens.<n>.vtk

        """
        inter = self.interface
        if sequence is True:
            filename = consecutive_filename(inter.universe, filename)
        write_scalar_grid(filename, inter.ngrid, inter.spacing,
                          inter.density_field)

    def particles(self, filename='pytim_part.vtk', group=None, sequence=False):
        """ Write to vtk files the particles in a group.

            :param str filename:    the file name
            :param bool sequence:   if true writes a sequence of files adding
                                    the frame to the filename
            :param AtomGroup group: if None, writes the whole universe

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import MICELLE_PDB
            >>> u = mda.Universe(MICELLE_PDB)
            >>> g = u.select_atoms('resname DPC')
            >>> inter= pytim.WillardChandler(u, group=g, alpha=3.0, mesh=2.0)

            >>> # writes on part.vtk
            >>> inter.writevtk.particles('part.vtk')
            >>> # writes on part.<frame>.vtk
            >>> inter.writevtk.particles('part.vtk',sequence=True)
        """

        inter = self.interface
        if sequence is True:
            filename = consecutive_filename(inter.universe, filename)
        if group is None:
            group = inter.universe.atoms
        self._dump_group(group, filename)

    def surface(self, filename='pytim_surf.vtk', sequence=False):
        """ Write to vtk files the triangulated surface.

            :param str filename:  the file name
            :param bool sequence: if true writes a sequence of files adding
                                  the frame to the filename

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import MICELLE_PDB
            >>> u = mda.Universe(MICELLE_PDB)
            >>> g = u.select_atoms('resname DPC')
            >>> inter= pytim.WillardChandler(u, group=g, alpha=3.0, mesh=2.0)
            >>> inter.writevtk.surface('surf.vtk') # writes on surf.vtk
            >>> inter.writevtk.surface('surf.vtk',sequence=True) # surf.<n>.vtk
        """
        inter = self.interface
        vertices, faces, normals = list(inter.triangulated_surface[0:3])
        if sequence is True:
            filename = consecutive_filename(inter.universe, filename)
        write_triangulation(filename, vertices, faces, normals)


def _format_vector(vector, format_str="{:f}"):
    formatted = ''
    for element in vector:
        formatted += format_str.format(element) + ' '
    return formatted


def write_scalar_grid(filename, grid_size, spacing, scalars):
    """write in a vtk file a scalar field on a rectangular grid

       :param string filename: the filename
       :param array grid_size: number of points in the grid along each
                               direction
       :param array spacing  : a (3,) array with the point spacing along the 3
                               directions
       :param array scalars  : a (grid_size,) array with the scalar field
                               values
    """
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 2.0\nscalar\nASCII\n")
        f.write("DATASET STRUCTURED_POINTS\nDIMENSIONS ")
        f.write(
            _format_vector(
                np.asarray(grid_size, dtype=int), format_str="{:d}") + "\n")
        f.write("SPACING " + _format_vector(spacing) + "\n")
        f.write("\n")
        f.write("ORIGIN " + _format_vector(spacing / 2.) + "\n")
        f.write("POINT_DATA " + str(len(scalars)) + "\n")
        f.write("SCALARS kernel floats 1\nLOOKUP_TABLE default\n")

        for val in scalars:
            f.write(str(val) + "\n")


# TODO: should move  all AtomGroup references to a higher level


def write_atomgroup(filename, group, color=None, radius=None):
    """ write in a vtk file the positions of particles

        :param string filename: the filename
        :param AtomGroup group: the group, whose positions are to be written to
                                the vtk file
        :param ndarray color (N,3): optional: array with triplets of RGB values
                                for each atom
        :param ndarray raidus : optional: array with atomic radii
    """
    pos = group.positions
    npos = len(pos)
    with open(filename, "w") as f:
        f.write(
            "# vtk DataFile Version 2.0\ntriangles\nASCII\nDATASET POLYDATA\n")
        f.write("POINTS " + str(len(pos)) + " floats\n")
        for p in pos:
            f.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")
        f.write("\nVERTICES " + str(len(pos)) + " " + str(len(pos) * 2) + "\n")
        for i in range(npos):
            f.write("1 " + str(i) + "\n")
        if radius is not None:
            f.write(
                "\nPOINT_DATA " + str(len(pos)) + "\nSCALARS radius float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for rad in radius:
                f.write(str(rad) + "\n")
        if color is not None:
            f.write("COLOR_SCALARS color 3\n")
            for c in color:
                f.write(_format_vector(c, format_str="{:1.2f}") + "\n")


def write_triangulation(filename, vertices, triangles, normals=None):
    """ write in a vtk file a triangulation

        :param string filename: the filename
        :param array vertices: (N,3) array of floats for N vertices
        :param array triangles: (M,3) array of indices to the vertices
        :param array triangles: (M,3) array of normal vectors
    """
    f = open(filename, "w")
    f.write("# vtk DataFile Version 2.0\nkernel\nASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")
    f.write("POINTS " + str(len(vertices)) + " float\n")
    for point in vertices:
        f.write(_format_vector(point) + "\n")

    f.write("\nCELLS " + str(len(triangles)) + " " + str(4 * len(triangles)) +
            "\n")
    for index in triangles:
        f.write("3 " + _format_vector(index, format_str="{:d}") + "\n")

    f.write("\nCELL_TYPES " + str(len(triangles)) + "\n")
    f.write("5\n" * len(triangles))

    if normals is not None:
        f.write("\nPOINT_DATA " + str(len(vertices)) + "\n")
        f.write("NORMALS normals float\n")
        for n in normals:
            f.write(_format_vector(n, format_str="{:f}") + "\n")


def consecutive_filename(universe, basename):
    if basename.endswith('.vtk'):
        basename = basename[:-4]
    return utilities.consecutive_filename(universe, basename, 'vtk')
