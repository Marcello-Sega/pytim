# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from pytim import utilities
import numpy as np

Bohr=.52917721067

def consecutive_filename(universe, basename):
    utilities.consecutive_filename(universe, basename,'cube')


def _write_atomgroup(f, group, atomic_numbers):
    """save the particles in a cube file named consecutively using the frame
    number. If the atom type string does not correspond to any symbol in the
    periodic table, zero is assigned. Note that this method does not guarantee

    It assumes that the header has already been written.

    :param file f: the file
    :param AtomGroup group: the atom group to be written to the cube file
    :param array atomic_numbers: the atomic number identifying the element of each atom, or None for automatic choice

    """

    types = group.types
    if atomic_numbers is None:
        try:
            atomic_numbers = [el for el in group.elements ]
        except:
            atomic_numbers  = [utilities.atomic_number_map.get(t,0) for t in types]

    for i, p in enumerate(group.positions):
        f.write(_format_atom(p/Bohr,atomic_numbers[i]))


def _format_atom(pos, number, format_str="{:d} {:g} {:g} {:g} {:g}"):
    return format_str.format(number,number,pos[0],pos[1],pos[2]) + '\n'


def _write_scalar_grid(f, scalars):
    """write in a cube file a scalar field on a rectangular grid.
       Assumes that the header has been already written.

       :param file f: the file
       :param array grid_size: number of points in the grid along each\
                               direction
       :param array spacing: a (3,) array with the point spacing along the 3\
                             directions
       :param array scalars: a (grid_size,) array with the scalar field values
    """
    for count, val in enumerate(scalars):
        f.write('{:E}'.format(val) + ' ')
        if count%6 == 5:
            f.write('\n')
    if count%6 != 5:
        f.write('\n')
    f.close()


def write_file(filename, group, grid_size, spacing,
               scalars,atomic_numbers=None,write_atoms=True,normalize=True):
    """write in a cube file a scalar field on a rectangular grid.
       Assumes that the header has been already written.

       :param string filename: the filename
       :param array grid_size: number of points in the grid along each\
                               direction
       :param array spacing: a (3,) array with the point spacing along the 3\
                             directions
       :param array scalars: a (grid_size,) array with the scalar field values
    """
    if normalize :
        maxval = np.max(scalars)
    else:
        maxval = 1.0

    spacing = np.array(spacing) / Bohr  # CUBE units are Bohr
    f = open(filename, "w")
    if write_atoms:
        natoms = len(group.atoms)
    else:
        natoms = 0
    f.write('CPMD CUBE FILE\n')
    f.write('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
    f.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(natoms,0.,0.,0.))
    # NOTE: only rectangular boxes so far
    f.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(grid_size[0],spacing[0],0.,0.))
    f.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(grid_size[1],0.,spacing[0],0.))
    f.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(grid_size[2],0.,0.,spacing[0]))
    if write_atoms:
        _write_atomgroup(f, group, atomic_numbers)
    _write_scalar_grid(f, scalars/maxval)
    f.close()

def consecutive_filename(universe, basename):
    return utilities.consecutive_filename(universe, basename,'cube')

