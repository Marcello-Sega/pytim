# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
from . import utilities
import numpy as np

Bohr = .52917721067


def consecutive_filename(universe, basename):
    if basename.endswith('.cube'):
        basename = basename[:-4]
    return utilities.consecutive_filename(universe, basename, 'cube')


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
        n0 = {'number': 0}
        atomic_numbers = [utilities.atoms_maps.get(t, n0)['number'] for t in types]

    for i, p in enumerate(group.positions):
        f.write(_format_atom(p / Bohr, atomic_numbers[i]))


def _format_atom(pos, number, format_str="{:d} {:g} {:g} {:g} {:g}"):
    return format_str.format(int(number), float(number), pos[0], pos[1], pos[2]) + '\n'


def _write_scalar_grid(f, scalars):
    """write in a cube file a scalar field on a rectangular grid.
       Assumes that the header has been already written.

       :param file f: the file
       :param array scalars: a (grid_size,) array with the scalar field values
    """
    for count, val in enumerate(scalars):
        f.write('{:E}'.format(val) + ' ')
        if count % 6 == 5:
            f.write('\n')
    if count % 6 != 5:
        f.write('\n')


def write_file(filename,
               group,
               grid_size,
               spacing,
               scalars,
               order='zyx',
               shift=[0,0,0],
               atomic_numbers=None,
               normalize=True):
    """write in a cube file a scalar field on a rectangular grid.
       Assumes that the header has been already written.

       :param string filename     : the filename
       :param array grid_size     : number of points in the grid along each
                                    direction
       :param array spacing       : a (3,) array with the point spacing along the 3
                                    directions
       :param array scalars       : a (grid_size,) array with the scalar field
                                    values
       :param string order        : 'xyz' or 'zyx', to adapt to the software used
                                    for visualization. Default: 'zyx'
       :param array shift         : add shift in grid units. Default: [0,0,0]
       :param array atomic_numbers: the atomic number identifying the element of each atom,
                                    or None for automatic choice. Default: None
       :param bool normalize      : if True, rescale to the maximum value. Default: True

    """
    if normalize:
        maxval = np.max(scalars)
    else:
        maxval = 1.0
    spacing = np.array(spacing) / Bohr  # CUBE units are Bohr
    with open(filename, "w") as f:
        if group is not None:
            natoms = len(group.atoms)
        else:
            natoms = 0
        grid_size = np.array(grid_size, dtype=int)
        shift = -np.array(shift) * spacing * Bohr
        #spacing =  spacing*(grid_size+1.)/grid_size
        f.write('CPMD CUBE FILE\n')
        f.write('GENERATED BY PYTIM\n')
        f.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(
            natoms, shift[0], shift[1], shift[2]))
        # NOTE: only rectangular boxes so far
        f.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(
            grid_size[0], spacing[0], 0., 0.))
        f.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(
            grid_size[1], 0., spacing[1], 0.))
        f.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(
            grid_size[2], 0., 0., spacing[2]))
        if group is not None:
            _write_atomgroup(f, group, atomic_numbers)

        field = scalars.reshape((grid_size[2], grid_size[1], grid_size[0]))
        if order == 'zyx': field = np.swapaxes(field, 2, 0) # xyz <-> zyx

        # NOTE the 6-fold roll has been determined empirically.
        field = np.roll(field,6,axis=-1).flatten()
        _write_scalar_grid(f, field / maxval)
