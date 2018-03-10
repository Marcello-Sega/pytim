# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
import numpy as np


def compute_compatible_mesh_params(mesh, box):
    """ given a target mesh size and a box, return the number of grid elements
        and spacing in each direction, which are commensurate with the box
    """
    n = np.array([np.ceil(b / mesh) for b in box])
    d = box / n
    return n, d


def generate_grid_in_box(box, npoints, order='zxy'):
    """generate an homogenous grid of npoints^3 points that spans the
       complete box.

       :param ndarray box: the simulation box edges
       :param ndarray npoints: the number of points along each direction

    """
    xyz = []
    for i in range(3):
        xyz.append(np.linspace(0., box[i] - box[i] / npoints[i], npoints[i]))
    if order == 'zyx':
        z, y, x = np.meshgrid(xyz[0], xyz[1], xyz[2], indexing='ij')
    else:
        x, y, z = np.meshgrid(xyz[2], xyz[1], xyz[0], indexing='ij')

    grid = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)
    grid = np.append(grid, z.reshape(-1, 1), axis=1)
    return grid.T
