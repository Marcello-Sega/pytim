# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import numpy as np
from __future__ import print_function


def compute_compatible_mesh_params(mesh, box):
    """ given a target mesh size and a box, return the number of grid elements
        and spacing in each direction, which are commensurate with the box
    """
    n = list(map(int, np.ceil(box / mesh)))
    d = box / n
    return n, d


def generate_grid_in_box(box, npoints, order='zxy'):
    """generate an homogenous grid of npoints^3 points that spans the
       complete box.

       :param ndarray box: the simulation box edges
       :param ndarray npoints: the number of points along each direction

    """

    x_ = np.linspace(0., box[0], npoints[0])
    y_ = np.linspace(0., box[1], npoints[1])
    z_ = np.linspace(0., box[2], npoints[2])
    if order == 'zyx':
        z, y, x = np.meshgrid(z_, y_, x_, indexing='ij')
    else:
        x, y, z = np.meshgrid(z_, y_, x_, indexing='ij')

    grid = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)
    grid = np.append(grid, z.reshape(-1, 1), axis=1)
    return grid.T
