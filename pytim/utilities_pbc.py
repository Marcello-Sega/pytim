# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
import numpy as np
import itertools


def generate_periodic_border(points, box, delta, method='3d'):
    """ Selects the pparticles within a skin depth delta from the
        simulation box, and replicates them to mimic periodic
        boundary conditions. Returns all points (original +
        periodic copies) and the indices of the original particles
    """
    extrapoints = np.copy(points)

    if method is '2d':
        shifts = np.array([
            el + (0, ) for el in list(itertools.product([1, -1, 0], repeat=2))
        ])
    else:
        shifts = np.array(list(itertools.product([1, -1, 0], repeat=3)))

    extraids = np.arange(len(points), dtype=np.int)
    for shift in shifts:
        if (np.sum(shift * shift)):  # avoid [0,0,0]
            # this needs some explanation:
            # if shift ==0  -> the condition is always true
            # if shift ==1  -> the condition is x > box - delta
            # if shift ==-1 -> the condition is -x > 0 - delta -> x <delta
            # Requiring np.all() to be true makes the logical and returns
            # (axis=1) True for all indices whose atoms satisfy the
            # condition
            selection = np.all(
                shift * points >= shift * shift *
                ((box + shift * box) / 2. - delta),
                axis=1)
            # add the new points at the border of the box
            extrapoints = np.append(
                extrapoints, points[selection] - shift * box, axis=0)
            # we keep track of the original ids.
            extraids = np.append(extraids, np.where(selection)[0])
    return extrapoints, extraids


def rebox(pos, edge, shift):
    """ rebox a vector along one dimension
        :param ndarray pos: the array of components to be reboxed
        :param float edge: the simulation box edge
        :param float shift: additional shift
    """
    condition = pos >= edge - shift
    pos[condition] -= edge

    condition = pos < 0 - shift
    pos[condition] += edge

    return pos
