# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: pytim.gaussian_kde_pbc
    ==============================
"""
from __future__ import print_function
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree


class gaussian_kde_pbc(gaussian_kde):
    # note that here "points" are those on the grid

    def evaluate_pbc_fast(self, points):
        grid = points
        pos = self.pos
        box = self.box
        d = self.sigma * 2.5
        results = np.zeros(grid.shape[1], dtype=float)
        gridT = grid[::-1].T[:]
        tree = cKDTree(gridT, boxsize=box)
        # the indices of grid elements within distane d from each of the pos
        scale = 2. * self.sigma**2
        indlist = tree.query_ball_point(pos, d)
        for n, ind in enumerate(indlist):
            dr = gridT[ind, :] - pos[n]
            cond = np.where(dr > box / 2.)
            dr[cond] -= box[cond[1]]
            cond = np.where(dr < -box / 2.)
            dr[cond] += box[cond[1]]
            dens = np.exp(-np.sum(dr * dr, axis=1) / scale)
            results[ind] += dens

        return results
