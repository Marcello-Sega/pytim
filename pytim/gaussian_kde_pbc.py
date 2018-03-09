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
        gridT = grid.T[:]
        tree = cKDTree(gridT,boxsize=box)
        # the indices of grid elements within distane d from each of the pos
        scale  = 2.*self.sigma**2
        indlist  = tree.query_ball_point(pos,d)
        for n,ind in enumerate(indlist):
            dr = gridT[ind,:] - pos[n]
            cond = np.where(dr > box/2.)
            dr [cond] -= box[cond[1]]
            cond = np.where(dr < -box/2.)
            dr [cond] += box[cond[1]]
            dens = np.exp( - np.sum(dr*dr,axis=1) / scale )
            results[ind] += dens
        
        return results

    def evaluate_pbc(self, points):
        """ PBC-enabled version of scipy.stats.gaussian_kde.evaluate()
        """

        points = np.atleast_2d(points)
        box = self.box
        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (
                    d, self.d)
                raise ValueError(msg)

        result = np.zeros((m, ), dtype=float)

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:, i, np.newaxis] - points
                diff = diff.T
                diff -= (diff > box / 2.) * box
                diff += (diff < -box / 2.) * box
                diff = diff.T
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result = result + np.exp(-energy)
        else:
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:, i, np.newaxis]
                diff = diff.T
                diff -= (diff > box / 2.) * box
                diff += (diff < -box / 2.) * box
                diff = diff.T
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result[i] = np.sum(np.exp(-energy), axis=0)

        result = result / self._norm_factor

        return result
