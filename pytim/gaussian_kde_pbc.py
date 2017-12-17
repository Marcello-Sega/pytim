# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: pytim.gaussian_kde_pbc
    ==============================
"""
import numpy as np
from scipy.stats import gaussian_kde


class gaussian_kde_pbc(gaussian_kde):
    # note that here "points" are those on the grid

    def search(self, p, grid, d):
        condition_x = np.logical_and(
            grid[0] > p[0] - d,
            grid[0] < p[0] + d
        )

        condition_y = np.logical_and(
            grid[1] > p[1] - d,
            grid[1] < p[1] + d
        )
        condition_z = np.logical_and(
            grid[2] > p[2] - d,
            grid[2] < p[2] + d
        )
        condition = np.logical_and(
            condition_z, np.logical_and(condition_x, condition_y))
        return np.where(condition)

    def evaluate_pbc_fast(self, points):
        grid = points
        pos = self.pos
        box = self.box
        d = self.sigma * 2.5
        results = np.zeros(grid.shape[1], dtype=float)
        periodic = np.copy(pos)

        for side, condition in enumerate([pos > box - d, pos < d]):
            pos_ = pos.copy()
            where = np.where(condition)
            if side == 0:
                pos_[where] -= box[where[1]]
                periodic = np.copy(pos_[np.any(condition, axis=1)])
            else:
                pos_[where] += box[where[1]]
                periodic = np.append(
                    periodic, pos_[np.any(condition, axis=1)], axis=0)
            periodic = np.append(pos, periodic, axis=0)

        for p in periodic:
            ind = self.search(p, grid, d)[0]

            x = grid[0][ind] - p[0]
            y = grid[1][ind] - p[1]
            z = grid[2][ind] - p[2]
            results[ind] += np.exp(-(x**2 + y**2 + z**2) / self.sigma**2 / 2.)

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

        result = np.zeros((m,), dtype=float)

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
