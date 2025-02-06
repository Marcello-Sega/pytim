# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: pytim.gaussian_kde_pbc
    ==============================
"""
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

if os.environ.get("PYTIM_BACKEND") == "scipy":
    has_jax = False
    print("Using scipy backend")
else:
    try:
        import jax.numpy as jnp
        from kdecv.calculate import GaussianKDE
        has_jax=True
        print("Using JAX backend")
    except ImportError:
        has_jax = False
        print("Using scipy backend")

class gaussian_kde_pbc(gaussian_kde):
    # note that here "points" are those on the grid

    def __init__(self, pos, box, sigma, weights=None):
        """ Initialize the gaussian_kde_pbc object.
        
        :param ndarray pos: the particle positions.
        :param ndarray box: the box dimensions.
        :param float sigma: the sigma value.
        :param ndarray weights: the weights of the points in the dataset.

        """
        self.box = box
        self.pos = pos
        self.sigma = sigma

        dataset = np.vstack([pos[::, 0], pos[::, 1], pos[::, 2]])

        self.stddev = dataset.std(ddof=1)
        self.bw = sigma / self.stddev

        super().__init__(dataset, bw_method=self.bw, weights=weights)

    def evaluate(self, points):
        """ Evaluate the estimated pdf on a set of points.

        :param ndarray points: the points where the pdf is to be evaluated.
        :return: the value of the estimated pdf at the points.
        :rtype: ndarray

        """
        if has_jax:
            kde = GaussianKDE(self.pos, self.box, self.bw, self.weights)
            return np.array(kde.evaluate(points.T).flatten())

        else:
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
