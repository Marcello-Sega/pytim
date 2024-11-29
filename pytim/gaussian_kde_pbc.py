# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: pytim.gaussian_kde_pbc
    ==============================
"""
import numpy as np
from scipy.spatial import cKDTree

try:
    from jax.scipy.stats import gaussian_kde
    has_jax = True
except ImportError:
    from scipy.stats import gaussian_kde
    has_jax = False

def make_supercell_images(pos, box, n_images=1):
    """ Generate supercell images of a set of points

        :param np.ndarray pos:  positions of the points
        :param np.ndarray box:  box dimensions
        :param int n_images:    number of images to generate

    """
    images = []
    for i in range(-n_images, n_images + 1):
        for j in range(-n_images, n_images + 1):
            for k in range(-n_images, n_images + 1):
                images.append(pos + np.array([i, j, k]) * box)
    return np.vstack(images)


class gaussian_kde_pbc(gaussian_kde):
    # note that here "points" are those on the grid

    def __init__(self, pos, box, sigma, weights=None, use_jax = False):
        """ Initialize the gaussian_kde_pbc object.
        
        :param ndarray pos: the particle positions.
        :param ndarray box: the box dimensions.
        :param float sigma: the sigma value.
        :param ndarray weights: the weights of the points in the dataset.

        """
        self.box = box
        self.pos = pos
        self.sigma = sigma
        self.use_jax = use_jax
        dataset = np.vstack([pos[::, 0], pos[::, 1], pos[::, 2]])

        if has_jax and self.use_jax:
            supercell = make_supercell_images(pos, box=box)
            dataset = np.vstack([supercell[::, 0], supercell[::, 1], supercell[::, 2]])

        self.stddev = dataset.std(ddof=1)
        bw_method = sigma / self.stddev

        super().__init__(dataset, bw_method=bw_method, weights=weights)

    def evaluate(self, points):
        """ Evaluate the estimated pdf on a set of points.

        :param ndarray points: the points where the pdf is to be evaluated.
        :return: the value of the estimated pdf at the points.
        :rtype: ndarray

        """
        if has_jax and self.use_jax:
            print('evaluating using jax')
            return np.array(super().evaluate(points))

        else:
            print('evaluating using pytim implementation')
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
