# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: pytim.gaussian_kde_pbc
    ==============================
"""
import numpy as np

try:
    from jax.scipy.stats import gaussian_kde
except ImportError:
    from scipy.stats import gaussian_kde


class gaussian_kde_pbc(gaussian_kde):
    # note that here "points" are those on the grid

    def evaluate(self, points):
        """ Evaluate the estimated pdf on a set of points.

        :param ndarray points: the points where the pdf is to be evaluated.
        :return: the value of the estimated pdf at the points.
        :rtype: ndarray

        """
        return np.array(super().evaluate(points))
