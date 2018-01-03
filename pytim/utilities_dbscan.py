# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
import numpy as np
from scipy.cluster import vq
from scipy.spatial import cKDTree
from pytim_dbscan import dbscan_inner


def do_cluster_analysis_DBSCAN(group,
                               cluster_cut,
                               box,
                               threshold_density=None,
                               molecular=True):
    """ Performs a cluster analysis using DBSCAN

        :returns [labels,counts]: lists of the id of the cluster to which every
                                  atom is belonging to, and of the number of
                                  elements in each cluster.

        Uses a slightly modified version of DBSCAN from sklearn.cluster
        that takes periodic boundary conditions into account (through
        cKDTree's boxsize option) and collects also the sizes of all
        clusters. This is on average O(N log N) thanks to the O(log N)
        scaling of the kdtree.

    """
    if isinstance(threshold_density, type(None)):
        min_samples = 2
    if isinstance(threshold_density, (float, int)):
        min_samples = threshold_density * 4. / 3. * np.pi * cluster_cut**3
        if min_samples < 2:
            min_samples = 2

    # NOTE: extra_cluster_groups are not yet implemented
    points = group.atoms.positions[:]

    tree = cKDTree(points, boxsize=box[:3])

    neighborhoods = np.array([
        np.array(neighbors)
        for neighbors in tree.query_ball_point(points, cluster_cut, n_jobs=-1)
    ])
    if len(neighborhoods.shape) != 1:
        raise ValueError("Error in do_cluster_analysis_DBSCAN(), the cutoff\
                          is probably too small")
    if molecular is False:
        n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
    else:
        n_neighbors = np.array([
            len(np.unique(group[neighbors].resids))
            for neighbors in neighborhoods
        ])

    if isinstance(threshold_density, str):
        if not (threshold_density == 'auto'):
            raise ValueError("Wrong value of 'threshold_density' passed\
                              to do_cluster_analysis_DBSCAN() ")
        modes = 2
        centroid, _ = vq.kmeans2(
            n_neighbors * 1.0, modes, iter=10, check_finite=False)
        # min_samples   = np.mean(centroid)
        min_samples = np.max(centroid)

    labels = -np.ones(points.shape[0], dtype=np.intp)
    counts = np.zeros(points.shape[0], dtype=np.intp)

    core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
    dbscan_inner(core_samples, neighborhoods, labels, counts)
    return labels, counts, n_neighbors
