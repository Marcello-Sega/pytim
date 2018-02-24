# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
import numpy as np
from scipy.cluster import vq
from scipy.spatial import cKDTree
from pytim_dbscan import dbscan_inner


def determine_samples(threshold_density, cluster_cut, n_neighbors):

    if isinstance(threshold_density, type(None)):
        return 2

    if isinstance(threshold_density, (float, int)):
        min_samples = threshold_density * 4. / 3. * np.pi * cluster_cut**3

    elif (threshold_density == 'auto'):
        modes = 2
        centroid, _ = vq.kmeans2(
            n_neighbors * 1.0, modes, iter=10, check_finite=False)
        min_samples = np.max(centroid)

    else:
        raise ValueError("Wrong value of 'threshold_density' passed\
                              to do_cluster_analysis_DBSCAN() ")

    return np.max([min_samples, 2])


def do_cluster_analysis_dbscan(group,
                               cluster_cut,
                               threshold_density=None,
                               molecular=True):
    """ Performs a cluster analysis using DBSCAN

        :returns [labels,counts,neighbors]: lists of the id of the cluster to 
                                  which every atom is belonging to, of the 
                                  number of elements in each cluster, and of 
                                  the number of neighbors for each atom 
                                  according to the specified criterion.

        Uses a slightly modified version of DBSCAN from sklearn.cluster
        that takes periodic boundary conditions into account (through
        cKDTree's boxsize option) and collects also the sizes of all
        clusters. This is on average O(N log N) thanks to the O(log N)
        scaling of the kdtree.

    """
    box = group.universe.dimensions[:3]

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

    min_samples = determine_samples(threshold_density, cluster_cut,
                                    n_neighbors)

    labels = -np.ones(points.shape[0], dtype=np.intp)
    counts = np.zeros(points.shape[0], dtype=np.intp)

    core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
    dbscan_inner(core_samples, neighborhoods, labels, counts)
    return labels, counts, n_neighbors


def _():
    """
    This is a collection of tests to check
    that the DBSCAN behavior is kept consistent

    >>> import MDAnalysis as mda
    >>> import pytim
    >>> pytim.utilities_dbscan._() ; # coverage
    >>> import numpy as np
    >>> from pytim.datafiles import ILBENZENE_GRO
    >>> from pytim.utilities import do_cluster_analysis_dbscan as DBScan
    >>> u = mda.Universe(ILBENZENE_GRO)
    >>> benzene = u.select_atoms('name C and resname LIG')
    >>> u.atoms.positions = u.atoms.pack_into_box()
    >>> l,c,n =  DBScan(benzene, cluster_cut = 4.5, threshold_density = None)
    >>> l1,c1,n1 = DBScan(benzene, cluster_cut = 8.5, threshold_density = 'auto')
    >>> td = 0.009
    >>> l2,c2,n2 = DBScan(benzene, cluster_cut = 8.5, threshold_density = td)
    >>> print (np.sort(c)[-2:])
    [   12 14904]

    >>> print (np.sort(c2)[-2:])
    [   0 9335]

    >>> print ((np.all(c1==c2), np.all(l1==l2)))
    (True, True)

    """
    pass
