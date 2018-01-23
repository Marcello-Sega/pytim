# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

from __future__ import print_function
import numpy as np


def trim_triangulated_surface(tri, box):
    """ Reduce a surface triangulation that has been extended to\
        allow for periodic boundary conditions\
            to the primary cell.

            The simplices within the primary cell are those with at\
            least two vertices within the cell boundaries.

        :param Delaunay tri: a 2D triangulation
        :param ndarray  box: box cell parameters
        :returns ndarray simplices: the simplices within the primary cell.
    """
    condition = np.logical_and(tri.points[tri.simplices] <= box[0:2],
                               tri.points[tri.simplices] > [0, 0])
    sum1 = condition.sum(axis=2)
    return tri.simplices[np.where((sum1 >= 2).sum(axis=1) >= 2)]


def triangulated_surface_stats(tri2d, points3d):
    """ Return basic statistics about a surface triangulation

        Implemented statistics are: surface area

        :param ndarray tri2d : indices of triangles vertices
        :param ndarray points3d : the heigth of each vertex along the third\
                                  dimension
        :returns list stats : the statistics :  [surface_area]
    """

    # NOTE: would it be possible to write a more efficient routine ?
    # some advanced indexing here...
    # points3d[reduced] is an array of shape (x,3,3)
    # we need to subtract the first of the three vectors
    # from each of them. This uses numpy.newaxis
    v = points3d[tri2d] - points3d[tri2d][:, 0][:, None]
    # then we need to make the cross product of the two
    # non-zero vectors of each triplet
    area = np.linalg.norm(np.cross(v[:, 1], v[:, 2]), axis=1).sum() / 2.
    return [area]


def polygonalArea(points):
    """ Calculate the area of a polygon from ordered points in 3D space.

        :param ndarray points: a (N,3) array
        :returns double: area

        Example:

        >>> import numpy as np
        >>> import pytim
        >>> from pytim.utilities import *
        >>> # Vertices of a pentagon and its area
        >>> c1 = np.cos(2*np.pi/5.) ; c2 = np.cos(np.pi/5.)
        >>> s1 = np.sin(2*np.pi/5.) ; s2 = np.sin(4*np.pi/5.)
        >>> pentagon = np.array([[1,0,0],[c1,s1,0],[-c2,s2,0],[-c2,-s2,0],[c1,-s1,0]])
        >>> A = 0.25 * np.sqrt(25+10*np.sqrt(5)) * 100./ (50+10*np.sqrt(5))
        >>> np.isclose(pytim.utilities.polygonalArea(pentagon),A)
        True

        >>> # now let's rotate it:
        >>> rotated = np.dot(EulerRotation(0,np.pi/2.,0),pentagon.T).T
        >>> np.isclose(pytim.utilities.polygonalArea(rotated),A)
        True

     """

    try:
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        n = np.cross(v1, v2)
    except BaseException:
        raise RuntimeError("Not enough or collinear points in polygonalArea()")
    n2 = np.sum(n**2)
    count = 0
    while n[2]**2 / n2 < 1e-3 and count < 3:
        points = np.roll(points, 1, axis=1)
        n = np.roll(n, 1)
        count += 1
    if count >= 3:
        Warning("Degenerate surface element encountered")
        return 0.0
    points2d = points[:, :2]
    nz2 = n[2]**2
    ratio = np.sqrt(n2 / nz2)
    return np.abs(ratio * np.sum([0.5, -0.5] * points2d * np.roll(
        np.roll(points2d, 1, axis=0), 1, axis=1)))


def pbc_compact(pos1, pos2, box):
    """  wraps points so that satisfy the minimum image
         convention with respect to a reference point

         :param ndarray pos1: an (N,3) array of positions
         :param ndarray pos2: either a point (3,) or an positions array (N,3)
         :param ndarray box: (3,) array with the rectangular box edges' length
         :return ndarray: the modified positions

    """

    cond_pbc = np.where(pos1 - pos2 >= box / 2)
    pos1[cond_pbc] -= box[cond_pbc[1]]
    cond_pbc = np.where(pos1 - pos2 < -box / 2)
    pos1[cond_pbc] += box[cond_pbc[1]]
    return pos1


def pbc_wrap(pos, box):
    """  wraps points so they are always in the simulation box

         :param ndarray pos: an (N,3) array of positions
         :param ndarray box: (3,) array with the rectangular box edges' length
         :return ndarray: the modified positions

    """

    cond_pbc = np.where(pos > box)
    pos[cond_pbc] -= box[cond_pbc[1]]
    cond_pbc = np.where(pos < 0.0)
    pos[cond_pbc] += box[cond_pbc[1]]
    return pos


def find_surface_triangulation(interface):
    """
        identifies all triangles which are part of the surface
        :param GITIM interface: a GITIM interface instance
        :returns ndarray: (N,3) indices of the triangles' vertices
    """
    intr = interface
    cond = intr.atoms.layers == 1
    layer_1 = intr.atoms[cond]
    tri = None
    for roll in [0, 1, 2, 3]:
        # slimplices[i] goes from 0 -> len(cluster_group) + periodic copies
        # layer_1_ids links the atoms in the 1st layer to the indexing of
        # simplices's points
        layer_1_ids = np.argwhere(
            np.isin(intr.cluster_group.indices, layer_1.indices))
        rolled = np.roll(intr.triangulation[0].simplices, 0, axis=1)[:, :3]
        # requires that triplets of points in the simplices belong to the 1st
        # layer
        select = np.argwhere(np.all(np.isin(rolled, layer_1_ids),
                                    axis=1)).flatten()
        if tri is None:
            tri = rolled[select]
        else:
            tri = np.append(tri, rolled[select], axis=0)
    return tri


def fit_sphere(points):
    """ least square fit of a sphere through a set of points.

        :param ndarray points: a (N,3) array
        :returns list: radius, center_x, center_y, center_z

    """

    px = points[::, 0]
    py = points[::, 1]
    pz = points[::, 2]
    f = np.sum(points * points, axis=1)
    A = np.zeros((len(points), 4))
    A[:, 0] = 2. * px
    A[:, 1] = 2. * py
    A[:, 2] = 2. * pz
    A[:, 3] = 1.
    C = np.dot(np.linalg.pinv(A), f)
    radius = np.sqrt(C[0] * C[0] + C[1] * C[1] + C[2] * C[2] + C[3])
    return radius, C[0], C[1], C[2]


def EulerRotation(phi, theta, psi):
    """ The Euler (3,1,3) rotation matrix

        :param phi: rotation around the z axis
        :param theta: rotation around the new x axis
        :param psi: rotation around the new z axis
        :returns double: area

        Example:

        >>> import numpy as np
        >>> import pytim
        >>> from pytim.utilities import *
        >>> np.set_printoptions(suppress=True)
        >>> print (EulerRotation(np.pi/2.,0,0))
        [[ 0.  1.  0.]
         [-1.  0.  0.]
         [ 0. -0.  1.]]
        >>> np.set_printoptions(suppress=False)

    """

    cph = np.cos(phi)
    cps = np.cos(psi)
    cth = np.cos(theta)
    sph = np.sin(phi)
    sps = np.sin(psi)
    sth = np.sin(theta)
    R1 = [cph * cps - cth * sph * sps, cps * sph + cth * cph * sps, sth * sps]
    R2 = [-cth * cps * sph - cph * sps, cth * cps * cph - sph * sps, cps * sth]
    R3 = [sth * sph, -cph * sth, cth]
    return np.array([R1, R2, R3])
