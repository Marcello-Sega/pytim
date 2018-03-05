# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
import numpy as np
import itertools


def generate_periodic_border_for_usti(points, box, delta, periodicity):
    """ Selects the pparticles within a skin depth delta from the
        simulation box, and replicates them to mimic periodic
        boundary conditions. Returns all points (original +
        periodic copies) and the indices of the original particles
    """
    extrapoints = np.copy(points)

    shifts = np.array(list(itertools.product([1, -1, 0], repeat=3)))

    extraids = np.arange(len(points), dtype=np.int)
    for shift in shifts:
        if (np.sum(shift * shift)):  # avoid [0,0,0]
            if (periodicity[0] < 0.1 and shift[0] != 0):
                continue  # skip copying in forbidden dimensions
            if (periodicity[1] < 0.1 and shift[1] != 0): continue
            if (periodicity[2] < 0.1 and shift[2] != 0): continue

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


def PBC(vector, box):
    if (vector[0] < -box[0] / 2.0): vector[0] += box[0]
    if (vector[0] > box[0] / 2.0): vector[0] -= box[0]
    if (vector[1] < -box[1] / 2.0): vector[1] += box[1]
    if (vector[1] > box[1] / 2.0): vector[1] -= box[1]
    if (vector[2] < -box[2] / 2.0): vector[2] += box[2]
    if (vector[2] > box[2] / 2.0): vector[2] -= box[2]


def squareDistancePBC(r1, r2, box):
    rij = r2 - r1
    PBC(rij, box)
    return sum(rij[:]**2)


def isTooLarge(a, b, c, d, box):
    d1 = squareDistancePBC(a, b, box)
    d2 = squareDistancePBC(a, c, box)
    d3 = squareDistancePBC(a, d, box)
    d4 = squareDistancePBC(b, c, box)
    d5 = squareDistancePBC(b, d, box)
    d6 = squareDistancePBC(c, d, box)
    boxx2 = (box[0] / 2.0)**2
    boxy2 = (box[1] / 2.0)**2
    boxz2 = (box[2] / 2.0)**2

    sqd = np.array([d1, d2, d3, d4, d5, d6])
    if np.any(sqd > boxx2):
        return True
    if np.any(sqd > boxy2):
        return True
    if np.any(sqd > boxz2):
        return True
    return False


def isUnique(indices, tHash):
    if (indices in tHash):
        return False
    return True


def findPBCNeighboringSimplices(t, originalIndex, finalSimplices, tHash,
                                extraids):
    pbcNeighbors = np.ones(
        (len(finalSimplices), len(finalSimplices[0])), dtype=np.int)
    neighborCounter = np.zeros(len(finalSimplices), dtype=np.int)
    neighbor = 0
    #vec1=np.zeros(4)
    tup = tuple()

    for index in range(0, len(originalIndex)):
        #pbcNeighbors.append([])
        for i in range(0, 4):
            neighbor = t.neighbors[originalIndex[index], i]
            tup = tuple(
                sorted((extraids[t.simplices[neighbor, 0]],
                        extraids[t.simplices[neighbor, 1]],
                        extraids[t.simplices[neighbor, 2]],
                        extraids[t.simplices[neighbor, 3]])))
            if (tup in tHash):
                pbcNeighbors[index][neighborCounter[index]] = (tHash[tup])
                neighborCounter[index] += 1
            else:
                pbcNeighbors[index][neighborCounter[index]] = -1
                neighborCounter[index] += 1
    return pbcNeighbors


def clearPBCtriangulation(triangulation, extrapoints, extraids, box):
    finalSimplices = []
    originalIndex = []
    outx = np.zeros(4)
    outy = np.zeros(4)
    outz = np.zeros(4)
    tHash = {}
    index = 0
    tup = tuple()

    for simplex in triangulation.simplices:
        #if (isTooLarge(t.points[extraids[simplex[0]]],t.points[extraids[simplex[1]]],
        #               t.points[extraids[simplex[2]]],t.points[extraids[simplex[3]]],box)):
        #    continue
        #vec1=np.array([extraids[simplex[0]], extraids[simplex[1]], extraids[simplex[2]], extraids[simplex[3]]])
        #vec1[:]=extraids[simplex[:]]
        tup = tuple(
            sorted((extraids[simplex[0]], extraids[simplex[1]],
                    extraids[simplex[2]],
                    extraids[simplex[3]])))  #tuple(sorted(vec1))
        #is inside of original domain?
        outx[:] = 0
        outy[:] = 0
        outz[:] = 0
        for i in range(0, 4):
            if (triangulation.points[simplex[i], 0] > box[0]
                    or triangulation.points[simplex[i], 0] < 0):
                outx[i] = 1
            if (triangulation.points[simplex[i], 1] > box[1]
                    or triangulation.points[simplex[i], 1] < 0):
                outy[i] = 1
            if (triangulation.points[simplex[i], 2] > box[2]
                    or triangulation.points[simplex[i], 2] < 0):
                outz[i] = 1
        if (sum(outx) <= 2 and sum(outy) <= 2 and sum(outz) <= 2):
            if (isUnique(tup, tHash)):
                finalSimplices.append(simplex)
                originalIndex.append(index)
                #print("test",len(finalSimplices),len(t.simplices))
                #vec1[:]=extraids[simplex[:]]
                #vec1=np.array([extraids[simplex[0]], extraids[simplex[1]], extraids[simplex[2]], extraids[simplex[3]]])
                #vec1=np.sort(vec1)
                tHash[tup] = len(finalSimplices) - 1
        index += 1
    neighbors = findPBCNeighboringSimplices(triangulation, originalIndex,
                                            finalSimplices, tHash, extraids)
    finalSimplices = np.asarray(finalSimplices, dtype=np.int)
    return [finalSimplices, neighbors]
