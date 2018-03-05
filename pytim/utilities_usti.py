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


class Cluster():
    def __init__(self, clusterGroup, box):
        self.tetrahedrons = []
        self.atomIndices = []
        self.neighboringAtoms = {}
        self.vertices = {}
        self.__interfaces = []
        self._interface = []
        self.__layers = []
        self.clusterGroup = clusterGroup
        self._volume = 0
        self._surfaces = []
        self.box = box

    def appendTetrahedron(self, index, tetrahedron, extraids, points):
        self.tetrahedrons.append(index)
        self.vertices[index] = (points[extraids[tetrahedron[0]]],
                                points[extraids[tetrahedron[1]]],
                                points[extraids[tetrahedron[2]]],
                                points[extraids[tetrahedron[3]]])
        for i in range(0, 4):
            index1 = extraids[tetrahedron[i]]
            if (not index1 in self.neighboringAtoms):
                self.atomIndices.append(index1)
                self.neighboringAtoms[index1] = []
            self.appendAtomNeighbors(
                index1, (extraids[tetrahedron[0]], extraids[tetrahedron[1]],
                         extraids[tetrahedron[2]], extraids[tetrahedron[3]]))

    def appendAtomNeighbors(self, index1, index2):
        for i in index2:
            if i == index1:
                continue
            if (not i in self.neighboringAtoms[index1]):
                self.neighboringAtoms[index1].append(i)

    def computeVolumeOfCluster(self):
        volume = 0
        vec1 = np.zeros(3)
        vec2 = np.zeros(3)
        vec3 = np.zeros(3)

        for t in self.tetrahedrons:
            vec1[0] = self.vertices[t][0][0] - self.vertices[t][3][0]
            vec1[1] = self.vertices[t][0][1] - self.vertices[t][3][1]
            vec1[2] = self.vertices[t][0][2] - self.vertices[t][3][2]
            vec2[0] = self.vertices[t][1][0] - self.vertices[t][3][0]
            vec2[1] = self.vertices[t][1][1] - self.vertices[t][3][1]
            vec2[2] = self.vertices[t][1][2] - self.vertices[t][3][2]
            vec3[0] = self.vertices[t][2][0] - self.vertices[t][3][0]
            vec3[1] = self.vertices[t][2][1] - self.vertices[t][3][1]
            vec3[2] = self.vertices[t][2][2] - self.vertices[t][3][2]

            utilities.PBC(vec1, self.box)
            utilities.PBC(vec2, self.box)
            utilities.PBC(vec3, self.box)

            vec2 = np.cross(vec2, vec3)
            volume += abs(np.dot(vec1, vec2)) / 6.0

        return volume

    def computeSurfaceOfCluster(self, interfaceIndex):
        surface = 0
        try:
            for t in self.__interfaces[interfaceIndex]:
                surface += self.interface[t].surface(self.box)
            return surface
        except ValueError:
            print("There is no interface with index: ", interfaceIndex)
            return 0

    @property
    def clusterDimension(self):
        return self._clusterDimension

    @clusterDimension.setter
    def clusterDimension(self, dimension):
        self._clusterDimension = dimension

    @property
    def clusterDensity(self):
        return self._clusterDensity

    @clusterDensity.setter
    def clusterDensity(self, _density):
        self._clusterDensity = _density

    @property
    def interfaces(self):
        return self.__interfaces

    @interfaces.setter
    def interfaces(self, value):
        for i in value:
            self.__interfaces.append(i)

    @property
    def interface(self):
        return self._interface

    @interface.setter
    def interface(self, value):
        self._interface = value

    @property
    def layers(self):
        return self.__layers

    @layers.setter
    def layers(self, value):
        for i in value:  #loop over interfaces
            self.__layers.append(i)


class Triangle():
    def __init__(self, A, B, C, pointA, pointB, pointC):
        self.A = A
        self.B = B
        self.C = C
        self.pointA = pointA
        self.pointB = pointB
        self.pointC = pointC

    def isNeighborOf(self, triangle, extraids):
        if (int(extraids[self.A] == extraids[triangle.A]
                or extraids[self.A] == extraids[triangle.B]
                or extraids[self.A] == extraids[triangle.C]) +
                int(extraids[self.B] == extraids[triangle.A]
                    or extraids[self.B] == extraids[triangle.B]
                    or extraids[self.B] == extraids[triangle.C]) +
                int(extraids[self.C] == extraids[triangle.A]
                    or extraids[self.C] == extraids[triangle.B]
                    or extraids[self.C] == extraids[triangle.C]) == 2):
            return True
        return False

    def surface(self, box):
        rij1 = np.zeros(3)
        rij2 = np.zeros(3)
        vec = np.zeros(3)
        surface = 0

        rij1[:] = self.pointB[:] - self.pointA[:]
        rij2[:] = self.pointC[:] - self.pointA[:]

        utilities.PBC(rij1, box)
        utilities.PBC(rij2, box)

        vec = np.cross(rij1, rij2)
        return 0.5 * math.sqrt(
            vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
