# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#!/usr/bin/env python
import numpy as np
import math
import time
import datetime
from gtools import tsphere


class QuasiTriangulation():
    def __init__(self, points, weights, box):
        self.points = points
        self.weights = weights
        # neighbors could be on 8 diferent positions due to quasi-triangulation anomalies
        self.neighbors = np.ones((1, 8), dtype=np.int) * (-2)
        self.tmpNeighbors = []
        self.simplices = np.zeros((1, 4), dtype=np.int)
        self.touchingRadii = np.zeros((1), dtype=np.int)
        self.touchingCenter = np.zeros((1, 3), dtype=np.float)
        self.box = box
        self.CoM = np.zeros(3)
        self.minxyz = np.zeros(3)
        self.maxxyz = np.zeros(3)
        self.auxiliarySimplices = []
        self.auxiliaryDistances = []

        self.triangulation()

    @staticmethod
    def _find_shared(simplices, incidents):
        incident_simplices = simplices[incidents]
        # we create a (X,3) array with all X triangles in simplices
        perm = incident_simplices[:, [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]]
        perm = perm.reshape(len(incident_simplices) * 4, 3)
        perm = np.sort(perm, axis=1)
        # now we select only those who are unique
        # it's a bit tricky, because the original algorithm
        # is not invariant under permutation of the elements of 'incidents'
        # we need to fish out the *unsorted* array of uniques.
        # TODO: understand the reason of this behavior.
        _, idx, counts = np.unique(
            perm, axis=0, return_index=True, return_counts=True)
        argsort = np.argsort(idx)
        backsort = idx[argsort]
        # we need now to extract the unsorted array...
        unique_triangles = perm[backsort]
        # ...and also the unsorted counts
        counts = counts[argsort]
        shared = np.zeros(len(unique_triangles), dtype=bool)
        cond = np.where(counts > 1)[0]
        shared[cond] = True
        # we return lists to keep consistency with the other functions
        # TODO: avoid switching back/forth where possible
        return list(unique_triangles), list(shared)

    def triangulation(self):
        # 1) Sorting atoms over distances and weights
        # 2) making of first tehtrahedron containing whole system
        # 3) incremental insertion of weightet points
        # 4) looking for incidenty -> added point intersect touching sphere of some tetrahedron in triangulation
        sortedListOdRadii = self.initialize()
        center = np.zeros(3)
        face = np.zeros(3)
        facetmp = np.zeros(3)
        shared = []
        incidentFaces = []
        buffersize = 0
        index = 0
        incidents = []
        used = False
        faceIndex = 0
        counter = 0
        cellIndex = 0
        indexx = 0
        indexy = 0
        indexz = 0
        maxdist = 0

        for pi in sortedListOdRadii:
            counter += 1
            incidentFaces = []
            shared = []
            index = pi[0]
            incidents = self.makeListOfIncidents(index)
            incidentFaces, shared = self._find_shared(self.simplices,
                                                      incidents)
            # TODO: should avoid manipulating by hand the output of Delaunay(), could lead to unexpected behaviour
            self.simplices = np.delete(self.simplices, incidents, 0)
            offset = 0
            dist = 0
            incidents.sort()
            self.touchingCenter = np.delete(self.touchingCenter, incidents, 0)
            self.touchingRadii = np.delete(self.touchingRadii, incidents, 0)
            # create new simplices -> connect hull faces with new point
            i = 0
            newSimplices = 0
            simplices_list, centers_list, radii_list = [], [], []
            for face in incidentFaces:
                if (shared[i]):
                    i += 1
                    continue
                simplices_list.append([face[0], face[1], face[2], index])
                cellIndex = len(self.simplices) - 1
                radius = tsphere(face[0], face[1], face[2], index, self.points,
                                 self.weights, center)
                centers_list.append(list(center))
                radii_list.append(radius)
                i += 1
                newSimplices += 1
            self.simplices = np.append(
                self.simplices, np.asarray(simplices_list, dtype=int), axis=0)
            self.touchingCenter = np.append(
                self.touchingCenter, np.asarray(centers_list), axis=0)
            self.touchingRadii = np.append(self.touchingRadii, radii_list)

        self.removeRedundantTetrahedrons()
        self.findNeighbors2()

    def findNeighbors2(self):
        neighborHash = {}
        indices = []
        counter = 0
        face = (-1, -1, -1)
        maxNeighbors = 0
        for i in range(0, len(self.simplices)):
            self.tmpNeighbors.append([])
            for i1 in range(0, 2):
                for i2 in range(i1 + 1, 3):
                    for i3 in range(i2 + 1, 4):
                        face = (self.simplices[i, i1], self.simplices[i, i2],
                                self.simplices[i, i3])
                        face = tuple(sorted(face))
                        if face not in neighborHash:
                            neighborHash[face] = []
                        neighborHash[face].append(i)

        self.neighbors = -2 * np.ones(
            8 * len(self.simplices), dtype=int).reshape(
                len(self.simplices), 8)
        for i in range(0, len(self.simplices)):
            del indices[:]
            for i1 in range(0, 2):
                for i2 in range(i1 + 1, 3):
                    for i3 in range(i2 + 1, 4):
                        face = (self.simplices[i, i1], self.simplices[i, i2],
                                self.simplices[i, i3])
                        face = tuple(sorted(face))
                        if (face in neighborHash):
                            for j in neighborHash[face]:
                                if (j not in self.tmpNeighbors[i] and j != i):
                                    self.tmpNeighbors[i].append(j)
                                    if (i3 == 2):  # get index of face
                                        if (3 in indices): indices.append(7)
                                        else: indices.append(3)
                                    else:
                                        if ((i3 - i2 - i1) in indices):
                                            indices.append(i3 - i2 - i1 + 4)
                                        else:
                                            indices.append(i3 - i2 - i1)
                            if (len(self.tmpNeighbors[i]) > maxNeighbors):
                                maxNeighbors = len(self.tmpNeighbors[i])

            self.neighbors[i][indices] = self.tmpNeighbors[i][:]
            if (-2 in self.neighbors[i][0:4]):
                self.neighbors[i][0:4][self.neighbors[i][0:4] == -2] = -1

    def areNeighbors(self, simplex1, simplex2):
        vertexCounter = 0
        if (simplex1[0] in simplex2): vertexCounter += 1
        if (simplex1[1] in simplex2): vertexCounter += 1
        if (simplex1[2] in simplex2): vertexCounter += 1
        if (simplex1[3] in simplex2): vertexCounter += 1
        if (vertexCounter == 3):
            return True
        return False

    def makeListOfIncidents(self, index):
        size = len(self.simplices)
        vect = self.points[index] - self.touchingCenter[:size]
        dist = np.linalg.norm(vect, axis=1)
        cond = dist - self.weights[index] < self.touchingRadii[:size]
        # the list of incidents
        return list(np.where(cond)[0])

    def initialize(self):
        listOfRadii = []
        self.CoM = np.mean(self.points, axis=0)
        self.minxyz = np.min(self.points, axis=0)
        self.maxxyz = np.max(self.points, axis=0)
        dR2 = np.sum((self.points - self.CoM)**2, axis=1)
        for i in range(0, len(self.weights)):
            listOfRadii.append(
                [i, self.weights[i],
                 sum((self.points[i] - self.CoM)**2)])
        listOfRadii.sort(key=lambda tup: tup[2])
        listOfRadii.sort(key=lambda tup: tup[1])

        self.generateFirstTetrahedron(self.CoM)

        return listOfRadii

    def generateFirstTetrahedron(self, CoM):
        # adding initial points
        tetraCoM = np.zeros(3)
        tetraPoints = np.zeros((4, 3), dtype=np.float)
        tetraPoints[0] = [1.0, 0, -1.0 / (2.0)**0.5]
        tetraPoints[1] = [-1.0, 0, -1.0 / (2.0)**0.5]
        tetraPoints[2] = [0, +1.0, 1.0 / (2.0)**0.5]
        tetraPoints[3] = [0, -1.0, 1.0 / (2.0)**0.5]

        tet = [t for t in tetraPoints]
        self.points = np.append(self.points, np.array(tet), axis=0)

        self.weights = np.append(self.weights, np.zeros(4))
        for i in range(4):
            self.simplices[0, i] = len(self.points) - i - 1

        isOutside = True
        while (isOutside):
            tetraCoM[:] = 0
            self.points[-4:, :] *= 1.05
            self.points[-4:, :] += CoM
            tetraCoM = np.mean(self.points[-4:, :], axis=0)
            s = self.simplices[0]
            self.touchingRadii[0] = tsphere(s[0], s[1], s[2], s[3],
                                            self.points, self.weights,
                                            self.touchingCenter[0])

            self.auxiliarySimplices.append(0)

            isOutside = False
            for i in range(0, len(self.points) - 5):
                dist = (sum((self.points[i] - tetraCoM)**2))**0.5
                if (dist - self.weights[i] > 0.3 * self.touchingRadii[0]):
                    isOutside = True
                    self.points[-4:, :] -= CoM
                    break

    def removeRedundantTetrahedrons(self):
        removable = []
        for i in range(0, len(self.simplices)):
            if (self.simplices[i, 0] >= len(self.points) - 4
                    or self.simplices[i, 1] >= len(self.points) - 4
                    or self.simplices[i, 2] >= len(self.points) - 4
                    or self.simplices[i, 3] >= len(self.points) - 4):
                removable.append(i)
        self.simplices = np.delete(self.simplices, removable, 0)
        self.touchingCenter = np.delete(self.touchingCenter, removable, 0)
        self.touchingRadii = np.delete(self.touchingRadii, removable, 0)

    def writeTriangulation(self):
        file_ = open("test.txt", 'wt')
        for i in range(0, len(self.points) - 4):
            file_.write(
                str(self.points[i, 0]) + ' ' + str(self.points[i, 1]) + ' ' +
                str(self.points[i, 2]) + '\n')

        file2 = open("test2.txt", 'wt')
        for i in range(0, len(self.simplices)):
            if (self.simplices[i, 0] < len(self.points) - 4
                    and self.simplices[i, 1] < len(self.points) - 4
                    and self.simplices[i, 2] < len(self.points) - 4
                    and self.simplices[i, 3] < len(self.points) - 4):
                file2.write(
                    str(self.points[self.simplices[i, 0], 0]) + ' ' +
                    str(self.points[self.simplices[i, 0], 1]) + ' ' +
                    str(self.points[self.simplices[i, 0], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 1], 0]) + ' ' +
                    str(self.points[self.simplices[i, 1], 1]) + ' ' +
                    str(self.points[self.simplices[i, 1], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 0], 0]) + ' ' +
                    str(self.points[self.simplices[i, 0], 1]) + ' ' +
                    str(self.points[self.simplices[i, 0], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 2], 0]) + ' ' +
                    str(self.points[self.simplices[i, 2], 1]) + ' ' +
                    str(self.points[self.simplices[i, 2], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 0], 0]) + ' ' +
                    str(self.points[self.simplices[i, 0], 1]) + ' ' +
                    str(self.points[self.simplices[i, 0], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 3], 0]) + ' ' +
                    str(self.points[self.simplices[i, 3], 1]) + ' ' +
                    str(self.points[self.simplices[i, 3], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 1], 0]) + ' ' +
                    str(self.points[self.simplices[i, 1], 1]) + ' ' +
                    str(self.points[self.simplices[i, 1], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 2], 0]) + ' ' +
                    str(self.points[self.simplices[i, 2], 1]) + ' ' +
                    str(self.points[self.simplices[i, 2], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 1], 0]) + ' ' +
                    str(self.points[self.simplices[i, 1], 1]) + ' ' +
                    str(self.points[self.simplices[i, 1], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 3], 0]) + ' ' +
                    str(self.points[self.simplices[i, 3], 1]) + ' ' +
                    str(self.points[self.simplices[i, 3], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 2], 0]) + ' ' +
                    str(self.points[self.simplices[i, 2], 1]) + ' ' +
                    str(self.points[self.simplices[i, 2], 2]) + '\n')
                file2.write(
                    str(self.points[self.simplices[i, 3], 0]) + ' ' +
                    str(self.points[self.simplices[i, 3], 1]) + ' ' +
                    str(self.points[self.simplices[i, 3], 2]) + '\n')
        file2.close
        file_.close
