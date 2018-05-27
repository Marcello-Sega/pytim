#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: sasa
    ============
"""
from __future__ import print_function
from multiprocessing import Process, Queue, cpu_count
import numpy as np
from scipy.spatial import distance

from . import utilities
from .sanity_check import SanityCheck
from .surface import SurfaceFlatInterface
from .surface import SurfaceGenericInterface

from .interface import Interface
from .patches import PatchTrajectory, PatchOpenMM, PatchMDTRAJ
from circumradius import circumradius
from .gitim import GITIM
from scipy.spatial import cKDTree


class SASA(GITIM):
    """ Identifies interfacial molecules at curved interfaces using the
        Lee-Richards SASA algorithm (Lee, B; Richards, FM. J Mol Biol.
        55, 379–400, 1971)

        :param Object universe:   The MDAnalysis_ Universe, MDTraj_ trajectory
                                  or OpenMM_ Simulation objects.
        :param Object group:        An AtomGroup, or an array-like object with
                                    the indices of the atoms in the group. Will
                                    identify the interfacial molecules from
                                    this group
        :param float alpha:         The probe sphere radius
        :param str normal:          'x','y,'z' or 'guess'
                                    (for planar interfaces only)
        :param bool molecular:      Switches between search of interfacial
                                    molecules / atoms (default: True)
        :param int max_layers:      The number of layers to be identified
        :param dict radii_dict:     Dictionary with the atomic radii of the
                                    elements in the group. If None is supplied,
                                    the default one (from GROMOS 43a1) will be
                                    used.
        :param float cluster_cut:   Cutoff used for neighbors or density-based
                                    cluster search (default: None disables the
                                    cluster analysis)
        :param float cluster_threshold_density: Number density threshold for
                                    the density-based cluster search. 'auto'
                                    determines the threshold automatically.
                                    Default: None uses simple neighbors cluster
                                    search, if cluster_cut is not None
        :param Object extra_cluster_groups: Additional groups, to allow for
                                    mixed interfaces
        :param bool biggest_cluster_only: Tag as surface atoms/molecules only
                                    those in the largest cluster. Need to
                                    specify also a :py:obj:`cluster_cut` value.
        :param bool centered:       Center the  :py:obj:`group`
        :param bool info:           Print additional info
        :param bool warnings:       Print warnings
        :param bool autoassign:     If true (default) detect the interface
                                    every time a new frame is selected.

        Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import MICELLE_PDB
        >>>
        >>> u = mda.Universe(MICELLE_PDB)
        >>> micelle = u.select_atoms('resname DPC')
        >>> inter = pytim.SASA(u, group=micelle, molecular=False)
        >>> inter.atoms
        <AtomGroup with 619 atoms>


    """
    nslices = 10
    nangles = 100

    # no __init__ function, we borrow it from GITIM
    def _sanity_checks(self):
        """ Basic checks to be performed after the initialization.

        """

    def alpha_shape(self, alpha, group, layer):
        raise AttributeError('alpha_shape does not work in SASA ')

    def _overlap(self, Ri, Rj, pij, dzi):
        dzj = pij[:, 2] - dzi
        ri2 = Ri**2 - dzi**2
        rj2 = Rj**2 - dzj**2

        cond = np.where(rj2 >= 0.0)[0]
        if len(cond) == 0:
            return [], []
        rj2 = rj2[cond]
        pij = pij[cond]

        ri, rj = np.sqrt(ri2), np.sqrt(rj2)
        pij2 = pij**2
        dij2 = pij2[:, 0] + pij2[:, 1]
        dij = np.sqrt(dij2)

        # slice within neighboring one
        if np.any(dij <= rj - ri):
            return np.array([2. * np.pi]), np.array([0.0])

        # c1 => no overlap; c2 => neighboring slice enclosed
        c1, c2 = dij < ri + rj, dij > ri - rj
        cond = np.where(np.logical_and(c1, c2))[0]
        if len(cond) == 0:
            return [], []

        arg = (ri2 + dij2 - rj2) / (ri * dij * 2.)
        alpha = 2. * np.arccos(arg[cond])
        argx, argy = pij[:, 0] / dij, pij[:, 1] / dij
        beta = np.arctan2(argx[cond], argy[cond])
        return alpha, beta

    def _atom_coverage(self, index):
        group = self.sasa_group
        box = group.universe.dimensions[:3]
        R = group.radii[index]
        cutoff = R + self.Rmax + 2. * self.alpha
        neighbors = self.tree.query_ball_point(group.positions[index], cutoff)
        neighbors = np.asarray(list(set(neighbors) - set([index])))
        covered_slices, exposed_area = 0, 4. * np.pi * R**2
        buried = False
        delta = R + self.alpha - 1e-3
        slices = np.arange(-delta, delta, 2. * delta / self.nslices)
        Ri, Rj = group.radii[index], group.radii[neighbors]
        Ri += self.alpha
        Rj += self.alpha
        pi, pj = group.positions[index], group.positions[neighbors]
        pij = pj - pi
        cond = np.where(pij > box / 2.)
        pij[cond] -= box[cond[1]]
        cond = np.where(pij < -box / 2.)
        pij[cond] += box[cond[1]]
        for dzi in slices:
            arc = np.zeros(self.nangles)
            alpha, beta = self._overlap(Ri, Rj, pij, dzi)
            if len(alpha) > 0:
                N = np.asarray(list(zip(beta - alpha / 2., beta + alpha / 2.)))
                N = np.asarray(N * self.nangles / 2 / np.pi, dtype=int)
                N = N % self.nangles
                for n in N:
                    if n[1] > n[0]:  # ! not >=
                        arc[n[0]:n[1]] = 1.0
                    else:
                        arc[n[0]:] = 1.0
                        arc[:n[1]] = 1.0

                if np.sum(arc) == len(arc):
                    covered_slices += 1
                A = (np.sum(arc) / self.nangles) * 2. * np.pi
                exposed_area -= 2. * R**2 / self.nslices * A
        if covered_slices >= len(slices):
            buried = True
        return buried, exposed_area

    def _atomlist_coverage(self, indices, queue=None):
        res = [], []
        for index in indices:
            b, a = self._atom_coverage(index)
            res[0].append(b), res[1].append(a)
        if queue is None:
            return res
        else:
            queue.put(res)

    def compute_sasa(self, group):
        box = group.universe.dimensions[:3]
        self.Rmax = np.max(group.radii)
        self.tree = cKDTree(group.positions, boxsize=box)
        self.sasa_group = group
        try:
            self.ncpu
        except:
            self.ncpu = cpu_count()

        indices = range(len(group.atoms))
        exposed = [False] * len(group.atoms)
        area = [0.0] * len(group.atoms)
        queue, proc = [], []

        for c in range(self.ncpu):
            queue.append(Queue())
            proc.append(Process(
                target=self._atomlist_coverage,
                args=(indices[c::self.ncpu], queue[c])))
            proc[c].start()

        for c in range(self.ncpu):
            sl = slice(c, len(indices), self.ncpu)
            res = queue[c].get()
            # in some cases zero atoms are assinged to some of the processes
            if len(res[0]) > 0:
                exposed[sl] = (~np.asarray(res[0]))
                area[sl] = res[1]

        self.area = np.array(area)
        return np.where(exposed)[0]

    def _assign_layers(self):
        """Determine the SASA layers."""

        alpha_group, dbs = self._assign_layers_setup()

        for layer in range(0, self.max_layers):

            alpha_ids = self.compute_sasa(alpha_group)

            group = alpha_group[alpha_ids]

            alpha_group = self._assign_layers_postprocess(
                dbs, group, alpha_group, layer)

        # reset the interpolator
        self._interpolator = None

    @property
    def layers(self):
        """Access the layers as numpy arrays of AtomGroups.

        The object can be sliced as usual with numpy arrays.
        Differently from :mod:`~pytim.itim.ITIM`, there are no sides. 

        """
        return self._layers

    def _():
        """ additional tests


        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import WATER_GRO
        >>> u = mda.Universe(WATER_GRO)
        >>> g=u.atoms[0:2]
        >>> g.positions=np.array([[0.,0.,0.],[0.0,0.,0.]])
        >>> inter = pytim.SASA(u,group=g,molecular=False)
        >>> g.radii=np.array([1.0,0.5])
        >>> inter = pytim.SASA(u,group=g,molecular=False)
        >>> print (np.all(np.isclose(inter.area,[4.*np.pi,0.0])))
        True


        """


#
