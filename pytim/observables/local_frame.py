# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: geometry
    ================
"""
from __future__ import print_function
import numpy as np
from scipy import stats
from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup
from scipy.spatial import cKDTree

from .observable import Observable


class LocalReferenceFrame(Observable):
    """ Compute the local surface using a definition based on that of
        S. O. Yesylevskyy and C. Ramsey, Phys.Chem.Chem.Phys., 2014, 16, 17052
        modified to include a guess for the surface normal direction


        Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import WATER_GRO
        >>> import numpy as np
        >>> g=mda.Universe(WATER_GRO).select_atoms('name OW')
        >>> inter = pytim.ITIM(g,molecular=False)
        >>> frame = pytim.observables.LocalReferenceFrame().compute(g)
        >>> np.set_printoptions(precision=6,threshold=2,suppress=None)
        >>> print(frame)
        [[[-0.330383  0.740913 -0.584718]
          [-0.937619 -0.328689  0.113291]
          [ 0.108252 -0.585672 -0.803287]]
        <BLANKLINE>
         [[-0.98614   0.164904  0.018286]
          [-0.146148 -0.811192 -0.566223]
          [ 0.078538  0.561048 -0.824049]]
        <BLANKLINE>
         [[-0.921816 -0.28798  -0.259465]
          [ 0.182384 -0.912875  0.365233]
          [ 0.342039 -0.289355 -0.894026]]
        <BLANKLINE>
         ...
        <BLANKLINE>
         [[-0.740137 -0.557695  0.375731]
          [ 0.369404  0.129692  0.920174]
          [-0.561906  0.819852  0.110024]]
        <BLANKLINE>
         [[-0.198256 -0.724093  0.660594]
          [ 0.64637  -0.603238 -0.467236]
          [-0.736818 -0.334356 -0.587627]]
        <BLANKLINE>
         [[ 0.268257 -0.337893 -0.902146]
          [ 0.233933 -0.885591  0.401253]
          [-0.934513 -0.31868  -0.158522]]]

        >>> np.set_printoptions()
    """

    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)
        try:
            self.cutoff = kwarg['cutoff']
        except:
            self.cutoff = 4.0
        try:
            self.refgroup_warning = kwarg['warning']
        except:
            self.refgroup_warning = True

    def remove_pbc(self, p, box):
        if box is None:
            return p
        cond = np.where(p - p[0] > box / 2.)
        p[cond] -= box[cond[1]]
        cond = np.where(p - p[0] < -box / 2.)
        p[cond] += box[cond[1]]
        return p

    def _compute_local_coords(self, box):
        local_coords = []
        for i, patch in enumerate(self.patches):
            p = self.tree.data[patch].copy()
            p = self.remove_pbc(p, box)
            cm = np.mean(p, axis=0)

            try:
                refp = self.reftree.data[self.refpatches[i]].copy()
                refp = self.remove_pbc(refp, box)
                # note that refp includes also surface atoms.
                # here we try to remove their contribution.
                if (len(refp) - len(p)):
                    refcm = (np.sum(refp, axis=0) - np.sum(p, axis=0)
                             ) / (len(refp) - len(p))
                else:
                    refcm = np.mean(refp)
                dcm = cm - refcm
            except AttributeError:
                dcm = np.zeros(3)

            dr = (p - cm)
            # compute the 'inertia' tensor
            I = -np.tensordot(dr[:], dr[:], axes=(0, 0)) + \
                np.diag([np.sum(dr**2) / len(dr)] * 3)
            U, S, V = np.linalg.svd(I)
            z = np.argmin(S)
            UU = np.roll(U, 2 - z, axis=-1).T
            # flip normal if cm on the same side of normal
            UU[2] *= ((np.dot(UU[2], dcm) <= 0) * 2. - 1.)
            local_coords.append(UU.T)
        return np.array(local_coords)

    def compute(self, inp, kargs=None):
        """ Compute the local reference frame

            :param AtomGroup inp:  the group of atoms that define the surface
            :returns:              a (len(inp),3,3) set of coordinates defining
                                   the local axes, with the slice [:2:] defining
                                   the normal vectors
        """

        if isinstance(inp, AtomGroup):
            box = inp.universe.dimensions[:3]
            pos = inp.positions
        if isinstance(inp, np.ndarray):
            box = None
            pos = inp
        try:
            refgroup = inp.universe.trajectory.interface.cluster_group
        except:
            try:
                refgroup = kargs['refgroup']
            except:
                if self.refgroup_warning:
                    print("Warning: without having a PYTIM interface or passing")
                    print("a refgroup, the surface normal will be meaningless")
                    self.refgroup_warning = False
        self.tree = cKDTree(pos, boxsize=box)
        self.patches = self.tree.query_ball_tree(self.tree, self.cutoff)
        try:
            self.reftree = cKDTree(
                refgroup.positions, boxsize=box)
            self.refpatches = self.tree.query_ball_tree(
                self.reftree, self.cutoff)
        except:
            pass

        return self._compute_local_coords(box)


class Curvature(Observable):
    """ Gaussian and mean curvature

        Compute the two curvatures in the local reference frame
        defined using the particle distribution within a given
        cutoff. Note that the mean curvature is not necessarily
        meaningful, as the local normal is not well defined. The
        Gaussian curvature, instead, as an intrinsic property,
        is well defined.

        :param float cutoff:  use particles within this range
                              to determine the local environment


        Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import WATER_GRO
        >>> import numpy as np
        >>> g=mda.Universe(WATER_GRO).select_atoms('name OW')
        >>> inter = pytim.ITIM(g,molecular=False)
        >>> curv = pytim.observables.Curvature().compute(g)
        >>> print(np.array_str((curv[:10,0]),precision=2,suppress_small=True))
        [-0.04 -0.17 -0.44 -0.15 -0.31  0.06  0.08 -0.22  0.08 -0.27]


    """

    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)
        try:
            self.cutoff = kwarg['cutoff']
        except:
            self.cutoff = 4.0
        try:
            self.refgroup_warning = kwarg['warning']
        except:
            self.refgroup_warning = True
        self.local_frame = LocalReferenceFrame(
            cutoff=self.cutoff, warning=self.refgroup_warning)

    def compute(self, inp, kargs=None):
        # TODO write a version that does not depend on passing an AtomGroup
        """ Compute the two curvatures

            :param AtomGroup inp:  the group of atoms that define the surface
            :returns:              a (len(inp),2) array with the Gaussian
                                   and mean curvature for each of the atoms in
                                   the input group inp
        """
        if isinstance(inp, AtomGroup):
            box = inp.universe.dimensions[:3]
        if isinstance(inp, np.ndarray):
            box = None
        local_basis = self.local_frame.compute(inp)
        tree = self.local_frame.tree
        patches = self.local_frame.patches
        GC, MC = [], []
        for i, patch in enumerate(patches):
            p = self.local_frame.tree.data[patch].copy()
            p = self.local_frame.remove_pbc(p, box)
            cm = np.mean(p, axis=0)
            dr = (p - cm)
            pp = np.dot(local_basis[i], dr.T).T
            x, y, z = pp[:, 0], pp[:, 1], pp[:, 2]
            # fit to a quadratic form
            A = np.array([np.ones(x.shape[0]), x, y, x**2, y**2, x * y]).T
            coeff, r, rank, s = np.linalg.lstsq(A, z, rcond=None)
            C, S, T, P, Q, R = coeff
            E, G, F = (1 + S**2), (1 + T**2), S * T
            L, M, N = 2 * P, R, 2 * Q
            # Gaussian and mean curvature
            GC.append((L * N - M**2) / (E * G - F**2))
            MC.append(0.5 * (E * N - 2 * F * M + G * L) / (E * G - F**2))
        return np.asarray(list(zip(GC, MC)))

    def _():
        """ additional tests

        here we generate a paraboloid (x^2+y^2) and a hyperbolic paraboloid
        (x^2-y^2) to check that the curvature code gives the right answers for
        the Gaussian (4, -4) and mean (2, 0) curvatures

        >>> import pytim
        >>> x,y=np.mgrid[-5:5,-5:5.]/2.
        >>> p = np.asarray(list(zip(x.flatten(),y.flatten())))
        >>> z1 = p[:,0]**2+p[:,1]**2
        >>> z2 = p[:,0]**2-p[:,1]**2
        >>>
        >>> for z in [z1, z2]:
        ...     pp = np.asarray(list(zip(x.flatten()+5,y.flatten()+5,z)))
        ...     curv = pytim.observables.Curvature(cutoff=1.,warning=False).compute(pp)
        ...     val =  (curv[np.logical_and(p[:,0]==0,p[:,1]==0)])
        ...     # add and subtract 1e3 to be sure to have -0 -> 0
        ...     print(np.array_str((val+1e3)-1e3, precision=2, suppress_small=True))
        [[4. 2.]]
        [[-4.  0.]]


        """
#
