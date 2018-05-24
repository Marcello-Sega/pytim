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
        >>> np.set_printoptions(precision=None,threshold=None,suppress=None)
        >>> print(frame)
        [[[-0.33038264  0.74091275 -0.5847184 ]
          [-0.93761874 -0.32868871  0.11329094]
          [ 0.10825164 -0.58567229 -0.80328672]]
        <BLANKLINE>
         [[-0.98614017  0.16490351  0.01828644]
          [-0.14614815 -0.81119166 -0.56622329]
          [ 0.07853841  0.56104806 -0.82404902]]
        <BLANKLINE>
         [[-0.92181645 -0.28798001 -0.25946474]
          [ 0.18238429 -0.91287521  0.36523256]
          [ 0.3420386  -0.28935509 -0.89402641]]
        <BLANKLINE>
         ...
        <BLANKLINE>
         [[-0.7401371  -0.55769455  0.3757311 ]
          [ 0.36940366  0.12969243  0.92017434]
          [-0.56190569  0.81985161  0.11002423]]
        <BLANKLINE>
         [[-0.19825565 -0.72409255  0.66059419]
          [ 0.64636999 -0.60323796 -0.46723634]
          [-0.73681785 -0.33435601 -0.58762702]]
        <BLANKLINE>
         [[ 0.2682568  -0.33789274 -0.90214566]
          [ 0.23393259 -0.88559102  0.40125315]
          [-0.93451262 -0.31868015 -0.15852169]]]

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
        return np.array(zip(GC, MC))

    def _():
        """ additional tests

        here we generate a paraboloid (x^2+y^2) and a hyperbolic paraboloid 
        (x^2-y^2) to check that the curvature code gives the right answers for 
        the Gaussian (4, -4) and mean (2, 0) curvatures 

        >>> import pytim
        >>> x,y=np.mgrid[-5:5,-5:5.]/2.
        >>> p = np.array(zip(x.flatten(),y.flatten()))
        >>> z1 = p[:,0]**2+p[:,1]**2
        >>> z2 = p[:,0]**2-p[:,1]**2
        >>> 
        >>> for z in [z1, z2]:
        ...     pp = np.array(zip(x.flatten()+5,y.flatten()+5,z))
        ...     curv = pytim.observables.Curvature(cutoff=1.,warning=False).compute(pp)
        ...     val =  (curv[np.logical_and(p[:,0]==0,p[:,1]==0)])
        ...     # add and subtract 1e3 to be sure to have -0 -> 0
        ...     print(np.array_str((val+1e3)-1e3, precision=2, suppress_small=True))
        [[4. 2.]]
        [[-4.  0.]]


        """
#
