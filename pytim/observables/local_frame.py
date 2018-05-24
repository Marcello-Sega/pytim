# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: geometry
    =========================
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
        
    """
    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)
        try:
            self.cutoff = kwarg['cutoff']
        except:
            self.cutoff = 4.0
        self.refgroup_warning=True

    def remove_pbc(self,p,box):
        cond = np.where(p-p[0] > box/2.)
        p[cond]-=box[cond[1]]
        cond = np.where(p-p[0] < -box/2.)
        p[cond]+=box[cond[1]]
        return p

    def compute(self, inp, kargs=None):
        """ Compute the local reference frame

            :param AtomGroup inp:  the group of atoms that define the surface
            :returns:              a (len(inp),3,3) set of coordinates defining
                                   the local axes, with the slice [:2:] defining
                                   the normal vectors
        """
        
        if not isinstance(inp,AtomGroup):
                raise RuntimeError("This observable requires an atomgroup")
        try:
            refgroup = inp.universe.trajectory.interface.cluster_group
        except:
            try:
                refgroup = kargs['refgroup']
            except:
                if self.refgroup_warning: 
                    print("Warning: without having a PYTIM interface or passing")
                    print("a refgroup, the surface normal will be meaningless")
                    self.refgroup_warning=False
        self.tree = cKDTree(inp.positions,boxsize=inp.universe.dimensions[:3])
        self.patches = self.tree.query_ball_tree(self.tree,self.cutoff)
        try:
            self.reftree = cKDTree(refgroup.positions,boxsize=inp.universe.dimensions[:3])
            self.refpatches = self.tree.query_ball_tree(self.reftree,self.cutoff)
        except:
            pass
        local_coords = []
        box = inp.universe.dimensions[:3]
        for i, patch in enumerate(self.patches): 
            p = self.tree.data[patch].copy()
            p = self.remove_pbc(p,box)
            cm = np.mean(p,axis=0)

            try:
                refp = self.reftree.data[self.refpatches[i]].copy()
                refp = self.remove_pbc(refp,box)
                # note that refp includes also surface atoms.
                # here we try to remove their contribution.
                if (len(refp)-len(p)):
                    refcm = (np.sum(refp,axis=0) - np.sum(p,axis=0))/(len(refp)-len(p))
                else:
                    refcm = np.mean(refp)
                dcm = cm - refcm
            except AttributeError:
                dcm = np.zeros(3)

            dr = (p-cm)
            # compute the 'inertia' tensor
            I = -np.tensordot(dr[:],dr[:],axes=(0,0)) + np.diag([np.sum(dr**2)/len(dr)]*3)
            U,S,V=np.linalg.svd(I)
            z=np.argmin(S)
            UU = np.roll(U,2-z,axis=-1).T
            # flip normal if cm on the same side of normal
            UU[2]*= ((np.dot(UU[2],dcm) <= 0) * 2. - 1.)
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

    """

    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)
        try:
            self.cutoff = kwarg['cutoff']
        except:
            self.cutoff = 4.0
        self.local_frame  = LocalReferenceFrame(cutoff=self.cutoff)


    def compute(self, inp, kargs=None):
        # TODO write a version that does not depend on passing an AtomGroup
        """ Compute the two curvatures

            :param AtomGroup inp:  the group of atoms that define the surface
            :returns:              a (len(inp),2) array with the Gaussian 
                                   and mean curvature for each of the atoms in
                                   the input group inp
        """
        if not isinstance(inp,AtomGroup):
                raise RuntimeError("This observable requires an atomgroup")
        local_basis = self.local_frame.compute(inp) 
        tree =  self.local_frame.tree
        patches = self.local_frame.patches
        box = inp.universe.dimensions[:3]
        GC,MC=[],[]
        for i,patch in enumerate(patches): 
            p = self.local_frame.tree.data[patch].copy()
            p = self.local_frame.remove_pbc(p,box)
            cm = np.mean(p,axis=0)
            dr = (p-cm)
            pp = np.dot(local_basis[i],dr.T).T
            x,y,z=pp[:,0],pp[:,1],pp[:,2]
            # fit to a quadratic form
            A = np.array([np.ones(x.shape[0]), x, y, x**2, y**2, x*y]).T
            coeff, r, rank, s = np.linalg.lstsq(A, z,rcond=None)
            C,S,T,P,Q,R  = coeff
            E,G,F = (1+S**2), (1+T**2), S*T
            L,M,N = 2*P,R,2*Q
            # Gaussian and mean curvature 
            GC.append( (L*N - M**2 )/ (E*G-F**2) )
            MC.append(  0.5 * (E*N-2*F*M+G*L)/(E*G-F**2))
        return np.array(zip(GC,MC))


