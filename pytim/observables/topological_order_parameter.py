# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: topological_order_parameter
    ===================================

    Implements the topological order parameter Psi as introduced in
    "Correlated Fluctuations of Structural Indicators Close to the 
     Liquid−Liquid Transition in Supercooled Water", 
     R. Foffi and F. Sciortino, J. Phys. Chem. B 2023, 127, 378−386
     DOI: 10.1021/acs.jpcb.2c07169

"""
from __future__ import print_function
import numpy as np
from scipy import stats, sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree
from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup
from MDAnalysis.lib import distances
from ..utilities_geometry import minimum_image
from .observable import Observable


class TopologicalOrderParameter(Observable):
    r""" Topological order parameter. This observable provides also several
        helper functions and stores additional information that is useful
        for analyzing the topological properties of a set of molecules.

    """

    def __init__(self, *arg, **kwarg):
        Observable.__init__(self, None)
        self.acceptor_type, self.topological_distance = arg[0],arg[1]
        self.M = None

    def compute(self, inp, kargs=None):
        r""" 
            Compute the order parameters and Euclidean distances
            for atoms in group separated by the chosen topologigal
            distance

            :param AtomGroup inp:  an AtomGroup of donors (e.g. O)
            The group needs to be homogeneous (same molecule type)
            or at least have the same number of acceptors of type
            acceptor_type

            Returns:

            psi : an ndarray containing the order parameters
            of each atom in group and an array with the indices
            of nearest neighbors at the chosen topological
            distance.

            Note that the observable stores also other information
            in the class:

            - neighbors: the index of the nearest neighbor at topological
            distance N. The distances to these neighbors are the values
            of psi.

            - dists: the Euclidean distance to all neighbors at the
            chosen topological distance N
            
            - predecessors: the matrix of predecessors. This can be
            used to obtain the sequence of molecules at topological
            distance 1,2, ..., N.
            
            - pairs: the lists of pairs at topological distance N
            
            - dist: the topological distances

            Example:

            >>> import numpy as np
            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.observables import TopologicalOrderParameter
            >>> from pytim.datafiles import WATER_GRO
            >>>
            >>> u = mda.Universe(WATER_GRO)
            >>> pos = u.atoms.positions[:]
            >>> # we create a molecule in the vapor phase,
            >>> # its value of the order parameter must be inf
            >>> pos[u.atoms.resindices==0]+=75
            >>> u.atoms.positions=pos
            >>> g = u.select_atoms('name OW')
            >>> psi = TopologicalOrderParameter('H',4)
            >>> print(psi.compute(g)[:4])
            [       inf 3.27362227 3.91287661 3.57974577]

        """

        # we do a soft check using the residue name
        assert np.all(inp.residues.resnames==inp.residues.resnames[0]), 'inp should be homogeneous' 
        nacceptors = inp.residues[0].atoms[inp.residues[0].atoms.types == self.acceptor_type].n_atoms
        box = inp.universe.dimensions[:3]
        size = (len(inp),len(inp))
        M = sparse.lil_matrix(size)*0
        pos = inp.pack_into_box(inplace=True)
        box = inp.dimensions[:3]
        tree = KDTree(inp.positions,boxsize=box)
        OO = tree.query_ball_tree(tree, 3.5)
        self.inp = inp
        for i,neighs in enumerate(OO):
            neigh_Os = inp[[a for a in neighs if a != i ]]
            sel = inp[[a for a in neighs if a != i ]].residues.atoms
            disp = minimum_image(sel[sel.types==self.acceptor_type].positions - inp[i].position, box)
            # OH_d: distances correpsonding to disp
            OH_d = np.linalg.norm(disp,axis=1)
            # OH_d: we reshape to (N,nacceptors) to extract the distance to the closest acceptor
            OH_d = np.min(OH_d.reshape((OH_d.shape[0]//nacceptors,nacceptors)),axis=1)
            # HB criterion on OH distance

            cond = np.argwhere(OH_d<2.45).flatten().tolist()
            # Update the list of HB pairs: use the residue index
            # note: cond runs over the elements of sel!
            for a in  np.unique(neigh_Os[cond].resindices).tolist():  M[i,a], M[a,i] = 1,1
        self.M = M
        res = dijkstra(M,
                     unweighted=True,
                     limit=self.topological_distance,
                     directed=True,
                     return_predecessors=True)
        self.dist,self.predecessors  = res
        
        pairs = np.argwhere(self.dist==self.topological_distance)
        # we use directed=True because we computed the matrix symmetrically
        dists = np.linalg.norm(minimum_image(inp[pairs[:,0]].positions - inp[pairs[:,1]].positions,
                                         inp.dimensions),axis=1)
        self.dists =dists
        # in some case (gas) a molecule might have no 4th neighbor, so we initialize
        # psi with inf.
        neighbors,psi=-np.ones(len(inp),dtype=int),np.inf*np.ones(len(inp),dtype=float)

        index = np.arange(len(inp))

        for i in np.arange(len(inp)):
            cond = np.where(pairs[:,0]==i)[0]
            if len(cond) > 0:
                amin =  np.argmin(dists[cond])
                neighbors[i],psi[i] =  pairs[cond,1][amin], np.min(dists[cond])
            # otherwise, there is no 4th neighbor, we keep the default value (inf)
        self.psi = psi
        # so far neighbors and pairs are indices of the local array, now we store
        # them as MDAnalysis indices
        self.neighbors = self.inp.indices[neighbors]
        self.pairs = self.inp.indices[pairs]
        return self.psi

    def path(self, start, end):
        r""" 
          Given the start and end of a path, returns the 
          molecules within the connected path, if it exists.
          Accepts indices (zero-based) of the input group
          used to compute the observable, or two Atom types, in
          which case it returns an AtomGroup.

          :param  int|Atom  :  start, the first atom in the chain
          :param  int|Atom  :  end, the last atom in the chain
        """ 
        if isinstance(start,AtomGroup) and isinstance(end,AtomGroup): ret_type=AtomGroup
        
        if ret_type == AtomGroup:
            start = np.where(inp.indices == start.index)[0][0]
            end   = np.where(inp.indices == end.index)[0][0]

        p = [end] 
        try:
            while  p[-1] != start:
                p.append(self.predecessors[start,p[-1]] )
        except:
            return []

        if ret_type == AtomGroup: return self.inp[p]
        return p

    def path_to_nearest_topological_neighbor(self,start):
        r"""
            Given an starting atom belonging to the input group
            of the observable, return the set of connected neighbors
            leading to the nearest neighbor at the chosen
            topological distance.

         """
        if isinstance(start, Atom): 
            start = np.argwhere(self.inp.indices ==atom.index)[0,0]
        end = self.neighbor[start]
        path = self.path(start,end)
        if isinstance(start, Atom): return self.inp[path]
        else: return path


