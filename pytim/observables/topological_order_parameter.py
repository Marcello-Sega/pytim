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
from abc import ABCMeta, abstractmethod
import numpy as np
import warnings
from scipy import stats, sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree
from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup
from MDAnalysis.lib import distances
from ..utilities_geometry import minimum_image
from .observable import Observable

class TopologicalOrderParameter(Observable):
    r"""Bond Topological order parameter. This observable provides
        also several helper functions and stores additional information that
        is useful for analyzing the topological properties of a set of molecules.
    """

    def __init__(self, bonding_criterion, topological_distance=4):
        r"""    
                :param BondingCriterion bonding_criterion: instance of a class to determine whether
                    atoms in two groups are bonded or not.
                :param int topological_distance: which topological distance to consider when
                    computing the order parameter

        """

        Observable.__init__(self, None)
        self.bonding_criterion = bonding_criterion
        self.topological_distance = topological_distance
        self.M = None

    def compute(self, inp ,kargs=None):
        r"""
            Compute the order parameters and Euclidean distances
            for atoms in group separated by the chosen topologigal
            distance

            :param AtomGroup inp: an AtomGroup of including donors, 
                acceptors and hydrogen 

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
            >>> from pytim.observables import TopologicalOrderParameter,WaterHydrogenBondingAdjacencyMatrix
            >>> from pytim.datafiles import WATER_GRO
            >>>
            >>> u = mda.Universe(WATER_GRO)
            >>> pos = u.atoms.positions[:]
            >>> # First, we need to define the criterion
            >>> # used to define the bonding
            >>> HB = WaterHydrogenBondingAdjacencyMatrix()
            >>> # we create a molecule in the vapor phase,
            >>> # its value of the order parameter must be inf
            >>> pos[u.atoms.resindices==0]+=75
            >>> u.atoms.positions=pos
            >>> g = u.select_atoms('name OW')
            >>> psi = TopologicalOrderParameter(HB,4)
            >>> with np.printoptions(precision=4):
            ...     print(psi.compute(g)[:4])
            [   inf 3.2736 3.9129 3.5797]

            >>> # Now we compute by hand the path uniting two 4-bonded topological neighbors...
            >>> at1 = g.atoms[42]
            >>> at4 = u.atoms[psi.neighbors[at1.resindex]]
            >>> path = psi.path(at1,at4)
            >>> path.indices
            array([3522, 8964, 4515, 7725,  126])

            >>> # ...and check that this is the same as what's reported by
            >>> # psi.path_to_nearest_topological_neighbor()
            >>> psi.path_to_nearest_topological_neighbor(at1).indices
            array([3522, 8964, 4515, 7725,  126])

            >>> all(path.indices == psi.path_to_nearest_topological_neighbor(at1).indices)
            True

        """
        self.group = inp
        M = self.bonding_criterion.compute(inp)
        # we might use directed=True to speedup because we computed the matrix symmetrically
        res = dijkstra(M,
                     unweighted=True,
                     limit=self.topological_distance,
                     directed=True,
                     return_predecessors=True)
        self.dist,self.predecessors  = res

        pairs = np.argwhere(self.dist==self.topological_distance)
        g = self.bonding_criterion.base_group
        dists = np.linalg.norm(minimum_image(g[pairs[:,0]].positions - g[pairs[:,1]].positions,
                                         self.bonding_criterion.group.universe.dimensions[:3]),axis=1)
        self.dists = dists
        # in some case (gas) a molecule might have no 4th neighbor, so we initialize
        # psi with inf.
        neighbors,psi=-np.ones(len(g),dtype=int),np.inf*np.ones(len(g),dtype=float)

        for i in np.arange(len(g)):
            # pairs has symmetric entries (eg if [1,2] is present, also [2,1] will)
            # because of how M has been built. These are the pairs that are connected
            # by a given topological_distance. Now we take the first column, of pairs
            # and we are sure, because of symmetry, that these are all the indices of
            # molecules that have a linked neighbour. We check for each of the donors
            # and acceptors if they are in this list.
            cond = np.where(pairs[:,0]==i)[0]
            if len(cond) > 0: # cond gives the list of donors or acceptors connected to
                              # the i-th molecule of the donor+acceptor group.
                # select the specific pair with minimum distance
                amin =  np.argmin(dists[cond])
                # the neighbor is the second column in the pair where the distance is
                # minimum, and the associated value of psi is the minimum distance
                # itself.
                neighbors[i],psi[i] =  pairs[cond,1][amin], np.min(dists[cond])
            # otherwise, there is no 4th neighbor, we keep the default value (inf)
        self.psi = psi
        # so far neighbors and pairs are indices of the local array, now we store
        # them as MDAnalysis indices. To be able to search for paths, we store also
        # the indices of the local arrau
        self.neighbors = g.indices[neighbors]
        self._neighbors_localindex = neighbors
        self.pairs = self.bonding_criterion.base_group.indices[pairs]
        return self.psi

    def path(self, start, end):
        r"""
            Given the start and end of a path, returns the
            molecules within the connected path, if it exists.
            Accepts indices (zero-based) of the input group
            used to compute the observable, or two Atom objects, 
            in which case it identifies the molecules to which they
            belong and returns an AtomGroup.

            :param  int|Atom  :  start, the first atom in the chain
            :param  int|Atom  :  end, the last atom in the chain
        """
        try: self.psi
        except (AttributeError): warnings.warn("The order parameter has not yet been computed",category=UserWarning)
        ret_type = list
        if isinstance(start,Atom) and isinstance(end,Atom):
            ret_type=AtomGroup
            if not start in self.group: raise ValueError("the start Atom does not belong to the input group")
            if not end in self.group: raise ValueError("the end Atom does not belong to input group")

        if ret_type == AtomGroup:
            start_id, end_id = start.resindex, end.resindex
        else:
            start_id, end_id = start, end
        p = [end_id]
        try:
            while  p[-1] != start_id:
                p.append(self.predecessors[start_id,p[-1]] )
        except:
            return []
        p = np.array(p).tolist()
        if ret_type == AtomGroup: return self.bonding_criterion.base_group[p]
        else: return p

    def path_to_nearest_topological_neighbor(self,start):
        r"""
            Given a starting atom belonging to the input group
            of the observable, return the set of connected neighbors
            leading to the nearest neighbor at the chosen
            topological distance.

            :param  int|Atom  :  start, the first atom in the chain.
                                 If an Atom is passed, this must be part
                                 of either the acceptor or donor groups.
                                 If an index is passed, this is interpreted
                                 as the residue index of one of the acceptor
                                 or donor group members.

         """
        try: self.psi
        except (AttributeError): warnings.warn("The order parameter has not yet been computed",category=UserWarning)
        if isinstance(start, Atom):
            if not start in self.bonding_criterion.base_group: raise ValueError("the start Atom does not belong to either of the acceptor or donor groups")
            start_id = start.resindex
        else:
            start_id = start
        end_id = self._neighbors_localindex[start_id]
        path = self.path(start_id,end_id)
        if isinstance(start, Atom): return self.bonding_criterion.base_group[path]
        else: return path


