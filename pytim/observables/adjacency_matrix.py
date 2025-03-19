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


__metaclass__ = ABCMeta

class AdjacencyMatrix(Observable):
    """
    Base class for defining different bonding criteria.
    """
    @property
    @abstractmethod
    def base_group(self):
        # TODO: comment
        pass

class DistanceBasedAdjacencyMatrix(AdjacencyMatrix):
    """
    Simple distance-based bonding criterion working on a per-atom basis.
    """
    def __init__(self, cutoff_distance=3.5, pbc=True):
        self.cutoff_distance = cutoff_distance
        self.pbc = pbc    
        self.group = None

    def compute(self, group):
        """
        Return an adjacency matrix 

        :param group: AtomGroup to compute the adjacency matrix of.

        :return: a (len(group),len(group)) ndarray
        """
        if self.pbc: box = group.dimensions[:3]
        else: box = None 
        self.group = group
        self.size = (len(group),len(group))
        M = sparse.lil_matrix(self.size)*0
        pos = group.pack_into_box(inplace=True)
        kdtree = KDTree(pos,boxsize=box)
        neighs_list = kdtree.query_ball_tree(kdtree, self.cutoff_distance)
        for i,neighs in enumerate(neighs_list): M[i,neighs] = 1
        self.M = M
        return M

    @property
    def base_group(self): return self.group
        

class WaterHydrogenBondingAdjacencyMatrix(AdjacencyMatrix):
    """
    Hydrogen bond criterion for water. 
    :param str method: one of
        - 'OO'     : Based on O...O distance.
        - 'OO_OH'  : Based on O...O and O...H distance.
        - 'OHO'    : Based on O...H...O angle.

    :param float OO: the donor-acceptor cutoff distance.
        Only donor-acceptor pairs within this cutoff are considered
        hydrogen-bonded. Default: 3.5 Angstrom. See also the other
        requirement, cut_hydrogen_acceptor. Note that requiring two distance
        cutoffs is practically equivalent to requiring a distance and an angle
        cutoff, if chosen appropriately.

    :param float OH: the maximum distance between a hydrogen atom and its 
        acceptor to be considered hydrogen bonded. Only used when the chosen
        method is 'OO_OH'. Default: 2.5 Angstrom


    :param float OHO: the maximum angle for two molecules to be considered
        h-bonded. Only used when the chosen method is 'OHO'. Default: 30 deg.
    """

    def __init__(self, method="OO_OH", OO=3.5, OH=2.5, OHO=30.0):
        self.method = method
        self.OO = OO  # O...O distance cutoff in Å
        self.OH = OH  # O...H distance cutoff in Å
        self.OHO = OHO  # O...H...O angle cutoff in degrees
        self.oxygens = None

    def compute(self, group):
        """
        Return an adjacency matrix.

        :param group: AtomGroup to compute the adjacency matrix of.

        :return: a (nmol, nmol) ndarray with nmol the number of water
            molecules in group.
        """
        # we do a soft check using the residue name
        assert np.all(group.residues.resnames==group.residues.resnames[0]), 'the group should be homogeneous'
        _ = group.pack_into_box(inplace=True)
        # number of hydrogens in each donor molecule
        self.box = group.dimensions[:3]
        self.size = (len(group.residues),len(group.residues))
        self.M = sparse.lil_matrix(self.size)*0
        self.group = group
        self.oxygens = group.atoms[group.atoms.types == 'O']
        self.hydrogens = group.atoms[group.atoms.types == 'H']
        if self.method == 'OO_OH': return self._adjacency_matrix_OO_OH()
        else: raise NotImplemented("methods OO and OHO not implemented")

    def _adjacency_matrix_OO_OH(self):
        tree = KDTree(self.oxygens.positions,boxsize=self.box)
        OOlist = tree.query_ball_tree(tree, self.OO)
        # the observable is based on residues, so the max size of the matrix will never be larger than
        # the number of residues. We build a lookup table to simplify handling them: to obtain the sequential
        # index within the oxygen group, pass its resindex to the table.
        # resindices is zero-based, so let's add one. Any residue not in the lookuptable will return -1
        lookup = -1 + np.zeros(1+np.max(self.oxygens.resindices),dtype=int)
        lookup[self.oxygens.resindices] = np.arange(len(self.oxygens))
        for i,neighs in enumerate(OOlist): # for each of the acceptors, the list of neighboring donors
            neigh_As = self.oxygens[[a for a in neighs if a != i ]] # group of donors neighbor to the i-th acceptor, exclude self.
            sel = neigh_As.residues.atoms # all the atoms in the molecules of the  neighboring donors of the i-th acceptor
            # displacement between i-th acceptor and all hydrogens in the neighboring donor molecules
            disp = minimum_image(sel[sel.types=='H'].positions - self.oxygens[i].position,self.box)
            #correpsonding Acceptor-Hydrogen (AH) distances
            AH_d = np.linalg.norm(disp,axis=1)
            # AH_d: we reshape to extract the closest of the 2 Hs to the acceptor
            AH_d = np.min(AH_d.reshape((AH_d.shape[0]//2,2)),axis=1)
            # HB criterion on AH distance
            cond = np.argwhere(AH_d<self.OH).flatten().tolist()
            # Update the list of HB pairs: we use the lookup table
            b = lookup[self.oxygens[i].resindex]
            assert b==i, "b is not i"+str(i)+' '+str(self.oxygens[i].resindex)+ " b="+str(b)
            # remember, neigh_As is the list of donors in the neighborhood of the i-th acceptor
            for a in np.unique(lookup[neigh_As[cond].resindices]).tolist():  self.M[b,a], self.M[a,b] = 1,1
        return self.M

    @property
    def base_group(self): return self.oxygens


