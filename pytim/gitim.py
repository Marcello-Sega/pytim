#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: gtim
    ============
"""

from   multiprocessing   import Process, Queue
import numpy             as np
from   scipy.spatial     import cKDTree
from   scipy.spatial     import Delaunay
from   scipy.spatial     import distance
from   scipy.interpolate import LinearNDInterpolator
import itertools
from __builtin__         import zip as builtin_zip
from   pytim             import utilities
import pytim

class GITIM(pytim.PYTIM):
    """ Identifies the interfacial molecules at macroscopically
        flat interfaces.

        :param Universe universe:      the MDAnalysis universe
        :param float alpha:            the probe sphere radius
        :param AtomGroup itim_group:   identify the interfacial molecules from this group
        :param dict radii_dict:        dictionary with the atomic radii of the elements in the itim_group.
                                       If None is supplied, the default one (from GROMOS 43a1) will be used.
        :param int max_layers:         the number of layers to be identified
        :param bool info:              print additional info
        :param bool multiproc:         parallel version (default: True. Switch off for debugging)

        Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from   pytim.datafiles import *
        >>>
        >>> u       = mda.Universe(MICELLE_PDB)
        >>> g       = u.select_atoms('resname DPC')
        >>> radii=pytim_data.vdwradii(G43A1_TOP)
        >>>
        >>> interface =pytim.GITIM(u,itim_group=g,molecular=False,symmetry='spherical',alpha=2.5,)
        >>> layer = interface.layers[0]
        >>> interface.writepdb('gitim.pdb',centered=False)
        >>> print layer
        <AtomGroup with 558 atoms>

    """

    def __init__(self,universe,alpha=2.0,symmetry='spherical',normal='guess',itim_group=None,radii_dict=None,
                 max_layers=1,cluster_cut=None,cluster_threshold_density=None,molecular=True,extra_cluster_groups=None,
                 info=False,multiproc=True):

        #TODO add type checking for each MDA class passed

        # dynamic monkey patch to change the behavior of the frame property

        self.universe=universe
        self.cluster_threshold_density = cluster_threshold_density
        self.alpha=alpha
        self.max_layers=max_layers
        self._layers=np.empty([max_layers],dtype=type(universe.atoms))
        self.info=info
        self.normal=None
        try:
            self.all_atoms = self.universe.select_atoms('all')
        except:
            raise Exception(self.WRONG_UNIVERSE)
        self.molecular=molecular

        self.cluster_cut          = cluster_cut
        self.extra_cluster_groups = extra_cluster_groups
        self.itim_group           = itim_group

        self._define_groups()

        self._assign_symmetry(symmetry)

        if(self.symmetry=='planar'):
            self._assign_normal(normal)

        self.assign_radii(radii_dict)
        self._sanity_checks()

        self.grid          = None
        self.use_threads   = False
        self.use_kdtree    = True
        self.use_multiproc = multiproc

        pytim.PatchTrajectory(universe.trajectory,self)
        self._assign_layers()

    def _assign_symmetry(self,symmetry):

        assert self.itim_group is not None, self.UNDEFINED_ITIM_GROUP
        if symmetry=='guess':
            assert False ,  "To be implemented"
            #self.normal=utilities.guess_symmetry(self.universe,self.itim_group)
        else:
            assert symmetry in self.symmetry_dict, self.WRONG_DIRECTION
            self.symmetry = symmetry

    def _sanity_checks(self):

        assert self.alpha > 0 ,                                           self.ALPHA_NEGATIVE
        assert self.alpha < np.amin(self.universe.dimensions[:3]) ,       self.ALPHA_LARGE

        if(self.cluster_cut is not None):
            assert len(self.cluster_cut)== 1 or len(self.cluster_cut) == 1+len(self.extra_cluster_groups) , self.MISMATCH_CLUSTER_SEARCH
        else:
            assert self.extra_cluster_groups is None , self.UNDEFINED_CLUSTER_SEARCH


    def alpha_prefilter(self,triangulation,alpha):
        t=triangulation
        threshold = 2.*alpha
        return t.simplices [ [np.max(distance.cdist(t.points[simplex],t.points[simplex],'euclidean'))>=threshold+2.*np.min(t.radii[simplex]) for simplex in t.simplices] ]

    def circumradius(self,simplex):

        points = self.triangulation.points
        radii  = self.triangulation.radii

        R      = []
        r_i    = points[simplex]
        R_i    = radii[simplex]
        d      = (R_i[0]- R_i)[1:]

        r_i2   = np.sum(r_i**2,axis=1)
        d_2    = d**2

        M      = (r_i[0] - r_i)[1:]
        s      = ((r_i2[0] - r_i2[1:] - d_2[0] + d_2))/2.

        try:
            u      = np.dot(np.linalg.inv(M),d)
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.message:
                print "Warning, singular matrix for ",r_i
                return 0 # TODO is this correct? The singular matrix most likely comes out of points alinged in the plane
            else:
                raise
        v      = r_i[1]-r_i[0]

        A      = - (R_i[0] - np.dot(u,v) )
        B      = np.linalg.norm(R_i[0]*u+v)
        C      = 1-np.sum(u**2)
        R.append( ( A + B )/C )
        R.append( ( A - B )/C )
        R=np.array(R)
        positiveR = R[R>=0]
        if positiveR.size == 1:
            return np.min(positiveR)
        else:
            return 1e6

    def alpha_shape(self,alpha):
        #print  utilities.lap()
        box    = self.universe.dimensions[:3]
        delta  = 2.*self.alpha+1e-6
        points = self.cluster_group.positions[:]
        extrapoints = np.copy(points)
        nrealpoints = len(points)
        extraids    = np.arange(len(points),dtype=np.int)
        tmpPoints   = []
        tmpIDs      = []
        for shift in np.array(list(itertools.product([1,-1,0],repeat=3))):
            if(np.sum(shift*shift)): # avoid [0,0,0]
                # this needs some explanation:
                # if shift ==0  -> the condition is always true
                # if shift ==1  -> the condition is x > box - delta
                # if shift ==-1 -> the condition is -x > 0 - delta -> x < delta
                # Requiring np.all() to be true makes the logical and returns (axis=1) True for all indices whose atoms satisfy the condition
                selection = np.all(shift * points >= shift*shift*( (box + shift * box)/2. - delta)   ,axis=1)
                # add the new points at the border of the box
                extrapoints=np.append(extrapoints,points[selection]-shift*box,axis=0)
                # we keep track of the original ids.
                extraids=np.append(extraids,np.where(selection)[0])

        # add points at the vertices of the expanded (by 2 alpha) box
        for dim in range(8):
                # [0,0,0],[0,0,1],[0,1,0],...,[1,1,1]
                tmp = np.array(np.array(list(np.binary_repr(dim,width=3)),dtype=np.int8),dtype=np.float)
                tmp *=(box+delta)
                tmp += (np.random.random(3)-0.5)*box*1e-8 # the random gitter (rescaled to be small wrt the box) is added to prevent coplanar points
                tmp[tmp<box/2.]-=delta
                tmp=np.reshape(tmp,(1,3))
                extrapoints=np.append(extrapoints,tmp,axis=0)
                extraids=np.append(extraids,-1)

        #print utilities.lap()
        self.triangulation = Delaunay(extrapoints)
        self.triangulation.radii = np.append(self.cluster_group.radii[extraids[extraids>=0]],np.zeros(8))
        #print utilities.lap()

        prefiltered = self.alpha_prefilter(self.triangulation, alpha)

        #print utilities.lap()
        a_shape = prefiltered [ [ self.circumradius(simplex) >=self.alpha for simplex in prefiltered      ]   ]

        #print utilities.lap()
        _ids = np.unique(a_shape.flatten())

        # remove the indices corresponding to the 8 additional points, which have extraid==-1
        ids = _ids[np.logical_and(_ids>=0 , _ids < nrealpoints)]

        #print utilities.lap()
        return ids

    def _assign_layers(self):
        """ Determine the GITIM layers.


        """
        # this can be used later to shift back to the original shift
        self.original_positions=np.copy(self.universe.atoms.positions[:])
        self.universe.atoms.pack_into_box()

        if(self.cluster_cut is not None): # groups have been checked already in _sanity_checks()
            labels,counts,n_neigh = utilities.do_cluster_analysis_DBSCAN(self.itim_group,self.cluster_cut[0],self.universe.dimensions[:6],self.cluster_threshold_density,self.molecular)
            labels = np.array(labels)
            label_max = np.argmax(counts) # the label of atoms in the largest cluster
            ids_max   = np.where(labels==label_max)[0]  # the indices (within the group) of the
                                                        # atoms belonging to the largest cluster
            self.cluster_group = self.itim_group[ids_max]

        else:
            self.cluster_group=self.itim_group ;

        if self.symmetry=='planar':
            utilities.centerbox(self.universe,center_direction=self.normal)
            self.center(self.cluster_group,self.normal)
            utilities.centerbox(self.universe,center_direction=self.normal)
        if self.symmetry=='spherical':
            self.center(self.cluster_group,'x',halfbox_shift=False)
            self.center(self.cluster_group,'y',halfbox_shift=False)
            self.center(self.cluster_group,'z',halfbox_shift=False)
            self.universe.atoms.pack_into_box(self.universe.dimensions[:3])



        # first we label all atoms in itim_group to be in the gas phase
        self.itim_group.atoms.bfactors = 0.5
        # then all atoms in the larges group are labelled as liquid-like
        self.cluster_group.atoms.bfactors = 0

        _radius=self.cluster_group.radii
        size = len(self.cluster_group.positions)
        self._seen=np.zeros(size,dtype=np.int8)

        alpha_ids = self.alpha_shape(self.alpha)

        # only the 1st layer is implemented in gitim so far
        if self.molecular == True:
            self._layers[0] = self.cluster_group[alpha_ids].residues.atoms
        else:
            self._layers[0] = self.cluster_group[alpha_ids]

        for layer in self._layers:
                layer.bfactors = 1

        # reset the interpolator
        self._interpolator=None

    def _generate_periodic_border_2D(self, group):
        _box = utilities.get_box(group.universe,self.normal)

        positions=utilities.get_pos(group,self.normal)

        shift=np.diagflat(_box)

        eps = min(2.*self.alpha,_box[0],_box[1])
        L = [eps,eps]
        U = [_box[0] - eps  , _box[1] - eps  ]

        pos=positions[:]
        Lx= positions[positions[:,0]<=L[0]]+shift[0]
        Ly= positions[positions[:,1]<=L[1]]+shift[1]
        Ux= positions[positions[:,0]>=U[0]]-shift[0]
        Uy= positions[positions[:,1]>=U[1]]-shift[1]

        LxLy = positions[np.logical_and(positions[:,0]<=L[0], positions[:,1]<=L[1])] + (shift[0]+shift[1])
        UxUy = positions[np.logical_and(positions[:,0]>=U[0], positions[:,1]>=U[1])] - (shift[0]+shift[1])
        LxUy = positions[np.logical_and(positions[:,0]<=L[0], positions[:,1]>=U[1])] + (shift[0]-shift[1])
        UxLy = positions[np.logical_and(positions[:,0]>=U[0], positions[:,1]<=L[1])] - (shift[0]-shift[1])

        return np.concatenate((pos,Lx,Ly,Ux,Uy,LxLy,UxUy,LxUy,UxLy))

    def triangulate_layer(self,layer=1):
        """ Triangulate a layer

            :param int layer:  (default: 1) triangulate this layer (on both sides of the interface)
            :return list triangulations:  a list of two Delaunay triangulations, which are also stored in self.surface_triangulation
        """
        assert len(self._layers[0])>=layer , self.UNDEFINED_LAYER

        box = self.universe.dimensions[:3]

        upper = self._layers[0][layer-1]
        lower = self._layers[1][layer-1]

        upperpos = self._generate_periodic_border_2D(upper)
        lowerpos = self._generate_periodic_border_2D(lower)

        self.surface_triangulation = [None,None]
        self.trimmed_surface_triangles = [None,None]
        self.triangulation_points= [None,None]
        self.surface_triangulation[0] = Delaunay(upperpos[:,0:2])
        self.surface_triangulation[1] = Delaunay(lowerpos[:,0:2])
        self.triangulation_points[0] = upperpos[:]
        self.triangulation_points[1] = lowerpos[:]
        self.trimmed_surface_triangles[0] = utilities.trim_triangulated_surface(self.surface_triangulation[0],box)
        self.trimmed_surface_triangles[1] = utilities.trim_triangulated_surface(self.surface_triangulation[1],box)
        return self.surface_triangulation

    def _initialize_distance_interpolator(self,layer):
        if self._interpolator == None :
            # we don't know if previous triangulations have been done on the same
            # layer, so just in case we repeat it here. This can be fixed in principle
            # with a switch
            self.triangulate_layer(layer)

            self._interpolator= [None,None]
            self._interpolator[0] = LinearNDInterpolator(self.surface_triangulation[0],
                                                         self.triangulation_points[0][:,2])
            self._interpolator[1] = LinearNDInterpolator(self.surface_triangulation[1],
                                                         self.triangulation_points[1][:,2])

    def interpolate_surface(self,positions,layer):
        self._initialize_distance_interpolator(layer)
        upper_set = positions[positions[:,2]>=0]
        lower_set = positions[positions[:,2]< 0]
        #interpolated values of upper/lower_set on the upper/lower surface
        upper_int = self._interpolator[0](upper_set[:,0:2])
        lower_int = self._interpolator[1](lower_set[:,0:2])
        #copy everything back to one array with the correct order
        elevation = np.zeros(len(positions))
        elevation[np.where(positions[:,2]>=0)] = upper_int
        elevation[np.where(positions[:,2]< 0)] = lower_int
        return elevation

    @property
    def layers(self):
        """ Access the layers as numpy arrays of AtomGroups

        The object can be sliced as usual with numpy arrays.
        """
        return self._layers

#
