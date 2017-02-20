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

        # TODO 
    """

    def __init__(self,universe,alpha=2.0,symmetry='spherical',itim_group=None,radii_dict=None,
                 max_layers=1,cluster_cut=None,molecular=True,extra_cluster_groups=None,
                 info=False,multiproc=True):

        #TODO add type checking for each MDA class passed
        self.universe=universe
        self.alpha=alpha
        self.max_layers=max_layers
        self.info=info
        self.normal=None
        try: 
            self.all_atoms = self.universe.select_atoms('all')
        except:
            raise Exception(self.WRONG_UNIVERSE)
        self._layers=None
        self.molecular=molecular

        self.cluster_cut          = cluster_cut
        self.extra_cluster_groups = extra_cluster_groups
        self.itim_group           = itim_group

        self._define_groups()

        self._assign_symmetry(symmetry)
        self.assign_radii(radii_dict)
        self._sanity_checks()

        self.grid          = None
        self.use_threads   = False
        self.use_kdtree    = True
        self.use_multiproc = multiproc

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
        return t.simplices [ [np.max(distance.cdist(t.points[simplex],t.points[simplex],'euclidean'))>=threshold for simplex in t.simplices] ]

    def circumcircle(self,simplex):
    
        def sq_norm(v): #squared norm 
            return np.linalg.norm(v)**2
        points = self.triangulation.points
        # Just a test, taken from https://plot.ly/python/alpha-shapes/
        A=[points[simplex[k]] for k in range(3)]
        M=[[1.0]*4]
        M+=[[sq_norm(A[k]), A[k][0], A[k][1], 1.0 ] for k in range(3)]
        M=np.asarray(M, dtype=np.float32)
        S=np.array([0.5*np.linalg.det(M[1:,[0,2,3]]), -0.5*np.linalg.det(M[1:,[0,1,3]])])
        a=np.linalg.det(M[1:, 1:])
        b=np.linalg.det(M[1:, [0,1,2]])
        return np.sqrt(b/a+sq_norm(S)/a**2) #center=S/a, radius=np.sqrt(b/a+sq_norm(S)/a**2)

    def alpha_shape(self,alpha):
        box    = self.universe.dimensions[:3]
        delta  = 2.*self.alpha+1e-6
        points = self.itim_group.positions[:]
        extrapoints  = [] 
        # add points at the vertices of the expanded (by 2 alpha) box
        for dim in range(8):
                # [0,0,0],[0,0,1],[0,1,0],...,[1,1,1]
                tmp = np.array(np.array(list(np.binary_repr(dim,width=3)),dtype=np.int8),dtype=np.float)
                tmp *=(box+delta)
                tmp[tmp<box/2.]-=delta
                extrapoints.append(tmp)
        points = np.append(points,extrapoints)
        time=dict()
        utilities.lap()

        self.triangulation = Delaunay(self.itim_group.positions[:]) 
        time['triangulate']=utilities.lap()

        prefiltered = self.alpha_prefilter(self.triangulation, alpha)
        time['prefilter']=utilities.lap()

        alpha_shape = prefiltered [ [ self.circumcircle(simplex) >=self.alpha for simplex in prefiltered      ]   ]
        time['alpha']=utilities.lap()


        print time
        _ids = np.unique(alpha_shape)
        # remove the indices corresponding to the 8 additional points
        ids =_ids
        # ids = _ids[_ids<len(points)-8]
        print len(self.triangulation.simplices),len(prefiltered),len(alpha_shape)
        self.itim_group[ids].write('tt.pdb')
        exit()
        print "simplices:"
        print self.triangulation.simplices
        #cc = lambda simplex,points: self.circumcircle(points,simplex)[1]
        print "CC:", [self.circumcircle(simplex) for simplex in self.triangulation.simplices]
    
        exit()
        #TODO: the option 'axis' will be introduced in numpy.unique() (numpy 1.13), check it out and replace this code.
        #border_indices = np.unique(alpha_complex.neighbors[np.where(dela.neighbors<0)[0]])
    
    def assign_layers(self):
        """ Determine the GITIM layers.


        """
        # this can be used later to shift back to the original shift
        self.reference_position=self.universe.atoms[0].position[:]

        self.universe.atoms.pack_into_box()

        if(self.cluster_cut is not None): # groups have been checked already in _sanity_checks()
            labels,counts = utilities.do_cluster_analysis_DBSCAN(self.itim_group,self.cluster_cut[0],self.universe.dimensions[:6])
            labels = np.array(labels)
            label_max = np.argmax(counts) # the label of atoms in the largest cluster
            ids_max   = np.where(labels==label_max)[0]  # the indices (within the group) of the 
                                                        # atoms belonging to the largest cluster
            self.cluster_group = self.itim_group[ids_max]
            
        else:
            self.cluster_group=self.itim_group ;
        if self.symmetry=='spherical':
            self.center(self.cluster_group,'x',halfbox_shift=False)
            self.center(self.cluster_group,'y',halfbox_shift=False)
            self.center(self.cluster_group,'z',halfbox_shift=False)
              
        # first we label all atoms in itim_group to be in the gas phase 
        self.itim_group.atoms.bfactors = 0.5
        # then all atoms in the larges group are labelled as liquid-like
        self.cluster_group.atoms.bfactors = 0

        _radius=self.cluster_group.radii
        size = len(self.cluster_group.positions)
        self._seen=np.zeros(size,dtype=np.int8)
    
        self.alpha_shape(self.alpha)

        self._layers=[]

        for nlayer,layer in enumerate(self._layers):
                _layer.bfactors = _nlayer+1 

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


    def layers(self,side='both',*ids):
        """ Select one or more layers.

        :param str side: 'upper', 'lower' or 'both'
        :param slice ids: the slice corresponding to the layers to be selcted (starting from 0)

        The slice can be used to select a single layer, or multiple, e.g. (using the example of the :class:`ITIM` class) :

        >>> interface.layers('upper')  # all layers, upper side
        [<AtomGroup with 406 atoms>, <AtomGroup with 411 atoms>, <AtomGroup with 414 atoms>, <AtomGroup with 378 atoms>]

        >>> interface.layers('lower',1)  # first layer, lower side
        <AtomGroup with 406 atoms>

        >>> interface.layers('both',0,3) # 1st - 3rd layer, on both sides
        [[<AtomGroup with 406 atoms>, <AtomGroup with 411 atoms>, <AtomGroup with 414 atoms>], [<AtomGroup with 406 atoms>, <AtomGroup with 418 atoms>, <AtomGroup with 399 atoms>]]

        >>> interface.layers('lower',0,4,2) # 1st - 4th layer, with a stride of 2, lower side
        [<AtomGroup with 406 atoms>, <AtomGroup with 399 atoms>]


        """
        _options={'both':slice(None),'upper':0,'lower':1}
        _side=_options[side]
        if len(ids) == 0:
            _slice = slice(None)
        if len(ids) == 1:
            _slice = ids[0]-1
        if len(ids) == 2:
            _slice = slice(ids[0],ids[1])
        if len(ids) == 3:
            _slice = slice(ids[0],ids[1],ids[2])

        if side != 'both':
            return self._layers[_side][_slice]
        else:
            return [ sub[_slice] for sub in self._layers]



#
