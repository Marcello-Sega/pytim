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

        >>> import MDAnalysis as mda
        >>> import pytim  
        >>> from   pytim.datafiles import *
        >>> 
        >>> u       = mda.Universe(MICELLE_PDB)
        >>> g       = u.select_atoms('resname DPC')
        >>> radii=pytim_data.vdwradii(G43A1_TOP)
        >>> 
        >>> interface =pytim.GITIM(u,itim_group=g,molecular=False,symmetry='spherical',alpha=2.5,)
        >>> layer = interface.layers(1)
        >>> interface.writepdb('gitim.pdb',centered=False)
        >>> print layer
        <AtomGroup with 558 atoms>

    """

    def __init__(self,universe,alpha=2.0,symmetry='spherical',itim_group=None,radii_dict=None,
                 max_layers=1,cluster_cut=None,molecular=True,extra_cluster_groups=None,
                 info=False,multiproc=True):

        #TODO add type checking for each MDA class passed

        # dynamic monkey patch to change the behavior of the frame property
        pytim.PatchTrajectory(universe.trajectory,self)

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
         
        u      = np.dot(np.linalg.inv(M),d)
        v      = r_i[1]-r_i[0]
        
        A      = - (R_i[0] - np.dot(u,v) )
        B      = np.linalg.norm(R_i[0]*u+v)
        C      = 1-np.sum(u**2)
        R.append( ( A + B )/C )
        R.append( ( A - B )/C )
        R=np.array(R)
        positiveR = R[R>=0]
        return np.min(positiveR) if positiveR.size == 1 else 0

##       check configuration  ##
##            print r_i
##            print R_i
##            from mpl_toolkits.mplot3d import Axes3D
##            import matplotlib.pyplot as plt
##            
##            def drawSphere(xCenter, yCenter, zCenter, r):
##                #draw sphere
##                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
##                x=np.cos(u)*np.sin(v)
##                y=np.sin(u)*np.sin(v)
##                z=np.cos(v)
##                # shift and scale sphere
##                x = r*x + xCenter
##                y = r*y + yCenter
##                z = r*z + zCenter
##                return (x,y,z)
##            
##            fig = plt.figure()
##            ax = fig.add_subplot(111, projection='3d')
##
##            # draw a sphere for each data point
##            for (xi,yi,zi,ri) in zip(r_i[:,0],r_i[:,1],r_i[:,2],R_i):
##                (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
##                ax.plot_wireframe(xs, ys, zs, color="r")
##            plt.show()
##            exit()

        

    def alpha_shape(self,alpha):
        box    = self.universe.dimensions[:3]
        delta  = 2.*self.alpha+1e-6
        points = self.cluster_group.positions[:]
        extrapoints  = [] 
        # add points at the vertices of the expanded (by 2 alpha) box
        for dim in range(8):
                # [0,0,0],[0,0,1],[0,1,0],...,[1,1,1]
                tmp = np.array(np.array(list(np.binary_repr(dim,width=3)),dtype=np.int8),dtype=np.float)
                tmp *=(box+delta)
                tmp[tmp<box/2.]-=delta
                extrapoints.append(tmp)
        points = np.append(points,extrapoints,axis=0)

#       time=dict();utilities.lap()

        self.triangulation = Delaunay(points) 
        self.triangulation.radii = np.append(self.cluster_group.radii[:],np.zeros(8))

#       time['triangulate']=utilities.lap()

        prefiltered = self.alpha_prefilter(self.triangulation, alpha)
#       time['prefilter']=utilities.lap()

        alpha_shape = prefiltered [ [ self.circumradius(simplex) >=self.alpha for simplex in prefiltered      ]   ]
#       time['alpha']=utilities.lap()
#       print time
#       print len(self.triangulation.simplices),len(prefiltered),len(alpha_shape),len(_ids)

        _ids = np.unique(alpha_shape)

        # remove the indices corresponding to the 8 additional points
        #ids =_ids
        ids = _ids[_ids<len(points)-8]

        return ids
    
    def _assign_layers(self):
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
    
        alpha_ids = self.alpha_shape(self.alpha)

        self._layers=[self.cluster_group[alpha_ids]]

        for nlayer,layer in enumerate(self._layers):
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


    def layers(self,*ids):
        """ Select one or more layers.

        :param slice ids: the slice corresponding to the layers to be selcted (starting from 0)

        The slice can be used to select a single layer, or multiple, e.g. using the example of the :class:`GITIM` class :

        """
        if len(ids) == 0:
            _slice = slice(None)
        if len(ids) == 1:
            _slice = ids[0]-1
        if len(ids) == 2:
            _slice = slice(ids[0],ids[1])
        if len(ids) == 3:
            _slice = slice(ids[0],ids[1],ids[2])

        return self._layers[_slice]

#
