#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: itim
    ============
"""

from   multiprocessing   import Process, Queue
import numpy             as np
from   scipy.spatial     import cKDTree
from   scipy.spatial     import Delaunay
from   scipy.interpolate import LinearNDInterpolator
from __builtin__         import zip as builtin_zip
from   pytim             import utilities
import pytim

class ITIM(pytim.PYTIM):
    """ Identifies the interfacial molecules at macroscopically
        flat interfaces.

        :param Universe universe:               the MDAnalysis universe
        :param float mesh:                      the grid spacing used for the testlines
        :param float alpha:                     the probe sphere radius
        :param str normal:                      the macroscopic interface normal direction 'x','y' or 'z' (by default is 'guess')
        :param AtomGroup itim_group:            identify the interfacial molecules from this group
        :param dict radii_dict:                 dictionary with the atomic radii of the elements in the itim_group.
                                                If None is supplied, the default one (from MDAnalysis) will be used.
        :param int max_layers:                  the number of layers to be identified
        :param float cluster_cut:               cutoff used for neighbors or density-based cluster search (default: None disables the cluster analysis)
        :param float cluster_threshold_density: Number density threshold for the density-based cluster search. 'auto' determines the threshold automatically. Default: None uses simple neighbors cluster search, if cluster_cut is not None
        :param bool molecular:                  Switches between search of interfacial molecules / atoms (default: True)
        :param bool info:                       print additional info
        :param bool multiproc:                  parallel version (default: True. Switch off for debugging)

        Example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import *
        >>>
        >>> u         = mda.Universe(WATER_GRO)
        >>> oxygens   = u.select_atoms("name OW")
        >>>
        >>> interface = pytim.ITIM(u, alpha=2.0, max_layers=4,molecular=False)
        >>>
        >>> print interface.layers[0,0]  # first layer, upper
        <AtomGroup with 406 atoms>

    """
    @property
    def layers(self):

        """ Access the layers as numpy arrays of AtomGroups

        The object can be sliced as usual with numpy arrays, so, for example:

        >>> interface.layers[0,:]  # upper side (0), all layers
        array([<AtomGroup with 406 atoms>, <AtomGroup with 411 atoms>,
               <AtomGroup with 414 atoms>, <AtomGroup with 378 atoms>], dtype=object)

        >>> interface.layers[1,0]  # lower side (1), first layer (0)
        <AtomGroup with 406 atoms>

        >>> interface.layers[:,0:3] # 1st - 3rd layer (0:3), on both sides
        array([[<AtomGroup with 406 atoms>, <AtomGroup with 411 atoms>,
                <AtomGroup with 414 atoms>],
               [<AtomGroup with 406 atoms>, <AtomGroup with 418 atoms>,
                <AtomGroup with 399 atoms>]], dtype=object)

        >>> interface.layers[1,0:4:2] # 1st - 4th layer, with a stride of 2 (0:4:2), lower side (1)
        array([<AtomGroup with 406 atoms>, <AtomGroup with 399 atoms>], dtype=object)

        """
        return self._layers


    def __init__(self,universe,mesh=0.4,alpha=2.0,normal='guess',itim_group=None,radii_dict=None,
                 max_layers=1,cluster_cut=None,cluster_threshold_density=None, molecular=True,extra_cluster_groups=None,
                 info=False,multiproc=True):

        #TODO add type checking for each MDA class passed


        self.symmetry='planar'
        self.universe=universe
        self.cluster_threshold_density = cluster_threshold_density
        self.target_mesh=mesh
        self.alpha=alpha
        self.max_layers=max_layers
        self._layers=np.empty([2,max_layers],dtype=type(universe.atoms))
        self.info=info
        self.normal=None
        self.PDB={}
        try:
            self.all_atoms = self.universe.select_atoms('all')
        except:
            raise Exception(self.WRONG_UNIVERSE)
        self.molecular=molecular

        self.cluster_cut          = cluster_cut
        self.extra_cluster_groups = extra_cluster_groups
        self.itim_group           = itim_group

        self._define_groups()

        self._assign_normal(normal)

        self.assign_radii(radii_dict)
        self._sanity_checks()

        self.grid          = None
        self.use_threads   = False
        self.use_kdtree    = True
        self.use_multiproc = multiproc

        pytim.PatchTrajectory(universe.trajectory,self)
        self._assign_layers()


    def _assign_mesh(self):
        """ determine a mesh size for the testlines that is compatible with the simulation box
        """
        box = utilities.get_box(self.universe,self.normal)
        self.mesh_nx=int(np.ceil(box[0]/self.target_mesh))
        self.mesh_ny=int(np.ceil(box[1]/self.target_mesh))
        self.mesh_dx=box[0]/self.mesh_nx
        self.mesh_dy=box[1]/self.mesh_ny
        self.delta=np.minimum(self.mesh_dx,self.mesh_dy)/10.
        if(self.use_kdtree==True):
            _x,_y = np.mgrid[0:box[0]:self.mesh_dx ,  0:box[1]:self.mesh_dy]
            self.meshpoints = builtin_zip(_x.ravel(), _y.ravel())
            # cKDTree requires a box vetor with length double the dimension, see other note
            # in this module
            _box=np.zeros(4)
            _box[:2]=box[:2]
            self.meshtree   = cKDTree(self.meshpoints,boxsize=_box)

    def _touched_lines(self,atom,_x,_y,_z,_radius):
        # NOTE: kdtree might be slower than bucketing in some cases
        if (self.use_kdtree==True) : # this is True by default
            return self.meshtree.query_ball_point([_x[atom],_y[atom]],_radius[atom]+self.alpha)
        else: # For some large configurations this fails. Don't switch off use_kdtree
            _dist=_radius[atom] + self.alpha + self.delta
            index_x = np.arange(
                np.floor((_x[atom]-_dist)/self.mesh_dx),
                np.ceil ((_x[atom]+_dist)/self.mesh_dx)
                )
            index_y = np.arange(
                np.floor((_y[atom]-_dist)/self.mesh_dy),
                np.ceil ((_y[atom]+_dist)/self.mesh_dy)
                )
            _distmap = ( (index_x*self.mesh_dx-_x[atom]).reshape(len(index_x),1)**2+
                        (index_y*self.mesh_dy -_y[atom])**2 )

            _xx, _yy  = np.where(_distmap<=(self.alpha+_radius[atom])**2)

            # now we need to go back to the real space map. Whenever
            # index_x (or index_y) is < 0 || > box we need to wrap it to
            # the other end of the box.
            sel_x = index_x[_xx]
            sel_y = index_y[_yy]
            sel_x[sel_x<0]+=self.mesh_nx
            sel_y[sel_y<0]+=self.mesh_ny
            sel_x[sel_x>=self.mesh_nx]-=self.mesh_nx
            sel_y[sel_y>=self.mesh_ny]-=self.mesh_ny
            print np.array([sel_x,sel_y]).astype(int)
            return np.ravel_multi_index(np.array([sel_x,sel_y]).astype(int),(self.mesh_nx,self.mesh_ny))

    def _assign_one_side(self,uplow,sorted_atoms,_x,_y,_z,
                        _radius,queue=None):
        _layers=[]
        for layer in range(0,self.max_layers) :
            # this mask tells which lines have been touched.
            mask = self.mask[uplow][layer]
            _inlayer=[]
            # atom here goes to 0 to #sorted_atoms, it is not a MDAnalysis index/atom
            for atom in sorted_atoms:
                if self._seen[uplow][atom] != 0 :
                    continue

                touched_lines  = self._touched_lines(atom,_x,_y,_z,_radius)
                _submask = mask[touched_lines]

                if(len(_submask[_submask==0])==0):
                    # no new contact, let's move to the next atom
                    continue

                # let's mark now:
                # 1) the touched lines
                mask[touched_lines]=1

                # 2) the sorted atom
                self._seen[uplow][atom]=layer+1 ; # start counting from 1, 0 will be

                # 3) if all lines have been touched, create a group that includes all atoms in this layer
                if np.sum(mask) == len(mask):  #NOTE that checking len(mask[mask==0])==0 is slower.
                    _inlayer_indices   = np.flatnonzero(self._seen[uplow]==layer+1)
                    _inlayer_group     = self.cluster_group[_inlayer_indices]

                    if self.molecular == True:
                        # we first select the (unique) residues corresponding to _inlayer_group, and then
                        # we create  group of the atoms belonging to them, with _inlayer_group.residues.atoms
                        _tmp   = _inlayer_group.residues.atoms
                        # and we copy it back to _inlayer_group
                        _inlayer_group = _tmp
                        # now we need the indices within the cluster_group, of those atoms which are in the
                        # molecular layer group
                        _indices = np.flatnonzero(np.in1d(self.cluster_group.atoms.ids,_inlayer_group.atoms.ids))
                        # and update the tagged, sorted atoms
                        self._seen[uplow][_indices] = layer + 1

                    # one of the two layers (upper,lower) or both are empty
                    assert _inlayer_group , self.EMPTY_LAYER

                    _layers.append(_inlayer_group)
                    break
        if(queue==None):
            return _layers
        else:
            queue.put(_layers)

    def _sanity_checks(self):

        assert isinstance(self.alpha,float) or isinstance(self.alpha,int),            self.ALPHA_NAN
        assert self.alpha > 0 ,                                                       self.ALPHA_NEGATIVE
        assert self.alpha < np.amin(self.universe.dimensions[:3]) ,                   self.ALPHA_LARGE

        assert isinstance(self.target_mesh,int) or isinstance(self.target_mesh,float),self.MESH_NAN
        assert self.target_mesh > 0 ,                                                 self.MESH_NEGATIVE
        assert self.target_mesh < np.amin(self.universe.dimensions[:3]) ,             self.MESH_LARGE

        if(self.cluster_cut is not None):
            assert len(self.cluster_cut)== 1 or len(self.cluster_cut) == 1+len(self.extra_cluster_groups) , self.MISMATCH_CLUSTER_SEARCH
        else:
            assert self.extra_cluster_groups is None ,                                self.UNDEFINED_CLUSTER_SEARCH

        try:
            np.arange(int(self.alpha/self.target_mesh))
        except:
            print("Error while initializing ITIM: alpha (%f) too large or mesh (%f) too small" % self.alpha,self.target_mesh)
            raise ValueError


    def _assign_layers(self):
        """ Determine the ITIM layers.


        """
        self._assign_mesh()
        delta = self.delta ;
        mesh_dx = self.mesh_dx ; mesh_dy = self.mesh_dy
        up=0 ; low=1
        self.layers_ids=[[],[]] ;# upper, lower
        self.mask=np.zeros((2,self.max_layers,self.mesh_nx*self.mesh_ny),
                            dtype=int);

        # this can be used later to shift back to the original shift
        self.original_positions=np.copy(self.universe.atoms.positions[:])

        self.universe.atoms.pack_into_box()

        if(self.cluster_cut is not None): # groups have been checked already in _sanity_checks()
            box = np.copy(self.universe.dimensions[:6])
            labels,counts,n_neigh = utilities.do_cluster_analysis_DBSCAN(self.itim_group,self.cluster_cut[0],box,self.cluster_threshold_density,self.molecular)
            labels = np.array(labels)
            label_max = np.argmax(counts) # the label of atoms in the largest cluster
            ids_max   = np.where(labels==label_max)[0]  # the indices (within the group) of the
                                                        # atoms belonging to the largest cluster
            self.cluster_group = self.itim_group[ids_max]

            self.n_neighbors = n_neigh

        else:
            self.cluster_group=self.itim_group ;
        utilities.centerbox(self.universe,center_direction=self.normal)
        self.center(self.cluster_group,self.normal)
        utilities.centerbox(self.universe,center_direction=self.normal)

        # first we label all atoms in itim_group to be in the gas phase
        self.itim_group.atoms.bfactors = 0.5
        # then all atoms in the largest group are labelled as liquid-like
        self.cluster_group.atoms.bfactors = 0

        _radius=self.cluster_group.radii
        self._seen=[[],[]]
        size = len(self.cluster_group.positions)
        self._seen[up] =np.zeros(size,dtype=np.int8)
        self._seen[low]=np.zeros(size,dtype=np.int8)

        _x=utilities.get_x(self.cluster_group,self.normal)
        _y=utilities.get_y(self.cluster_group,self.normal)
        _z=utilities.get_z(self.cluster_group,self.normal)
        sort = np.argsort( _z + _radius * np.sign(_z) )
        # NOTE: np.argsort returns the sorted *indices*

        if self.use_multiproc:
            # so far, it justs exploit a simple scheme splitting
            # the calculation between the two sides. Would it be
            # possible to implement easily 2d domain decomposition?
            proc=[None,None]
            queue = [ Queue() , Queue() ]
            proc[up]  = Process(target=self._assign_one_side,
                                args=(up,sort[::-1],_x,_y,_z,_radius,queue[up]))
            proc[low] = Process(target=self._assign_one_side,
                                args=(low,sort[::] ,_x,_y,_z,_radius,queue[low]))

            for p in proc: p.start()
            for uplow  in [up,low]:
                for index,group in enumerate(queue[uplow].get()):
                    # cannot use self._layers[uplow][index] = group, otherwise info about universe is lost (do not know why yet)
                    # must use self._layers[uplow][index] = self.universe.atoms[group.ids]
                    self._layers[uplow][index] = self.universe.atoms[group.ids]
            for p in proc: p.join()
        else:
            for index,group in enumerate(self._assign_one_side(up,sort[::-1],_x,_y,_z,_radius)):
                self._layers[up][index] = group
            for index,group in enumerate(self._assign_one_side(low,sort[::],_x,_y,_z,_radius)):
                self._layers[low][index] = group


        # assign bfactors to all layers
        for uplow in [up,low]:
            for _nlayer,_layer in enumerate(self._layers[uplow]):
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


