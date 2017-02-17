#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: pytim
    =============
"""

#from threading import Thread
from multiprocessing import Process, Queue
import numpy as np
from scipy.spatial import *
from scipy.interpolate import *
from MDAnalysis.core.AtomGroup   import *
from pytim.datafiles import *
from pytim import utilities
from MDAnalysis.topology import tables
from dbscan import dbscan_inner
import __builtin__

class ITIM():
    """ Identifies the interfacial molecules at macroscopically
        flat interfaces.

        :param Universe universe:      the MDAnalysis universe
        :param float mesh:             the grid spacing used for the testlines
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
        >>> from pytim.datafiles import *
        >>>
        >>> u         = mda.Universe(WATER_GRO)
        >>> oxygens   = u.select_atoms("name OW")
        >>>
        >>> interface = pytim.ITIM(u, alpha=2.0, max_layers=4,molecular=False)
        >>> interface.assign_layers()
        >>>
        >>> print interface.layers('upper',1)  # first layer, upper
        <AtomGroup with 406 atoms>

        # TODO Add here an example on how to use the variuos other options
    """

    def define_error_messages(self):
        self.ALPHA_NEGATIVE = "parameter alpha in %s.%s must be positive" % ( (__name__) , (self.__class__.__name__) )
        self.ALPHA_LARGE= "parameter alpha in %s.%s must be smaller than the smaller box side" % ( (__name__) , (self.__class__.__name__) )
        self.MESH_NEGATIVE = "parameter mesh in %s.%s must be positive" % ( (__name__) , (self.__class__.__name__) )
        self.MESH_LARGE= "parameter mesh in %s.%s must be smaller than the smaller box side" % ( (__name__) , (self.__class__.__name__) )
        self.UNDEFINED_RADIUS= "one or more atoms do not have a corresponding radius in the default or provided dictionary in %s.%s" % ( (__name__) , (self.__class__.__name__) )
        self.UNDEFINED_CLUSTER_SEARCH= "If extra_cluster_groups is defined, a cluster_cut should e provided ( %s.%s )" % ( (__name__) , (self.__class__.__name__) )
        self.MISMATCH_CLUSTER_SEARCH= "cluster_cut in %s.%s should be either a scalar or an array matching the number of groups (including itim_group" % ( (__name__) , (self.__class__.__name__) )
        self.EMPTY_LAYER="One or more layers are empty"
        self.CLUSTER_FAILURE="Cluster algorithm failed: too small cluster cutoff provided?"
        self.UNDEFINED_LAYER="No layer defined: forgot to call assign_layers() or not enough layers requested"
        self.WRONG_UNIVERSE="Wrong Universe passed to ITIM class"
        




    def __init__(self,universe,mesh=0.4,alpha=2.0,itim_group=None,radii_dict=None,
                 max_layers=1,cluster_cut=None,molecular=True,extra_cluster_groups=None,
                 info=False,multiproc=True):

        #TODO add type checking for each MDA class passed
        self.define_error_messages()
        self.universe=universe
        assert type(universe)==MDAnalysis.core.AtomGroup.Universe , self.WRONG_UNIVERSE
        self.target_mesh=mesh
        self.alpha=alpha
        self.max_layers=max_layers
        self.info=info
        self.all_atoms = self.universe.select_atoms('all')
        self._layers=None
        self.molecular=molecular

        self.cluster_cut          = cluster_cut
        self.extra_cluster_groups = extra_cluster_groups
        self.itim_group           = itim_group

        self._define_groups()
        self._assign_radii(radii_dict)
        self._sanity_checks()

        self.grid          = None
        self.use_threads   = False
        self.use_kdtree    = True
        self.use_multiproc = multiproc


    def _define_groups(self):
        # we first make sure cluster_cut is either None, or an array
        if self.cluster_cut is not None and not isinstance(self.cluster_cut, (list, tuple, np.ndarray)):
            if type(self.cluster_cut) is int or type(self.cluster_cut) is float:
                self.cluster_cut = np.array([float(self.cluster_cut)])
        # same with extra_cluster_groups
        if self.extra_cluster_groups is not None and not isinstance(self.extra_cluster_groups, (list, tuple, np.ndarray)):
            self.extra_cluster_groups = [self.extra_cluster_groups]
   
        # fallback for itim_group 
        if self.itim_group is None:
            self.itim_group = self.all_atoms

    def _assign_radii(self,radii_dict):
        try:
            _groups = self.extra_cluster_groups[:] # deep copy
        except:
            _groups = []
        _groups.append(self.itim_group)
        for _g in _groups:
            # TODO: add a switch that allows to use the atom name instead of the type!
            if _g is not None:
                _types = np.copy(_g.types)
                if np.any(_g.radii is None)  or radii_dict is None : # either radii are not set or dict provided
                    if radii_dict is None:  # radii not set, dict not provided -> fallback to MDAnalysis vdwradii
                        _radii_dict = tables.vdwradii
                else: # use the provided dict.
                    _radii_dict = radii_dict

                _radii = np.zeros(len(_g.types))
                for _atype in np.unique(_types):
                     try:
                         _radii[_types==_atype]=_radii_dict[_atype]
                     except:
                         pass # those types which are not listed in the dictionary
                              # will have radius==0. TODO handle this
                _g.radii=_radii[:] #deep copy
                assert np.all(_g.radii is not None) , self.UNDEFINED_RADIUS
                del _radii
                del _types

    def _sanity_checks(self):

        assert self.alpha > 0 ,                                           self.ALPHA_NEGATIVE
        assert self.alpha < np.amin(self.universe.dimensions[:3]) ,       self.ALPHA_LARGE

        assert self.target_mesh > 0 ,                                     self.MESH_NEGATIVE
        assert self.target_mesh < np.amin(self.universe.dimensions[:3]) , self.MESH_LARGE

        if(self.cluster_cut is not None):
            assert len(self.cluster_cut)== 1 or len(self.cluster_cut) == 1+len(self.extra_cluster_groups) , self.MISMATCH_CLUSTER_SEARCH
        else:
            assert self.extra_cluster_groups is None , self.UNDEFINED_CLUSTER_SEARCH

        try:
            np.arange(int(self.alpha/self.target_mesh))
        except:
            print("Error while initializing ITIM: alpha (%f) too large or mesh (%f) too small" % self.alpha,self.target_mesh)
            raise ValueError


    def writepdb(self,filename='layers.pdb',multiframe=True):
        """ Write the frame to a pdb file, marking the atoms belonging
            to the layers with different beta factor.

        :param filename:   string  -- the output file name
            :param multiframe: boolean -- append to pdb file if True

            Example:

            >>> interface.writepdb('layers.pdb',multiframe=False)

        """

        try:
            PDB=MDAnalysis.Writer(filename, multiframe=True, bonds=False,
                            n_atoms=self.universe.atoms.n_atoms)
            PDB.write(self.universe.atoms)
        except:
            print("Error writing pdb file")





    def _assign_mesh(self):
        """ determine a mesh size for the testlines that is compatible with the simulation box
        """
        self.mesh_nx=int(np.ceil(self.universe.coord.dimensions[0]/
                         self.target_mesh))
        self.mesh_ny=int(np.ceil(self.universe.coord.dimensions[1]/
                         self.target_mesh))
        self.mesh_dx=self.universe.coord.dimensions[0]/self.mesh_nx
        self.mesh_dy=self.universe.coord.dimensions[1]/self.mesh_ny
        self.delta=np.minimum(self.mesh_dx,self.mesh_dy)/10.
        if(self.use_kdtree==True):
            _box = self.universe.coord.dimensions  # TODO normals other than Z!
            _x,_y = np.mgrid[0:_box[0]:self.mesh_dx ,  0:_box[1]:self.mesh_dy]
            self.meshpoints = __builtin__.zip(_x.ravel(), _y.ravel())
            # cKDTree requires a box vetor with length double the dimension, see other note
            # in this module
            self.meshtree   = cKDTree(self.meshpoints,boxsize=_box[:4])

    def _touched_lines(self,atom,_x,_y,_z,_radius):
        # NOTE: kdtree might be slower than bucketing in some cases
        if (self.use_kdtree==True) : # this is False by default
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
                        # atom is an index to cluster_group (== liquid part of itim_group)
                        _inlayer_resindices= _inlayer_group.resids-1
                        # this is the group of molecules in the layer
                        _inlayer_group   = self.universe.residues[_inlayer_resindices].atoms
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

    def _init_NN_search(self,group):
        #NOTE: boxsize shape must be (6,), and the last three elements are overwritten in cKDTree:
        #   boxsize_arr = np.empty(2 * self.m, dtype=np.float64)
        #   boxsize_arr[:] = boxsize
        #   boxsize_arr[self.m:] = 0.5 * boxsize_arr[:self.m]

        # TODO: handle macroscopic normal different from z
        #NOTE: coords in cKDTree must be in [0,L), but pytim uses [-L/2,L/2) on the 3rd axis.
        #We shift them here
        _box=self.universe.coord.dimensions[:]
        _shift=np.array([0.,0.,_box[2]])/2.
        _pos=group.positions[:]+_shift
        self.KDTree=cKDTree(_pos,boxsize=_box[:6],copy_data=True)

    def _NN_query(self,position,qrange):
        return self.KDTree.query_ball_point(position,qrange,n_jobs=-1)

    def _do_cluster_analysis_DBSCAN(self):
        """ Performs a cluster analysis using DBSCAN 

            :returns [labels,counts]: lists of the id of the cluster to which every atom is belonging to, and of the number of elements in each cluster. 

            Uses a slightly modified version of DBSCAN from sklearn.cluster
            that takes periodic boundary conditions into account (through
            cKDTree's boxsize option) and collects also the sizes of all 
            clusters. This is on average O(N log N) thanks to the O(log N) 
            scaling of the kdtree.

        """
        # TODO: extra_cluster_groups are not yet implemented
        min_samples = 2 ;
        _box=self.universe.coord.dimensions[:]
        points = self.itim_group.atoms.positions[:]+np.array([0.,0.,_box[2]])/2.
        tree = cKDTree(points,boxsize=_box[:6])
        neighborhoods = np.array( [ np.array(neighbors) 
                                for neighbors in tree.query_ball_point(points, self.cluster_cut,n_jobs=-1) ] )
        n_neighbors = np.array([len(neighbors)
                                for neighbors in neighborhoods])
        labels = -np.ones(points.shape[0], dtype=np.intp)
        counts = np.zeros(points.shape[0], dtype=np.intp)
        core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, labels, counts)
        return labels, counts


    def _do_cluster_analysis(self):

        # _cluster_map[_aid]        : (atom id)        -> cluster id  | tells to which cluster atom _aid belongs
        # _cluster_analyzed[_aid]   : (atom id)        -> true/false  | tells wether _aid has been associated to a cluster
        # _cluster_size[_clusterid] : (cluster id)     -> size        | tells how many atoms belong to cluster _clusterid
        # _cluster_index[_nn_id]    : (NN id)          -> atom id     | tells you the id of the atom in the cluster being currently analyzed. Does not need to be initialized
        self.cluster_mask = [[] for _ in self.cluster_groups]
        _box=self.universe.coord.dimensions[:]

        for _gid,_g in enumerate(self.cluster_groups):

            self._init_NN_search(_g)

            self.cluster_mask[_gid] = np.ones(_g.n_atoms, dtype=np.int8) * -1

            _cluster_analyzed   =np.zeros(_g.n_atoms,dtype=np.bool )
            _cluster_map        =np.zeros(_g.n_atoms,dtype=np.intc)
            _cluster_size       =np.zeros(_g.n_atoms,dtype=np.intc)
            _cluster_index      =np.zeros(_g.n_atoms,dtype=np.intc)
            _nanalyzed = 1
            _clusterid = 0
            _nn_id = 0
            _current_max_size=0
            for _aid, _atom  in enumerate(_g) :
                if (_cluster_analyzed[_aid] == False) :
                    _cluster_analyzed[_aid] = True
                    _cluster_map[_aid] = _clusterid
                    _cluster_size[_clusterid]+=1
                    _cluster_index[_nn_id] = _aid
                    _nn_id+=1
                    while _nn_id >= _nanalyzed:
                        _aid2 = _cluster_index[_nanalyzed-1]
                        _shift=np.array([0.,0.,_box[2]])/2.
                        _neighbors_id = self._NN_query(_g.atoms[_aid2].position+_shift,self.cluster_cut[_gid])
                        # Alternative fact: the commented version goes slower
                        #_not_analyzed = np.array(_neighbors_id)[np.invert(_cluster_analyzed[_neighbors_id])]
                        #_cluster_analyzed[_not_analyzed]=True
                        #_cluster_map[_not_analyzed] = _clusterid
                        #_cluster_size[_clusterid] += len(_not_analyzed)
                        #_cluster_index[range(_nn_id,_nn_id+len(_not_analyzed))]=_not_analyzed
                        #_nn_id+=len(_not_analyzed)
                        for _nid in _neighbors_id:
                            if (_cluster_analyzed[_nid] == False) :
                                _cluster_analyzed[_nid]=True
                                _cluster_map[_nid]=_clusterid
                                _cluster_size[_clusterid]+=1
                                _cluster_index[_nn_id]=_nid
                                _nn_id+=1
                        _nanalyzed+=1
                    _clusterid+=1

            _cluster_id_largest = np.argmax(_cluster_size)
            # All atoms in the largest cluster have mask==0, those in the other clusters have mask==-1
            self.cluster_mask[_gid][(_cluster_map==_cluster_id_largest)] = 0
        # TODO implement inclusiveness
            assert np.max(_cluster_size)>1 , self.CLUSTER_FAILURE
        return self.cluster_groups[0][self.cluster_mask[0]==0]


    def assign_layers(self):
        """ Determine the ITIM layers.


        """
        self._assign_mesh()
        delta = self.delta ;
        mesh_dx = self.mesh_dx ; mesh_dy = self.mesh_dy
        up=0 ; low=1
        self.layers_ids=[[],[]] ;# upper, lower
        self.mask=np.zeros((2,self.max_layers,self.mesh_nx*self.mesh_ny),
                            dtype=int);

        utilities.rebox(self.universe)
        if(self.cluster_cut is not None): # groups have been checked already in _sanity_checks()
            labels,counts = self._do_cluster_analysis_DBSCAN()
            labels = np.array(labels)
            label_max = np.argmax(counts) # the label of atoms in the largest cluster
            ids_max   = np.where(labels==label_max)[0]  # the indices (within the group) of the 
                                                        # atoms belonging to the largest cluster
            self.cluster_group = self.itim_group[ids_max]
            
        else:
            self.cluster_group=self.itim_group ;

        utilities.center(self.universe,self.cluster_group)
               
        # first we label all atoms in itim_group to be in the gas phase 
        self.itim_group.atoms.bfactors = 0.5
        # then all atoms in the larges group are labelled as liquid-like
        self.cluster_group.atoms.bfactors = 0

        _radius=self.cluster_group.radii
        self._seen=[[],[]]
        self._seen[up] =np.zeros(len(utilities.get_x(self.cluster_group)),dtype=np.int8)
        self._seen[low]=np.zeros(len(utilities.get_x(self.cluster_group)),dtype=np.int8)

        _x=utilities.get_x(self.cluster_group)
        _y=utilities.get_y(self.cluster_group)
        _z=utilities.get_z(self.cluster_group)

        self._layers=[[],[]]
        sort = np.argsort( _z + _radius * np.sign(_z) )
        # NOTE: np.argsort returns the sorted *indices*

        if self.use_multiproc:
            # so far, it justs exploit a simple scheme splitting
            # the calculation between the two sides. Would it be
            # possible to implement easily 2d domain decomposition?
            proc=[[],[]]
            queue = [ Queue() , Queue() ] 
            proc[up]  = Process(target=self._assign_one_side,
                                args=(up,sort[::-1],_x,_y,_z,_radius,queue[up]))
            proc[low] = Process(target=self._assign_one_side,
                                args=(low,sort[::] ,_x,_y,_z,_radius,queue[low]))

            for p in proc: p.start()
            for uplow  in [up,low]:
                self._layers[uplow] = queue[uplow].get()
            for p in proc: p.join()
        else:
            self._layers[up]   = self._assign_one_side(up,sort[::-1],_x,_y,_z,_radius)
            self._layers[low] = self._assign_one_side(low,sort,_x,_y,_z,_radius)


        # assign bfactors to all layers
        for uplow in [up,low]:
            for _nlayer,_layer in enumerate(self._layers[uplow]):
                _layer.bfactors = _nlayer+1 

        # reset the interpolator        
        self._interpolator=None

    def _generate_periodic_border_2D(self, positions):
        _box = self.universe.coord.dimensions[:3]

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
        upper = self._layers[0][layer-1]
        lower = self._layers[1][layer-1]

        upperpos = self._generate_periodic_border_2D(upper.positions)
        lowerpos = self._generate_periodic_border_2D(lower.positions)

        self.surface_triangulation = [None,None] 
        self.triangulation_points= [None,None] 
        self.surface_triangulation[0] = Delaunay(upperpos[:,0:2]) 
        self.surface_triangulation[1] = Delaunay(lowerpos[:,0:2]) 
        self.triangulation_points[0] = upperpos[:]
        self.triangulation_points[1] = lowerpos[:]
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



if __name__ == "__main__":
    import argparse
    from matplotlib import pyplot as plt
    from observables import *
    parser = argparse.ArgumentParser(description='Description...')
    #TODO add series of pdb/gro/...
    parser.add_argument('--top'                                       )
    parser.add_argument('--trj'                                       )
    parser.add_argument('--info'     , action  = 'store_true'         )
    parser.add_argument('--alpha'    , type = float , default = 1.0   )
    parser.add_argument('--selection', default = 'all'                )
    parser.add_argument('--layers'   , type = int   , default = 1     )
    parser.add_argument('--dump'   ,default = False
                                                     ,help="Output to pdb trajectory")
    # TODO: add noncontiguous sampling
    args = parser.parse_args()

    u=None
    if args.top is None and args.trj is None:
        parser.print_help()
        exit()
    else:
        try:
            u = Universe(args.top,args.trj)
        except:
            pass

        if args.top is None:
            u = Universe(args.trj)
        if args.trj is None:
            u = Universe(args.top)

    if u is None:
        print "Error loadinig input files",exit()

    g = u.select_atoms(args.selection)

    interface = ITIM(u,
                info=args.info,
                alpha=args.alpha,
                max_layers = args.layers,
                itim_group = g
                )
    rdf=None
    rdf2=None
    orientation=MolecularOrientation(u)
    orientation2=MolecularOrientation(u,options='normal')

    all1 = u.select_atoms("name OW")
    all2 = u.select_atoms("name OW")
    for frames, ts in enumerate(u.trajectory[::50]) :
        print "Analyzing frame",ts.frame+1,\
              "(out of ",len(u.trajectory),") @ ",ts.time,"ps"
        interface.assign_layers()
        g1=interface.layers('upper',1)
        g2=interface.layers('upper',1)
        if(args.dump):
            interface.writepdb()

        tmp = InterRDF(all1,all2,range=(0.,ts.dimensions[0]/2.),function=orientation.compute)
#        tmp = InterRDF2D(g1,g2,range=(0.,ts.dimensions[0]/2.))
        tmp.sample(ts)
        tmp.normalize()

#        tmp2 = InterRDF2D(g1,g2,range=(0.,ts.dimensions[0]/2.),function=orientation2.compute)
#        tmp2.sample(ts)
#        tmp2.normalize()


        if rdf  is None:
            rdf  =tmp.rdf
#            rdf2 =tmp2.rdf
        else:
            rdf +=tmp.rdf
#            rdf2+=tmp2.rdf

    np.savetxt('angle3d.dat',np.column_stack((tmp.bins,rdf/(frames+1))))




#
