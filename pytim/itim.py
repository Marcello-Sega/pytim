#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: pytim
    =============
"""

from timeit import default_timer as timer
#from threading import Thread
from multiprocessing import Process, Queue
import numpy as np
from scipy.spatial import *
from MDAnalysis.core.AtomGroup   import *
from pytim.datafiles import *
from pytim import utilities 
from MDAnalysis.topology import tables



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
        >>> interface = pytim.ITIM(u, alpha=2.0, max_layers=4)
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
        self.UNDEFINED_CLUSTER_SEARCH= "Either both cluster_cut and cluster_groups in %s.%s should be defined, or set to None" % ( (__name__) , (self.__class__.__name__) )
        self.MISMATCH_CLUSTER_SEARCH= "cluster_cut in %s.%s should be either a scalar or an array matching the number of groups" % ( (__name__) , (self.__class__.__name__) )
        self.EMPTY_LAYER="One or more layers are empty"
        self.CLUSTER_FAILURE="Cluster algorithm failed: too small cluster cutoff provided?"
        self.UNDEFINED_LAYER="No layer defined: forgot to call assign_layers() ?"


 
    def __init__(self,universe,mesh=0.4,alpha=2.0,itim_group=None,radii_dict=None,
                 max_layers=1,cluster_groups=None,cluster_cut=None,
                 info=False,multiproc=True):

        self.define_error_messages()
        self.universe=universe
        self.target_mesh=mesh
        self.alpha=alpha
        self.max_layers=max_layers
        self.info=info
        self.all_atoms = self.universe.select_atoms('all')
        self._layers=None

        self.cluster_cut=cluster_cut
        if cluster_cut is not None and not isinstance(cluster_cut, (list, tuple, np.ndarray)):
            if type(cluster_cut) is int or type(cluster_cut) is float:
                self.cluster_cut = np.array([float(cluster_cut)])

        self.cluster_groups=cluster_groups
        if cluster_groups is not None and not isinstance(cluster_groups, (list, tuple, np.ndarray)):
            self.cluster_groups = [cluster_groups]

        if itim_group is None:
            self.itim_group = self.all_atoms
        else:
            self.itim_group = itim_group
        try:
            _groups = self.cluster_groups[:] # deep copy
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
                assert np.all(_g.radii is not None), self.UNDEFINED_RADIUS
                del _radii
                del _types

        self._sanity_checks()
        
        self.grid=None
        self.use_threads=False
        self.use_kdtree=False
        self.use_multiproc=multiproc
        self.tic=timer()

    def _sanity_checks(self):

        # these are done at the beginning to prevent burdening the inner loops
        assert self.alpha > 0,                                           self.ALPHA_NEGATIVE
        assert self.alpha < np.amin(self.universe.dimensions[:3]),       self.ALPHA_LARGE

        assert self.target_mesh > 0,                                     self.MESH_NEGATIVE
        assert self.target_mesh < np.amin(self.universe.dimensions[:3]), self.MESH_LARGE
    
        assert ( (self.cluster_cut is None) and (self.cluster_groups is  None) ) or (  (self.cluster_cut is not  None) and ( self.cluster_groups is not  None) ),self.UNDEFINED_CLUSTER_SEARCH
        if(self.cluster_cut is not None):
            assert len(self.cluster_cut)== 1 or len(self.cluster_cut) == len(self.cluster_groups), self.MISMATCH_CLUSTER_SEARCH

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
        # NOTE: here the kdtree is slower (15%) than bucketing (only 1 line of code though...)
        if (self.use_kdtree==True) : # this is False by default
            return self.meshtree.query_ball_point([_x[atom],_y[atom]],_radius[atom]+self.alpha) 
        else:
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

            return np.ravel_multi_index(np.array([sel_x,sel_y]).astype(int),(self.mesh_nx,self.mesh_ny))

    def _assign_one_side(self,uplow,sorted_atoms,_x,_y,_z,
                        _radius,queue=None):
        for layer in range(0,self.max_layers) :
            mask = self.mask[uplow][layer]
            _inlayer=[]
            for atom in sorted_atoms:
                if self._seen[atom] != 0 :
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
                self._seen[atom]=layer+1 ; # start counting from 1, 0 will be
                                           # unassigned, -1 for gas phase TODO: to be
                                           # implemented
                # 3) let's add the atom id to the list of atoms in this layer
                _inlayer.append(atom)
                if np.sum(mask) == len(mask):
                    self.layers_ids[uplow].append(_inlayer)
                    #NOTE that checking len(mask[mask==0])==0 is slower. 
                    #     np.count_nonzero is comparable. For _submask, 
                    #     np.sum() is slightly slower, instead.
                    break
        if queue != None:
            queue.put(self._seen)
            queue.put(self.layers_ids[uplow])
        assert self.layers_ids[uplow],self.EMPTY_LAYER  # one of the two layers (upper,lower) or both are empty
            

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
        # TODO: modify to accept group of atoms (n_jobs=-1 could be used if more than x atoms supplied)
        return self.KDTree.query_ball_point(position,qrange)

    def _define_layers_groups(self,group):
        _layers=[[],[]]
        for i,uplow in enumerate(self.layers_ids):
            for j,layer in enumerate(uplow):
                _layers[i].append(group.atoms[layer])
        self._layers=np.array(_layers)
        assert np.any(self._layers) != False, self.UNDEFINED_LAYER

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
                if (_cluster_analyzed[_aid] == 0) :
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
            _group=self._do_cluster_analysis()
        else: 
            _group=self.itim_group ; 
        utilities.center(self.universe,_group) 

        _radius=_group.radii
        self._seen=np.zeros(len(utilities.get_x(_group)))

        _x=utilities.get_x(_group)
        _y=utilities.get_y(_group)
        _z=utilities.get_z(_group)

        sort = np.argsort( _z + _radius * np.sign(_z) )
        # NOTE: np.argsort returns the sorted *indices*

        if self.use_multiproc:
            # so far, it justs exploit a simple scheme splitting
            # the calculation between the two sides. Would it be
            # possible to implement easily 2d domain decomposition? 
            proc=[[],[]] ; queue=[[],[]] ; seen=[[],[]]
            queue[up]=Queue()
            proc[up]  = Process(target=self._assign_one_side,
                                args=(up,sort[::-1],_x,_y,_z,_radius,
                                queue[up]))
            queue[low]=Queue()
            proc[low] = Process(target=self._assign_one_side,
                                args=(low,sort,_x,_y,_z,_radius,
                                queue[low]))
            for p in proc: p.start()
            for q in queue: self._seen+=q.get()
            self.layers_ids[low]+=queue[low].get()
            self.layers_ids[up]+=queue[up].get()
            for p in proc: p.join()
        else:
            self._assign_one_side(up,sort[::-1],_x,_y,_z,_radius)
            self._assign_one_side(low,sort,_x,_y,_z,_radius)
        self._define_layers_groups(_group)
        self.itim_group.atoms.bfactors=-1
        _group.atoms.bfactors=self._seen

    def layers(self,side='both',*ids):
        """ Select one or more layers.

        :param str side: 'upper', 'lower' or 'both'
        :param slice ids: the slice corresponding to the layers to be selcted (starting from 0) 

        The slice can be used to select a single layer, or multiple, e.g. (using the example of the :class:`ITIM` class) :

        >>> interface.layers('upper')  # all layers, upper side
        array([<AtomGroup with 406 atoms>, <AtomGroup with 411 atoms>,
               <AtomGroup with 414 atoms>, <AtomGroup with 378 atoms>], dtype=object)

        >>> interface.layers('lower',1)  # first layer, lower side
        <AtomGroup with 406 atoms>

        >>> interface.layers('both',0,3) # 1st - 3rd layer, on both sides
        array([[<AtomGroup with 406 atoms>, <AtomGroup with 411 atoms>,
                <AtomGroup with 414 atoms>],
               [<AtomGroup with 406 atoms>, <AtomGroup with 418 atoms>,
                <AtomGroup with 399 atoms>]], dtype=object)

        >>> interface.layers('lower',0,4,2) # 1st - 4th layer, with a stride of 2, lower side 
        array([<AtomGroup with 406 atoms>, <AtomGroup with 399 atoms>], dtype=object)


        """

        _options={'both':slice(None),'upper':0,'lower':1}
        _side=_options[side]
        if len(ids) == 0:
            _slice = slice(None)
        if len(ids) == 1:
            _slice = slice(ids[0])
        if len(ids) == 2:
            _slice = slice(ids[0],ids[1])
        if len(ids) == 3:
            _slice = slice(ids[0],ids[1],ids[2])

        if len(ids) == 1 and side != 'both':
            return self._layers[_side,_slice][0]

        if len(ids) == 1 :
            return self._layers[_side,_slice][:,0]

        return self._layers[_side,_slice]



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
