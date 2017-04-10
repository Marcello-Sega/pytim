# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis
from timeit import default_timer as timer
import numpy as np
from scipy.spatial import *
from dbscan import dbscan_inner
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.cluster import vq
from MDAnalysis.topology import tables

def lap(show=False):
    if not hasattr(lap, "tic"):
        lap.tic=timer()
    else:
        toc=timer()
        dt = toc-lap.tic
        lap.tic=toc
        if show:
            print("LAP >>> "+str(dt))
        return dt

def get_box(universe,normal=2):
    box=universe.coord.dimensions[0:3]
    if normal==2:
        return box
    if normal==0:
        return np.roll(box,2)
    if normal==1:
        return np.roll(box,1)

def get_x(group=None,normal=2):
    return group.positions[:, (0+1+normal)%3]

def get_y(group=None,normal=2):
    return group.positions[:, (1+1+normal)%3]

def get_z(group=None,normal=2):
    return group.positions[:, (2+1+normal)%3]

def get_pos(group=None,normal=2):
    pos = group.positions[:]
    if normal==2:
        return pos
    if normal==0:
        return np.roll(pos,2,axis=1)
    if normal==1:
        return np.roll(pos,1,axis=1)

def centerbox(universe,x=None,y=None,z=None,vector=None,center_direction=2,halfbox_shift=True):
    # in ITIM, the system is always centered at 0 along the normal direction (halfbox_shift==True)
    # To center to the middle of the box along all directions, set halfbox_shift=False
    dim = universe.coord.dimensions
    stack=False
    dirdict = {'x':0,'y':1,'z':2}
    if center_direction in dirdict:
        center_direction = dirdict[center_direction]
    assert center_direction in [0,1,2], "Wrong direction supplied to centerbox"

    shift=np.array([0.,0.,0.])
    if halfbox_shift==True:
        shift[center_direction]=dim[center_direction]/2.
    # we rebox the atoms in universe, and not a vector
    if x is None and y is None and z is None and vector is None:
        stack=True ;
        x=get_x(universe.atoms)
        y=get_y(universe.atoms)
        z=get_z(universe.atoms)
    if x is None and y is None and z is None and vector is not None:
         try:
            vector[vector>= dim[center_direction]-shift[center_direction]]-=dim[center_direction]
            vector[vector< -dim[center_direction]-shift[center_direction]]+=dim[center_direction]
         except:
            pass
    if x is not None or  y is not None or z is not None :
        for index, val in enumerate((x,y,z)):
            try:
                # let's just try to rebox all directions. Will succeed only for those which are not None
                # the >= convention is needed for cKDTree
                val[val>= dim[index]-shift[index]]-=dim[index]
                val[val< -dim[index]-shift[index]]+=dim[index]
            except:
                pass
    if stack:
        universe.coord.positions=np.column_stack((x,y,z))

def guess_normal(universe, group):
    """
    Guess the normal of a liquid slab

    """
    universe.atoms.pack_into_box()
    dim = universe.coord.dimensions

    delta = []
    for direction in range(0,3):
        histo,edges=np.histogram(group.positions[:,direction], bins=5,
                                 range=(0,dim[direction]),
                                 density=True) ;
        max_val=np.amax(histo)
        min_val=np.amin(histo)
        delta.append( np.sqrt((max_val-min_val)**2 ))
    return np.argmax(delta)

def trim_triangulated_surface(tri,box):
    """ Reduce a surface triangulation that has been extended to allow for periodic boundary conditions
        to the primary cell.

        The simplices within the primary cell are those with at least two vertices within the cell boundaries.

        :param Delaunay tri: a 2D triangulation
        :param ndarray  box: box cell parameters
        :returns ndarray simplices: the simplices within the primary cell.
    """
    return tri.simplices[np.where((np.logical_and(tri.points[tri.simplices]<=box[0:2],
                                                  tri.points[tri.simplices]>[0,0]).sum(axis=2)>=2).sum(axis=1)>=2)]

def triangulated_surface_stats(tri2d,points3d):
    """ Return basic statistics about a surface triangulation

        Implemented statistics are: surface area

        :param tri2d             : indices of triangles vertices
        :param ndarray  points3d : the heigth of each vertex along the third dimension
        :returns list stats      : the statistics :  [surface_area]
    """

    # TODO: write a more efficient routine ?
    # some advanced indexing here...
    # points3d[reduced] is an array of shape (x,3,3)
    # we need to subtract the first of the three vectors
    # from each of them. This uses numpy.newaxis
    v = points3d[tri2d]-points3d[tri2d][:,0][:,None]
    # then we need to make the cross product of the two
    # non-zero vectors of each triplet
    area = np.linalg.norm(np.cross(v[:,1],v[:,2]),axis=1).sum()/2.
    return [area]


def _init_NN_search(group,box):
    #NOTE: boxsize shape must be (6,), and the last three elements are overwritten in cKDTree:
    #   boxsize_arr = np.empty(2 * self.m, dtype=np.float64)
    #   boxsize_arr[:] = boxsize
    #   boxsize_arr[self.m:] = 0.5 * boxsize_arr[:self.m]

    # TODO: handle macroscopic normal different from z
    #NOTE: coords in cKDTree must be in [0,L), but pytim uses [-L/2,L/2) on the 3rd axis.
    #We shift them here
    shift=np.array([0.,0.,box[2]])/2.
    pos=group.positions[:]+shift
    return cKDTree(pos,boxsize=box[:6],copy_data=True)

def _NN_query(kdtree,position,qrange):
    return kdtree.query_ball_point(position,qrange,n_jobs=-1)

def do_cluster_analysis_DBSCAN(group,cluster_cut,box,threshold_density=None,molecular=True):
    """ Performs a cluster analysis using DBSCAN

        :returns [labels,counts]: lists of the id of the cluster to which every atom is belonging to, and of the number of elements in each cluster.

        Uses a slightly modified version of DBSCAN from sklearn.cluster
        that takes periodic boundary conditions into account (through
        cKDTree's boxsize option) and collects also the sizes of all
        clusters. This is on average O(N log N) thanks to the O(log N)
        scaling of the kdtree.

    """
    if isinstance(threshold_density,type(None)):
        min_samples = 2
    if isinstance(threshold_density,float) or isinstance(threshold_density,int):
        min_samples = threshold_density * 4./3. * np.pi * cluster_cut**3
        if min_samples < 2 :
            min_samples = 2

    # TODO: extra_cluster_groups are not yet implemented
    points = group.atoms.positions[:]

    tree = cKDTree(points,boxsize=box[:6])
    neighborhoods = np.array( [ np.array(neighbors)
                            for neighbors in tree.query_ball_point(points, cluster_cut,n_jobs=-1) ] )
    assert len(neighborhoods.shape) is 1, "Error in do_cluster_analysis_DBSCAN(), the cutoff is probably too small"
    if molecular==False:
        n_neighbors = np.array([len(neighbors)
                                for neighbors in neighborhoods])
    else:
        n_neighbors = np.array([len(np.unique(group[neighbors].resids))
                                    for neighbors in  neighborhoods ])

    if isinstance(threshold_density,str):
        assert threshold_density == 'auto', "Internal error: wrong parameter 'threshold_density' passed to do_cluster_analysis_DBSCAN"
        max_neighbors = np.max(n_neighbors)
        min_neighbors = np.min(n_neighbors)
        avg_neighbors = (min_neighbors + max_neighbors)/2.
        modes = 2
        centroid,_ = vq.kmeans2(n_neighbors*1.0, modes , iter=10, check_finite=False)
        min_samples   = np.mean(centroid)

    labels = -np.ones(points.shape[0], dtype=np.intp)
    counts = np.zeros(points.shape[0], dtype=np.intp)

    core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
    dbscan_inner(core_samples, neighborhoods, labels, counts)
    return labels, counts, n_neighbors


def _do_cluster_analysis(cluster_groups):

    # _cluster_map[_aid]        : (atom id)        -> cluster id  | tells to which cluster atom _aid belongs
    # _cluster_analyzed[_aid]   : (atom id)        -> true/false  | tells wether _aid has been associated to a cluster
    # _cluster_size[_clusterid] : (cluster id)     -> size        | tells how many atoms belong to cluster _clusterid
    # _cluster_index[_nn_id]    : (NN id)          -> atom id     | tells you the id of the atom in the cluster being currently analyzed. Does not need to be initialized
    cluster_mask = [[] for _ in cluster_groups]
    _box=self.universe.coord.dimensions[:]

    for _gid,_g in enumerate(cluster_groups):

        kdtree = _init_NN_search(_g)

        cluster_mask[_gid] = np.ones(_g.n_atoms, dtype=np.int8) * -1

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
                    _neighbors_id = _NN_query(kdtree,_g.atoms[_aid2].position+_shift,self.cluster_cut[_gid])
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
        cluster_mask[_gid][(_cluster_map==_cluster_id_largest)] = 0
    # TODO implement inclusiveness
        assert np.max(_cluster_size)>1 , self.CLUSTER_FAILURE
    return self.cluster_groups[0][self.cluster_mask[0]==0]




