# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis
from timeit import default_timer as timer
import numpy as np
from scipy.spatial import *
from dbscan import dbscan_inner
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
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

def get_x(group=None):
    return group.positions[:,0]

def get_y(group=None):
    return group.positions[:,1]

def get_z(group=None):
    return group.positions[:,2]

def rebox(universe,x=None,y=None,z=None):
    # TODO check that the correct frame-dependent box is used !! 
    dim = universe.coord.dimensions
    stack=False
    shift=np.array([0,0,dim[2]/2.])

    # we rebox the atoms in universe, and not a vector
    if x is None and y is None and z is None:
        stack=True ; 
        x=get_x(universe.atoms)
        y=get_y(universe.atoms)
        z=get_z(universe.atoms)
    for index, val in enumerate((x,y,z)):
        try:
            # the >= convention is needed for cKDTree
            val[val>= dim[index]-shift[index]]-=dim[index]
            val[val< 0        -shift[index]]+=dim[index]
        except:
            pass
    if stack:
        universe.coord.positions=np.column_stack((x,y,z))

def center(universe, group):
    """ 
    Centers the liquid slab in the simulation box.

    The algorithm tries to avoid problems with the definition
    of the center of mass. First, a rough density profile
    (10 bins) is computed. Then, the support group is shifted
    and reboxed until the bins at the box boundaries have a
    density lower than a threshold delta

    
    """
    #TODO: implement shifting back for final coordinate writing as an option
    dim = universe.coord.dimensions
    shift=dim[2]/100. ;# TODO, what about non ortho boxes?
    total_shift=0
    rebox(universe)
    #self._liquid_mask=np.zeros(len(self.itim_group), dtype=np.int8) 
    _z_group = get_z(group)
    _x = get_x(universe.atoms)
    _y = get_y(universe.atoms)
    _z = get_z(universe.atoms)

    histo,edges=np.histogram(_z_group, bins=10,
                             range=(-dim[2]/2.,dim[2]/2.), density=True) ;
        #TODO handle norm!=z
    max=np.amax(histo)
    min=np.amin(histo)
    delta=min+(max-min)/3. ;# TODO test different cases

    # let's first avoid crossing pbc with the liquid phase. This can fail:
    # TODO handle failure
    while(histo[0]>delta or histo[-1]> delta):
        total_shift+=shift
        _z_group +=shift
        rebox(universe,z=_z_group)
        histo,edges=np.histogram(_z_group, bins=10,
                                 range=(-dim[2]/2.,dim[2]/2.), density=True);
    #TODO: clean up
    _center=np.average(_z_group)

    _z += total_shift - _center
    # finally, we copy everything back
    universe.coord.positions=np.column_stack((_x,_y,_z))
 
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

def do_cluster_analysis_DBSCAN(group,cluster_cut,box):
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
    points = group.atoms.positions[:]+np.array([0.,0.,box[2]])/2.
    tree = cKDTree(points,boxsize=box[:6])
    neighborhoods = np.array( [ np.array(neighbors) 
                            for neighbors in tree.query_ball_point(points, cluster_cut,n_jobs=-1) ] )
    n_neighbors = np.array([len(neighbors)
                            for neighbors in neighborhoods])
    labels = -np.ones(points.shape[0], dtype=np.intp)
    counts = np.zeros(points.shape[0], dtype=np.intp)
    core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
    dbscan_inner(core_samples, neighborhoods, labels, counts)
    return labels, counts


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




