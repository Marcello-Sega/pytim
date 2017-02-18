# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from timeit import default_timer as timer
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
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
     


