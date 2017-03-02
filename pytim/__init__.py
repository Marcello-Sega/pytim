# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

from   abc import ABCMeta, abstractmethod
import numpy as np
import MDAnalysis
from   MDAnalysis.topology import tables

class PYTIM(object):
    """ The PYTIM metaclass
    """
    __metaclass__ = ABCMeta

    directions_dict={0:'x',1:'y',2:'z','x':'x','y':'y','z':'z','X':'x','Y':'y','Z:':'z'}
    symmetry_dict={'cylindrical':'cylindrical','spherical':'spherical'}

    ALPHA_NEGATIVE = "parameter alpha must be positive"
    ALPHA_LARGE= "parameter alpha must be smaller than the smaller box side"
    MESH_NEGATIVE = "parameter mesh must be positive"
    MESH_LARGE= "parameter mesh must be smaller than the smaller box side"
    UNDEFINED_RADIUS= "one or more atoms do not have a corresponding radius in the default or provided dictionary"
    UNDEFINED_CLUSTER_SEARCH= "If extra_cluster_groups is defined, a cluster_cut should e provided"
    MISMATCH_CLUSTER_SEARCH= "cluster_cut should be either a scalar or an array matching the number of groups (including itim_group)"
    EMPTY_LAYER="One or more layers are empty"
    CLUSTER_FAILURE="Cluster algorithm failed: too small cluster cutoff provided?"
    UNDEFINED_LAYER="No layer defined: forgot to call assign_layers() or not enough layers requested"
    WRONG_UNIVERSE="Wrong Universe passed to ITIM class"
    UNDEFINED_ITIM_GROUP="No itim_group defined"
    WRONG_DIRECTION="Wrong direction supplied. Use 'x','y','z' , 'X', 'Y', 'Z' or 0, 1, 2"
    CENTERING_FAILURE="Cannot center the group in the box. Wrong direction supplied?"

    def writepdb(self,filename='layers.pdb',centered=True,multiframe=True):
        """ Write the frame to a pdb file, marking the atoms belonging
            to the layers with different beta factor.

            :param filename:   string  -- the output file name
            :param multiframe: boolean -- append to pdb file if True

            Example: save the positions (centering the interface in the cell) without appending

            >>> interface.writepdb('layers.pdb',multiframe=False)

            Example: save the positions without centering the interface. This will
                     leave the atoms in the original position with respect to the cell.
                     The :multiframe: option set to :False: will overwrite the file.

            >>> interface.writepdb('layers.pdb',centered=False)

        """

        try:
            if centered==False:
                translation = self.reference_position - self.universe.atoms[0].position[:]
                self.universe.atoms.translate(translation)
            PDB=MDAnalysis.Writer(filename, multiframe=True, bonds=False,
                            n_atoms=self.universe.atoms.n_atoms)
            PDB.write(self.universe.atoms)
            if centered==False:
                self.universe.atoms.translate(-translation)
        except:
            print("Error writing pdb file")

    def savepdb(self,filename='layers.pdb',centered=True,multiframe=True):
        self.writepdb(filename,centered,multiframe)

    def assign_radii(self,radii_dict):
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

    def center(self, group, direction=None, halfbox_shift=True):
        """
        Centers the liquid slab in the simulation box.

        The algorithm tries to avoid problems with the definition
        of the center of mass. First, a rough density profile
        (10 bins) is computed. Then, the group is shifted
        and reboxed until the bins at the box boundaries have a
        density lower than a threshold delta


        In ITIM, the system along the normal direction is always
        centered at 0 (halfbox_shift==True). To center to the middle
        of the box along all directions, set halfbox_shift=False

        """
        if direction is None:
            direction = self.normal
        dim = group.universe.coord.dimensions
        total_shift=0

        assert direction in self.directions_dict, self.WRONG_DIRECTION
        _dir = self.directions_dict[direction]
        if _dir == 'x':
            direction = 0
            _pos_group =  utilities.get_x(group)
        if _dir == 'y':
            direction = 1
            _pos_group =  utilities.get_y(group)
        if _dir == 'z':
            direction = 2
            _pos_group =  utilities.get_z(group)

        shift=dim[direction]/100. ;


        _x = utilities.get_x(group.universe.atoms)
        _y = utilities.get_y(group.universe.atoms)
        _z = utilities.get_z(group.universe.atoms)

        if(halfbox_shift==True):
            _range=(-dim[direction]/2.,dim[direction]/2.)
        else:
            _range=(0.,dim[direction])

        histo,edges=np.histogram(_pos_group, bins=10, range=_range, density=True)

        max=np.amax(histo)
        min=np.amin(histo)
        delta=min+(max-min)/3. ;# TODO test different cases

        # let's first avoid crossing pbc with the liquid phase. This can fail:
        while(histo[0]>delta or histo[-1]> delta):
            total_shift+=shift
            assert total_shift<dim[direction], self.CENTERING_FAILURE
            _pos_group +=shift
            if _dir == 'x':
                utilities.centerbox(group.universe,x=_pos_group,center_direction=direction,halfbox_shift=halfbox_shift)
            if _dir == 'y':
                utilities.centerbox(group.universe,y=_pos_group,center_direction=direction,halfbox_shift=halfbox_shift)
            if _dir == 'z':
                utilities.centerbox(group.universe,z=_pos_group,center_direction=direction,halfbox_shift=halfbox_shift)


            histo,edges=np.histogram(_pos_group, bins=10, range=_range, density=True)

        #TODO: clean up
        _center=np.average(_pos_group)

        if(halfbox_shift==False):
            box_half = dim[direction]/2.
        else:
            box_half = 0.
        if _dir == 'x':
            _x += total_shift - _center + box_half
        if _dir == 'y':
            _y += total_shift - _center + box_half
        if _dir == 'z':
            _z += total_shift - _center + box_half
        # finally, we copy everything back
        group.universe.coord.positions=np.column_stack((_x,_y,_z))


    @abstractmethod
    def assign_layers(self):
        pass

    @abstractmethod
    def triangulate_layer(self):
        pass

    @abstractmethod
    def interpolate_surface(self):
        pass

    @abstractmethod
    def layers(self):
        pass

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


from pytim.itim import  ITIM
from pytim.gitim import GITIM

#__all__ = [ 'itim' , 'gitim' , 'observables', 'datafiles', 'utilities']


