# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

from abc import ABCMeta, abstractmethod
import numpy as np
import MDAnalysis
from MDAnalysis.topology import tables

class PYTIM(object):
    """ The PYTIM metaclass
    """
    __metaclass__ = ABCMeta

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

    def writepdb(self,filename='layers.pdb',multiframe=True):
        """ Write the frame to a pdb file, marking the atoms belonging
            to the layers with different beta factor.
    
            :param filename:   string  -- the output file name
            :param multiframe: boolean -- append to pdb file if True
    
            Example:
    
            >>> writepdb(interface,'layers.pdb',multiframe=False)
        """

        try:
            PDB=MDAnalysis.Writer(filename, multiframe=True, bonds=False,
                            n_atoms=self.universe.atoms.n_atoms)
            PDB.write(self.universe.atoms)
        except:
            print("Error writing pdb file")

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

from pytim.itim import ITIM

__all__ = [ 'itim' , 'gitim' , 'observables', 'datafiles', 'tests', 'utilities']


