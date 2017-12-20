# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import MDAnalysis
from MDAnalysis import Universe
from MDAnalysis.topology import tables
import __builtin__
import importlib
from version import __version__
from pytim.patches import PatchTrajectory, PatchOpenMM, PatchMDTRAJ
from pytim.properties import _create_property
from pytim.sanity_check import SanityCheck
from . import messages
from pytim.pdb import _writepdb


class PYTIM(object):
    """ The PYTIM metaclass
    """
    __metaclass__ = ABCMeta

    directions_dict = {0: 'x', 1: 'y', 2: 'z',
                       'x': 'x', 'y': 'y', 'z': 'z',
                       'X': 'x', 'Y': 'y', 'Z:': 'z'}
    symmetry_dict = {'cylindrical': 'cylindrical',
                     'spherical': 'spherical',
                     'planar': 'planar'}

    # main properties shared by all implementations of the class
    # When required=True is passed, the implementation of the class *must*
    # override the method when instantiating the class (i.e., before __init__)
    # By default required=False, and the name is set to None

    # interface *must* be created first.
    alpha, _alpha =\
        _create_property('alpha', "(float) real space cutoff")
    layers, _layers =\
        _create_property('layers', "AtomGroups of atoms in layers")
    itim_group, _itim_group =\
        _create_property('itim_group', "(AtomGroup) the group,"
                         "the surface of which should be computed")
    cluster_cut, _cluster_cut =\
        _create_property('cluster_cut', "(real) cutoff for phase"
                         "identification")
    molecular, _molecular =\
        _create_property('molecular', "(bool) wheter to compute"
                         "surface atoms or surface molecules")
    surfaces, _surfaces =\
        _create_property('surfaces', "Surfaces associated to the interface",
                         readonly=True)
    info, _info =\
        _create_property('info', "(bool) print additional information")

    multiproc, _multiproc =\
        _create_property('multiproc', "(bool) use parallel implementation")

    extra_cluster_groups, _extra_cluster_groups =\
        _create_property('extra_cluster_groups',
                         "(ndarray) additional cluster groups")
    radii_dict, _radii_dict =\
        _create_property('radii_dict', "(dict) custom atomic radii")

    max_layers, _max_layers =\
        _create_property('max_layers',
                         "(int) maximum number of layers to be identified")
    cluster_threshold_density, _cluster_threshold_density =\
        _create_property('cluster_threshold_density',
                         "(float) threshold for the density-based filtering")
    # TODO: does this belong here ?
    _interpolator, __interpolator =\
        _create_property('_interpolator', "(dict) custom atomic radii")

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _assign_layers(self):
        pass

    @property
    def atoms(self):
        return self._layers[:].sum()

    @property
    def method(self):
        return self.__class__.__name__

    def label_planar_sides(self):
        # Assign to all layers a label (tempfactor) that can be used in pdb files.
        # Additionally, set the new layers and sides
        for uplow in [0, 1]:
            for nlayer, layer in enumerate(self._layers[uplow]):
                self.label_group(layer, beta=nlayer + 1.0,
                                 layer=nlayer + 1, side=uplow)

    def label_group(self, group, beta=None, layer=None, cluster=None, side=None):
        if group is None:
            raise RuntimeError(
                'one of the groups, possibly a layer one, is None. Something is wrong...')
        if self.molecular == True:
            _group = group.residues.atoms
        else:
            _group = group

        if beta is not None:
            _group.tempfactors = float(beta)
        if layer is not None:
            _group.layers = layer
        if side is not None:
            _group.sides = side
        if cluster is not None:
            _group.clusters = cluster

    def _assign_symmetry(self, symmetry):
        if self.itim_group is None:
            raise TypeError(messages.UNDEFINED_ITIM_GROUP)
        if symmetry == 'guess':
            raise ValueError("symmetry 'guess' To be implemented")
        else:
            if not (symmetry in self.symmetry_dict):
                raise ValueError(messages.WRONG_DIRECTION)
            self.symmetry = symmetry

    def _define_cluster_group(self):
        if(self.cluster_cut is not None):
            # groups have been checked already in _sanity_checks()
            labels, counts, neighbors = utilities.do_cluster_analysis_DBSCAN(
                self.itim_group, self.cluster_cut[0],
                self.universe.dimensions[:6],
                self.cluster_threshold_density, self.molecular)
            labels = np.array(labels)
            # the label of atoms in the largest cluster
            label_max = np.argmax(counts)
            # the indices (within the group) of the
            ids_max = np.where(labels == label_max)[0]

            # atoms belonging to the largest cluster
            if (self.extra_cluster_groups is not None):
                extra = np.sum(self.extra_cluster_groups[:])
                self.extra = extra
                x_labels, x_counts, x_neighbors = utilities.do_cluster_analysis_DBSCAN(
                    extra, self.cluster_cut[0],
                    self.universe.dimensions[:6],
                    self.cluster_threshold_density, self.molecular)
                x_labels = np.array(x_labels)
                x_label_max = np.argmax(x_counts)
                x_ids_other = np.where(x_labels != x_label_max)[0]

                # we mark them initially as non-main-cluster, some will be
                # overwritten
                self.label_group(extra, cluster=1)
                self.cluster_group = np.sum(
                    [self.itim_group[ids_max], extra[x_ids_other]])
            else:
                self.cluster_group = self.itim_group[ids_max]
                self.n_neighbors = neighbors
        else:
            self.cluster_group = self.itim_group

        self.label_group(self.itim_group, cluster=1)
        self.label_group(self.cluster_group, cluster=0)

    def _center(self, group, direction=None, halfbox_shift=True):
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
        total_shift = 0

        if not (direction in self.directions_dict):
            raise ValueError(messages.WRONG_DIRECTION)
        _dir = self.directions_dict[direction]
        _xyz = {'x': (0, utilities.get_x),
                'y': (1, utilities.get_y),
                'z': (2, utilities.get_z)}

        if _dir in _xyz.keys():
            direction = _xyz[_dir][0]
            _pos_group = (_xyz[_dir][1])(group)

        shift = dim[direction] / 100.

        _x = utilities.get_x(group.universe.atoms)
        _y = utilities.get_y(group.universe.atoms)
        _z = utilities.get_z(group.universe.atoms)

        _range = (0., dim[direction])
        if(halfbox_shift == True):
            _range = (-dim[direction] / 2., dim[direction] / 2.)

        histo, _ = np.histogram(_pos_group, bins=10, range=_range,
                                density=True)

        max_val, min_val = np.amax(histo), np.amin(histo)
        # NOTE maybe allow user to set different values
        delta = min_val + (max_val - min_val) / 3.

        # let's first avoid crossing pbc with the liquid phase. This can fail:
        while(histo[0] > delta or histo[-1] > delta):
            total_shift += shift
            if total_shift >= dim[direction]:
                raise ValueError(messages.CENTERING_FAILURE)
            _pos_group += shift
            if _dir == 'x':
                utilities.centerbox(group.universe, x=_pos_group,
                                    center_direction=direction,
                                    halfbox_shift=halfbox_shift)
            if _dir == 'y':
                utilities.centerbox(group.universe, y=_pos_group,
                                    center_direction=direction,
                                    halfbox_shift=halfbox_shift)
            if _dir == 'z':
                utilities.centerbox(group.universe, z=_pos_group,
                                    center_direction=direction,
                                    halfbox_shift=halfbox_shift)
            histo, _ = np.histogram(_pos_group, bins=10, range=_range,
                                    density=True)

        _center = np.average(_pos_group)
        if(halfbox_shift == False):
            box_half = dim[direction] / 2.
        else:
            box_half = 0.
        _pos = {'x': _x, 'y': _y, 'z': _z}
        if _dir in _pos.keys():
            _pos[_dir] += total_shift - _center + box_half
        # finally, we copy everything back
        group.universe.atoms.positions = np.column_stack((_x, _y, _z))

    def _shift_positions_to_middle(self):
        box = self.universe.dimensions[self.normal]
        translation = [0, 0, 0]
        translation[self.normal] = box / 2.
        self.universe.atoms.positions += np.array(translation)
        self.universe.atoms.pack_into_box(self.universe.dimensions[:3])

    def center(self, planar_to_origin=False):
        if self.symmetry == 'planar':
            utilities.centerbox(self.universe, center_direction=self.normal)
            self._center(self.cluster_group, self.normal)
            utilities.centerbox(self.universe, center_direction=self.normal)
            if planar_to_origin == False:
                self._shift_positions_to_middle()
            self.centered_positions = np.copy(self.universe.atoms.positions[:])

        if self.symmetry == 'spherical':
            for xyz in ['x', 'y', 'z']:
                self._center(self.cluster_group, xyz, halfbox_shift=False)
            self.universe.atoms.pack_into_box(self.universe.dimensions[:3])
            self.centered_positions = np.copy(self.universe.atoms.positions[:])

    def writepdb(self, filename='layers.pdb', centered='no', group='all', multiframe=True):
        _writepdb(self, filename=filename, centered=centered,
                  group=group, multiframe=multiframe)


from pytim.itim import ITIM
from pytim.gitim import GITIM
from pytim.willard_chandler import WillardChandler
from pytim.chacon_tarazona import ChaconTarazona
from pytim import observables, utilities, datafiles
