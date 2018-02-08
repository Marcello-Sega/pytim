# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from abc import ABCMeta, abstractmethod
import numpy as np
from properties import _create_property
from pdb import _writepdb
import messages
import utilities


class Interface(object):
    """ The Interface metaclass. Classes for interfacial determination 
	(ITIM, GITIM,...) are derived from this one
    """
    __metaclass__ = ABCMeta

    directions_dict = {
        0: 'x',
        1: 'y',
        2: 'z',
        'x': 'x',
        'y': 'y',
        'z': 'z',
        'X': 'x',
        'Y': 'y',
        'Z:': 'z'
    }
    symmetry_dict = {
        'generic': 'generic',
        'cylindrical': 'cylindrical',
        'spherical': 'spherical',
        'planar': 'planar'
    }

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
        _create_property('itim_group', "(AtomGroup) the group, "
                         "the surface of which should be computed")
    cluster_cut, _cluster_cut =\
        _create_property('cluster_cut', "(real) cutoff for phase "
                         "identification")
    molecular, _molecular =\
        _create_property('molecular', "(bool) whether to compute "
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
    autoassign , _autoassign=\
        _create_property('autoassign',
                         "(bool) assign layers every time a frame changes")
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
        """ Assign to all layers a label (the beta tempfactor)
            that can be used in pdb files. Additionally, set
            the new layers and sides.
        """
        for uplow in [0, 1]:
            for nlayer, layer in enumerate(self._layers[uplow]):
                self.label_group(
                    layer, beta=nlayer + 1.0, layer=nlayer + 1, side=uplow)

    def label_group(self,
                    group,
                    beta=None,
                    layer=None,
                    cluster=None,
                    side=None):
        if group is None:
            raise RuntimeError(
                'one of the groups, possibly a layer one, is None.' +
                ' Something is wrong...')
        if len(group) == 0:
            return
        if self.molecular is True:
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
        self.cluster_group = self.universe.atoms[:0] # empty
        if (self.cluster_cut is not None):
            # we start by adding the atoms in the smaller clusters
            # of the opposit phase, if extra_cluster_groups are provided
            if (self.extra_cluster_groups is not None):
                for extra in self.extra_cluster_groups:
                    x_labels, x_counts, _ = utilities.do_cluster_analysis_dbscan(
                        extra, self.cluster_cut[0], self.cluster_threshold_density,
                        self.molecular)
                    x_labels = np.array(x_labels)
                    x_label_max = np.argmax(x_counts)
                    x_ids_other = np.where(x_labels != x_label_max)[0]

                    # we mark them initially as non-main-cluster, some will be
                    # overwritten
                    self.label_group(extra, cluster=1)
                    self.cluster_group += extra[x_ids_other]
                    # TODO (bug) self.n_neighbors in this branching will be wrong
                    # TODO implement different cutoff radii for different extra groups
            else:
                self.n_neighbors = neighbors

            # next, we add the atoms belonging to the main phase
            self.cluster_group += self.itim_group

            # groups have been checked already in _sanity_checks()
            # self.cluster_group at this stage is composed of itim_group +
            # the smaller clusters of the other phase
            labels, counts, neighbors = utilities.do_cluster_analysis_dbscan(
                self.cluster_group, self.cluster_cut[0],
                self.cluster_threshold_density, self.molecular)
            labels = np.array(labels)
            # the label of atoms in the largest cluster
            label_max = np.argmax(counts)
            # the indices (within the group) of the largest cluster
            ids_max = np.where(labels == label_max)[0]
            self.cluster_group = self.cluster_group[ids_max]


        else:
            self.cluster_group = self.itim_group

        self.label_group(self.itim_group, cluster=1)
        self.label_group(self.cluster_group, cluster=0)

    def reset_labels(self):
        self.label_group(
            self.universe.atoms, beta=0.0, layer=-1, cluster=-1, side=-1)

    @staticmethod
    def _center(group, direction, halfbox_shift=False):
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

        dim = group.universe.coord.dimensions
        total_shift = 0

        if not (direction in Interface.directions_dict):
            raise ValueError(messages.WRONG_DIRECTION)
        _dir = Interface.directions_dict[direction]
        _xyz = {
            'x': (0, utilities.get_x),
            'y': (1, utilities.get_y),
            'z': (2, utilities.get_z)
        }

        if _dir in _xyz.keys():
            direction = _xyz[_dir][0]
            _pos_group = (_xyz[_dir][1])(group)

        shift = dim[direction] / 100.

        _x = utilities.get_x(group.universe.atoms)
        _y = utilities.get_y(group.universe.atoms)
        _z = utilities.get_z(group.universe.atoms)

        _range = (0., dim[direction])
        if (halfbox_shift is True):
            _range = (-dim[direction] / 2., dim[direction] / 2.)

        histo, _ = np.histogram(
            _pos_group, bins=10, range=_range, density=True)

        max_val, min_val = np.amax(histo), np.amin(histo)
        # NOTE maybe allow user to set different values
        delta = min_val + (max_val - min_val) / 3.

        # let's first avoid crossing pbc with the liquid phase. This can fail:
        while (histo[0] > delta or histo[-1] > delta):
            total_shift += shift
            if total_shift >= dim[direction]:
                raise ValueError(messages.CENTERING_FAILURE)
            _pos_group += shift
            if _dir == 'x':
                utilities.centerbox(
                    group.universe,
                    x=_pos_group,
                    center_direction=direction,
                    halfbox_shift=halfbox_shift)
            if _dir == 'y':
                utilities.centerbox(
                    group.universe,
                    y=_pos_group,
                    center_direction=direction,
                    halfbox_shift=halfbox_shift)
            if _dir == 'z':
                utilities.centerbox(
                    group.universe,
                    z=_pos_group,
                    center_direction=direction,
                    halfbox_shift=halfbox_shift)
            histo, _ = np.histogram(
                _pos_group, bins=10, range=_range, density=True)

        _center_ = np.average(_pos_group)
        if (halfbox_shift is False):
            box_half = dim[direction] / 2.
        else:
            box_half = 0.
        _pos = {'x': _x, 'y': _y, 'z': _z}
        if _dir in _pos.keys():
            _pos[_dir] += total_shift - _center_ + box_half
        # finally, we copy everything back
        group.universe.atoms.positions = np.column_stack((_x, _y, _z))

    @staticmethod
    def shift_positions_to_middle(universe, normal):
        box = universe.dimensions[normal]
        translation = [0, 0, 0]
        translation[normal] = box / 2.
        universe.atoms.positions += np.array(translation)
        universe.atoms.pack_into_box(universe.dimensions[:3])

    def _shift_positions_to_middle(self):
        Interface.shift_positions_to_middle(self.universe, self.normal)

    @staticmethod
    def center_system(symmetry, group, direction, planar_to_origin=False):
        if symmetry == 'planar':
            utilities.centerbox(group.universe, center_direction=direction)
            Interface._center(group, direction, halfbox_shift=True)
            utilities.centerbox(group.universe, center_direction=direction)
            if planar_to_origin is False:
                Interface.shift_positions_to_middle(group.universe, direction)
        if symmetry == 'spherical':
            for xyz in ['x', 'y', 'z']:
                Interface._center(group, xyz, halfbox_shift=False)
            group.universe.atoms.pack_into_box(group.universe.dimensions[:3])

    def center(self, planar_to_origin=False):
        Interface.center_system(
            self.symmetry,
            self.cluster_group,
            self.normal,
            planar_to_origin=planar_to_origin)

        self.centered_positions = np.copy(self.universe.atoms.positions[:])

    def writepdb(self,
                 filename='layers.pdb',
                 centered='no',
                 group='all',
                 multiframe=True):
        _writepdb(
            self,
            filename=filename,
            centered=centered,
            group=group,
            multiframe=multiframe)
