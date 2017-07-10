# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

from abc import ABCMeta, abstractmethod, abstractproperty
from distutils.version import LooseVersion
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import MDAnalysis
from MDAnalysis import Universe
from MDAnalysis.topology import tables
from difflib import get_close_matches
import importlib
import __builtin__


def PatchTrajectory(trajectory, interface):
    """ Patch the MDAnalysis trajectory class

        this patch makes makes the layer assignement being automatically
        called whenever a new frame is loaded.
    """
    try:
        trajectory.interface
    except:
        trajectory.interface = interface
        trajectory.original_read_next_timestep = trajectory._read_next_timestep

        if LooseVersion(interface._MDAversion) >= LooseVersion('0.16'):
            trajectory.original_read_frame_with_aux = trajectory._read_frame_with_aux
        else:
            trajectory.original_read_frame_with_aux = None

        class PatchedTrajectory(trajectory.__class__):

            def _read_next_timestep(self, ts=None):
                tmp = self.original_read_next_timestep(ts=ts)
                self.interface._assign_layers()
                return tmp

            def _read_frame_with_aux(self, frame):
                if frame != self.frame:
                    tmp = self.original_read_frame_with_aux(frame)
                    self.interface._assign_layers()
                    return tmp
                return self.ts

        oldname = trajectory.__class__.__name__
        oldmodule = trajectory.__class__.__module__

        PatchedTrajectory.__name__ = oldname
        PatchedTrajectory.__module__ = oldmodule
        trajectory.__class__ = PatchedTrajectory


def _create_property(property_name, docstring=None,
                     readonly=False, required=False):

    def getter(self):
        return self.__getattribute__('_' + property_name)

    def setter(self, value):
        self.__setattr__('_' + property_name, value)

    if readonly == True:
        setter = None
    if required == False:
        absprop = None
    else:
        absprop = abstractproperty(None)

    return property(fget=getter, fset=setter, doc=docstring), absprop


class SanityCheck(object):

    def __init__(self, interface):
        self.interface = interface
        self.interface._MDAversion = MDAnalysis.__version__
        self.VERS = LooseVersion(self.interface._MDAversion)
        self.V015 = LooseVersion('0.15')
        self.V016 = LooseVersion('0.16')
        if self.VERS < self.V015:
            raise Exception("Must use MDAnalysis  >= 0.15")

    def assign_radii(self, radii_dict):
        try:
            _groups = np.copy(self.interface.extra_cluster_groups[:])
        except BaseException:
            _groups = []
        _groups.append(self.interface.itim_group)
        for _g in _groups:
            # NOTE: maybe add a switch to use the atom name instead of the type
            if _g is not None:
                _types = np.copy(_g.types)
                if not (np.any(np.equal(_g.radii, None)) or
                        np.any(np.isnan(_g.radii))):  # all radii already set
                    break
                if radii_dict is None:
                    # some radii are not set and no dict provided
                    _radii_dict = tables.vdwradii
                else:  # use the provided dict.
                    _radii_dict = radii_dict

                _radii = np.zeros(len(_g.types))
                for _atype in np.unique(_types):
                    try:
                        matching_type = get_close_matches(
                            _atype, _radii_dict.keys(), n=1, cutoff=0.1
                        )
                        _radii[_types == _atype] = _radii_dict[matching_type[0]]
                    except BaseException:
                        avg = np.average(_radii_dict.values())
                        _radii[_types == _atype] = avg

                        print("!!                              WARNING")
                        print("!! No appropriate radius was found for the "
                              "atomtype " + _atype)
                        print("!! Using the average radius (" + str(avg) + ") "
                              "as a fallback option...")
                        print("!! Pass a dictionary of radii (in Angstrom) "
                              "with the option radii_dict")
                        print("!! for example: r={'" + _atype + "':1.2,...} ; "
                              "inter=pytim.ITIM(u,radii_dict=r)")

                _g.radii = np.copy(_radii[:])

                if np.any(np.equal(_g.radii, None)):
                    raise ValueError(self.interface.UNDEFINED_RADIUS)
                del _radii
                del _types

    def assign_mesh(self, mesh):
        self.interface.target_mesh = mesh
        if not isinstance(self.interface.target_mesh, (int, float)):
            raise TypeError(self.interface.MESH_NAN)
        if self.interface.target_mesh <= 0:
            raise ValueError(self.interface.MESH_NEGATIVE)
        if self.interface.target_mesh >= np.amin(self.interface.universe.dimensions[:3]) / 2.:
            raise ValueError(self.interface.MESH_LARGE)

        try:
            np.arange(int(self.interface.alpha / self.interface.target_mesh))
        except BaseException:
            print(
                "Error while initializing ITIM: alpha ({0:f}) too large or\
                  mesh ({1:f}) too small".format(
                    self.interface.alpha,
                    self.interface.target_mesh))
            raise ValueError

    def assign_normal(self, normal):
        if not (self.interface.symmetry == 'planar'):
            raise ValueError(" wrong symmetry for normal assignement")
        if self.interface.itim_group is None:
            raise TypeError(self.interface.UNDEFINED_ITIM_GROUP)
        if normal == 'guess':
            self.interface.normal = utilities.guess_normal(self.interface.universe,
                                                           self.interface.itim_group)
        else:
            if not (normal in self.interface.directions_dict):
                raise ValueError(self.interface.WRONG_DIRECTION)
            self.interface.normal = self.interface.directions_dict[normal]

    def _define_groups(self):
        # we first make sure cluster_cut is either None, or an array
        if isinstance(self.interface.cluster_cut, (int, float)):
            self.interface.cluster_cut = np.array(
                [float(self.interface.cluster_cut)])
        # same with extra_cluster_groups
        if not isinstance(self.interface.extra_cluster_groups,
                          (list, tuple, np.ndarray, type(None))):
            self.interface.extra_cluster_groups = [
                self.interface.extra_cluster_groups]

        # fallback for itim_group
        if self.interface.itim_group is None:
            self.interface.itim_group = self.interface.all_atoms

    def _missing_attributes(self, universe):
        if self.VERS >= self.V016:  # new topology system
            self.topologyattrs = importlib.import_module(
                'MDAnalysis.core.topologyattrs'
            )
            guessers = MDAnalysis.topology.guessers
            self._check_missing_attribute('radii', 'Radii', universe.atoms,
                                          np.nan, universe)
            self._check_missing_attribute('tempfactors', 'Tempfactors',
                                          universe.atoms, 0.0, universe)
            self._check_missing_attribute('bfactors', 'Bfactors',
                                          universe.atoms, 0.0, universe)
            self._check_missing_attribute('altLocs', 'AltLocs',
                                          universe.atoms, ' ', universe)
            self._check_missing_attribute('icodes', 'ICodes',
                                          universe.residues, ' ', universe)
            self._check_missing_attribute('occupancies', 'Occupancies',
                                          universe.atoms, 1, universe)
            self._check_missing_attribute('elements', 'Elements',
                                          universe.atoms, 1, universe)

    def _check_missing_attribute(self, name, classname, group, value, universe):
        """ Add an attribute, which is necessary for pytim but
            missing from the present topology.

            An example of how the code below would expand is:

            if 'radii' not in dir(universe.atoms):
                from MDAnalysis.core.topologyattrs import Radii
                radii = np.zeros(len(universe.atoms)) * np.nan
                universe.add_TopologyAttr(Radii(radii))

             * MDAnalysis.core.topologyattrs ->  self.topologyattrs
             * Radii -> missing_class
             * radii -> values
        """

        if name not in dir(universe.atoms):
            missing_class = getattr(self.topologyattrs, classname)
            values = np.array([value] * len(group))
            universe.add_TopologyAttr(missing_class(values))
            if name == 'elements':
                types = MDAnalysis.topology.guessers.guess_types(group.names)
                # is there an inconsistency in the way 'element' is defined
                # different modules in MDA?
                group.elements = np.array(
                    [utilities.atomic_number_map.get(t, 0) for t in types])

    def assign_universe(self, universe):
        try:
            self.interface.all_atoms = universe.select_atoms('all')
        except BaseException:
            raise Exception(self.interface.WRONG_UNIVERSE)
        self.interface.universe = universe
        self._missing_attributes(self.interface.universe)

    def assign_alpha(self, alpha):
        try:
            box = self.interface.universe.dimensions[:3]
        except:
            raise Exception("Cannot set alpha before having a simulation box")
        if alpha < 0:
            raise ValueError(self.interface.ALPHA_NEGATIVE)
        if alpha >= np.amin(box):
            raise ValueError(self.interface.ALPHA_LARGE)
        self.interface.alpha = alpha
        return True

    def assign_groups(self, itim_group, cluster_cut, extra_cluster_groups):
        elements = 0
        extraelements = -1

        self.interface.itim_group = itim_group
        self.interface.cluster_cut = cluster_cut
        self.interface.extra_cluster_groups = extra_cluster_groups

        self._define_groups()
        if(len(self.interface.itim_group) == 0):
            raise StandardError(self.interface.UNDEFINED_ITIM_GROUP)
        interface = self.interface

        if(interface.cluster_cut is not None):
            elements = len(interface.cluster_cut)
        if(interface.extra_cluster_groups is not None):
            extraelements = len(interface.extra_cluster_groups)

        if not (elements == 1 or elements == 1 + extraelements):
            raise StandardError(self.interface.MISMATCH_CLUSTER_SEARCH)

        return True


class PYTIM(object):
    """ The PYTIM metaclass
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
        'Z:': 'z'}
    symmetry_dict = {
        'cylindrical': 'cylindrical',
        'spherical': 'spherical',
        'planar': 'planar'}

    ALPHA_NEGATIVE = "parameter alpha must be positive"
    ALPHA_LARGE = "parameter alpha must be smaller than the smaller box side"
    MESH_NEGATIVE = "parameter mesh must be positive"
    MESH_LARGE = "parameter mesh must be smaller than the smaller box side"
    UNDEFINED_RADIUS = "one or more atoms do not have a corresponding radius in the default or provided dictionary"
    UNDEFINED_CLUSTER_SEARCH = "If extra_cluster_groups is defined, a cluster_cut should e provided"
    MISMATCH_CLUSTER_SEARCH = "cluster_cut should be either a scalar or an array matching the number of groups (including itim_group)"
    EMPTY_LAYER = "One or more layers are empty"
    CLUSTER_FAILURE = "Cluster algorithm failed: too small cluster cutoff provided?"
    UNDEFINED_LAYER = "No layer defined: forgot to call _assign_layers() or not enough layers requested"
    WRONG_UNIVERSE = "Wrong Universe passed to ITIM class"
    UNDEFINED_ITIM_GROUP = "No itim_group defined, or empty"
    WRONG_DIRECTION = "Wrong direction supplied. Use 'x','y','z' , 'X', 'Y', 'Z' or 0, 1, 2"
    CENTERING_FAILURE = "Cannot center the group in the box. Wrong direction supplied?"

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

    @property
    def method(self):
        return self.__class__.__name__

    def label_group(self, group, value):
        if self.molecular == True:
            _group = group.residues.atoms
        else:
            _group = group

        if LooseVersion(self._MDAversion) < LooseVersion('0.16'):
            _group.bfactors = float(value)
        else:
            _group.tempfactors = float(value)

    def _assign_symmetry(self, symmetry):
        if self.itim_group is None:
            raise TypeError(self.UNDEFINED_ITIM_GROUP)
        if symmetry == 'guess':
            raise ValueError("symmetry 'guess' To be implemented")
        else:
            if not (symmetry in self.symmetry_dict):
                raise ValueError(self.WRONG_DIRECTION)
            self.symmetry = symmetry

    def _generate_periodic_border_2d(self, group):
        _box = utilities.get_box(group.universe, self.normal)

        positions = utilities.get_pos(group, self.normal)

        shift = np.diagflat(_box)

        eps = min(2. * self.alpha, _box[0], _box[1])
        L = [eps, eps]
        low = [None, None]
        upp = [None, None]
        U = [_box[0] - eps, _box[1] - eps]

        pos = positions[:]
        for xy in [0, 1]:
            low[xy] = positions[positions[:, xy] <= L[xy]] + shift[xy]
            upp[xy] = positions[positions[:, xy] >= U[xy]] - shift[xy]

        low_x_low_y = positions[np.logical_and(
            positions[:, 0] <= L[0], positions[:, 1] <= L[1])] + (shift[0] + shift[1])
        upp_x_upp_y = positions[np.logical_and(
            positions[:, 0] >= U[0], positions[:, 1] >= U[1])] - (shift[0] + shift[1])
        low_x_upp_y = positions[np.logical_and(
            positions[:, 0] <= L[0], positions[:, 1] >= U[1])] + (shift[0] - shift[1])
        upp_x_low_y = positions[np.logical_and(
            positions[:, 0] >= U[0], positions[:, 1] <= L[1])] - (shift[0] - shift[1])

        return np.concatenate(
            (pos,
             low[0],
             low[1],
             upp[0],
             upp[1],
             low_x_low_y,
             upp_x_upp_y,
             low_x_upp_y,
             upp_x_low_y))

    def LayerAtomGroupFactory(self, indices, universe):
        if LooseVersion(MDAnalysis.__version__) >= \
                LooseVersion('0.16'):  # new topology system
            AtomGroup = MDAnalysis.core.groups.AtomGroup
        else:
            AtomGroup = MDAnalysis.core.AtomGroup.AtomGroup

        class LayerAtomGroup(AtomGroup):
            def __init__(self, interface, *args):
                AtomGroup.__init__(self, *args)
                self.interface = interface
                self._in_layers = self.LayerAtomGroupSelector(
                    self.interface._layers)

            @property
            def in_layers(self):
                return self._in_layers

            class LayerAtomGroupSelector(object):
                def __init__(self, layers):
                    self._layers = layers

                def __getitem__(self, key):
                    obj = self._layers[key]
                    if isinstance(obj, AtomGroup):
                        return obj
                    if isinstance(obj, np.ndarray):
                        return obj.sum()
                    return None
        if LooseVersion(MDAnalysis.__version__) >= \
                LooseVersion('0.16'):  # new topology system
            return LayerAtomGroup(self, indices, universe)
        else:
            return LayerAtomGroup(self, universe.atoms[indices])

    @property
    def atoms(self):
        return self._atoms

    def writepdb(self, filename='layers.pdb', centered='no', group='all', multiframe=True):
        """ Write the frame to a pdb file, marking the atoms belonging
            to the layers with different beta factor.

            :param filename:   string  -- the output file name
            :param centered:   string  -- 'origin', 'middle', or 'no'
            :param group:      AtomGroup -- if 'all' is passed, the universe is used
            :param multiframe: boolean -- append to pdb file if True

            Example: save the positions (centering the interface in the cell) without appending

            >>> interface.writepdb('layers.pdb',multiframe=False)

            Example: save the positions without centering the interface. This will
                     not shift the atoms from the original position (still, they will
                     be put into the basic cell).
                     The :multiframe: option set to :False: will overwrite the file.

            >>> interface.writepdb('layers.pdb',centered='no')

        """
        if isinstance(group, self.universe.atoms.__class__):
            self.group = group
        else:
            self.group = self.universe.atoms

        temp_pos = np.copy(self.universe.atoms.positions)
        options = {'no': False, False: False, 'middle': True, True: True}

        if options[centered] != self.do_center:
            # i.e. we don't have already what we want ...
            if self.do_center == False:  # we need to center
                self.center(planar_to_origin=True)
            else:  # we need to put back the original positions
                try:
                    # original_positions are (must) always be defined
                    self.universe.atoms.positions = self.original_positions
                except:
                    raise NameError
        try:
            # it exists already, let's add information about the box, as
            # MDAnalysis forgets to do so for successive frames. A bugfix
            # should be on the way for the next version...
            self.PDB[filename].CRYST1(
                self.PDB[filename].convert_dimensions_to_unitcell(
                    self.universe.trajectory.ts
                )
            )
        except BaseException:
            if LooseVersion(self._MDAversion) >= LooseVersion('0.16'):
                bondvalue = None
            else:
                bondvalue = False
            self.PDB[filename] = MDAnalysis.Writer(
                filename, multiframe=True,
                n_atoms=self.group.atoms.n_atoms,
                bonds=bondvalue
            )
        self.PDB[filename].write(self.group.atoms)
        self.universe.atoms.positions = np.copy(temp_pos)

    def savepdb(self, filename='layers.pdb', centered='no', multiframe=True):
        """ An alias to :func:`writepdb`
        """
        self.writepdb(filename, centered, multiframe)

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
            self.cluster_group = self.itim_group[ids_max]
            self.n_neighbors = neighbors
        else:
            self.cluster_group = self.itim_group

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
            raise ValueError(self.WRONG_DIRECTION)
        _dir = self.directions_dict[direction]
        if _dir == 'x':
            direction = 0
            _pos_group = utilities.get_x(group)
        if _dir == 'y':
            direction = 1
            _pos_group = utilities.get_y(group)
        if _dir == 'z':
            direction = 2
            _pos_group = utilities.get_z(group)

        shift = dim[direction] / 100.

        _x = utilities.get_x(group.universe.atoms)
        _y = utilities.get_y(group.universe.atoms)
        _z = utilities.get_z(group.universe.atoms)

        if(halfbox_shift == True):
            _range = (-dim[direction] / 2., dim[direction] / 2.)
        else:
            _range = (0., dim[direction])

        histo, _ = np.histogram(_pos_group, bins=10, range=_range,
                                density=True)

        max_val = np.amax(histo)
        min_val = np.amin(histo)
        # NOTE maybe allow user to set different values
        delta = min_val + (max_val - min_val) / 3.

        # let's first avoid crossing pbc with the liquid phase. This can fail:
        while(histo[0] > delta or histo[-1] > delta):
            total_shift += shift
            if total_shift >= dim[direction]:
                raise ValueError(self.CENTERING_FAILURE)
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
        if _dir == 'x':
            _x += total_shift - _center + box_half
        if _dir == 'y':
            _y += total_shift - _center + box_half
        if _dir == 'z':
            _z += total_shift - _center + box_half
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
            self._center(self.cluster_group, 'x', halfbox_shift=False)
            self._center(self.cluster_group, 'y', halfbox_shift=False)
            self._center(self.cluster_group, 'z', halfbox_shift=False)
            self.universe.atoms.pack_into_box(self.universe.dimensions[:3])
            self.centered_positions = np.copy(self.universe.atoms.positions[:])

    @abstractmethod
    def _assign_layers(self):
        pass


from pytim.itim import ITIM
from pytim.gitim import GITIM
from pytim.willard_chandler import WillardChandler
from pytim.chacon_tarazona import ChaconTarazona
from pytim import observables, utilities, datafiles

#__all__ = [ 'itim' , 'gitim' , 'observables', 'datafiles', 'utilities']
