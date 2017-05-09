# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

from abc import ABCMeta, abstractmethod, abstractproperty
from distutils.version import LooseVersion
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import MDAnalysis
from MDAnalysis import Universe
from MDAnalysis.topology import tables
from difflib import get_close_matches
import importlib


def PatchTrajectory(trajectory, interface):
    """ Patch the MDAnalysis trajectory class

        this patch makes makes the layer assignement being automatically
        called whenever a new frame is loaded.
    """
    trajectory.interface = interface
    trajectory.original_read_next_timestep = trajectory._read_next_timestep

    class PatchedTrajectory(trajectory.__class__):

        def _read_next_timestep(self, ts=None):
            tmp = self.original_read_next_timestep(ts=ts)
            self.interface._assign_layers()
            return tmp

    oldname = trajectory.__class__.__name__
    oldmodule = trajectory.__class__.__module__

    PatchedTrajectory.__name__ = oldname
    PatchedTrajectory.__module__ = oldmodule
    trajectory.__class__ = PatchedTrajectory


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
    UNDEFINED_ITIM_GROUP = "No itim_group defined"
    WRONG_DIRECTION = "Wrong direction supplied. Use 'x','y','z' , 'X', 'Y', 'Z' or 0, 1, 2"
    CENTERING_FAILURE = "Cannot center the group in the box. Wrong direction supplied?"

    def __init__(self):
        self.cluster_cut = None
        self.extra_cluster_groups = None
        self.itim_group = None
        self.radii_dict = None
        self.max_layers = 1
        self.cluster_threshold_density = None
        self.molecular = True
        self.info = False
        self.multiproc = True
        self._interpolator = None

    def label_layer(self, group, value):
        if LooseVersion(self._MDAversion) <= LooseVersion('0.15'):
            group.bfactors = value
        else:
            group.tempfactors = value

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

    def _sanity_check_alpha(self):
        if self.alpha < 0:
            raise ValueError(self.ALPHA_NEGATIVE)
        if self.alpha >= np.amin(self.universe.dimensions[:3]):
            raise ValueError(self.ALPHA_LARGE)

    def _sanity_check_cluster_cut(self):
        if(self.cluster_cut is not None):
            elements = len(self.cluster_cut)
            try:
                extraelements = len(self.extra_cluster_groups)
            except TypeError:
                extraelements = -1
            if  not (elements == 1 or elements == 1 + extraelements):
                raise  StandardError(self.MISMATCH_CLUSTER_SEARCH)
        else:
            if self.extra_cluster_groups is not None:
                raise ValueError(self.UNDEFINED_CLUSTER_SEARCH)


    def _basic_checks(self, universe):
        self._MDAversion = MDAnalysis.__version__
        LooseV = LooseVersion(self._MDAversion)
        V015 = LooseVersion('0.15')
        V016 = LooseVersion('0.16')
        try:
            self.all_atoms = universe.select_atoms('all')
        except BaseException:
            raise Exception(self.WRONG_UNIVERSE)

        if LooseV < V015:
            raise Exception("Must use MDAnalysis  >= 0.15")

        if LooseV >= V016:  # new topology system

            self.topologyattrs = importlib.import_module(
                'MDAnalysis.core.topologyattrs'
            )

            self._check_missing_attribute('radii', 'Radii', universe.atoms,
                                          np.nan, universe)
            self._check_missing_attribute('tempfactors', 'Tempfactors',
                                          universe.atoms, 0, universe)
            self._check_missing_attribute('bfactors', 'Bfactors',
                                          universe.atoms, 0, universe)
            self._check_missing_attribute('altLocs', 'AltLocs',
                                          universe.atoms, ' ', universe)
            self._check_missing_attribute('icodes', 'ICodes',
                                          universe.residues, ' ', universe)
            self._check_missing_attribute('occupancies', 'Occupancies',
                                          universe.atoms, 1, universe)

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

    def writepdb(self, filename='layers.pdb', centered='no', multiframe=True):
        """ Write the frame to a pdb file, marking the atoms belonging
            to the layers with different beta factor.

            :param filename:   string  -- the output file name
            :param centered:   string  -- 'origin', 'middle', or 'no'
            :param multiframe: boolean -- append to pdb file if True

            Example: save the positions (centering the interface in the cell) without appending

            >>> interface.writepdb('layers.pdb',multiframe=False)

            Example: save the positions without centering the interface. This will
                     not shift the atoms from the original position (still, they will
                     be put into the basic cell).
                     The :multiframe: option set to :False: will overwrite the file.

            >>> interface.writepdb('layers.pdb',centered='no')

        """
        options={'no':False,False:False,'middle':True,True:True}
        if options[centered] == False:
            self.universe.atoms.positions = self.original_positions

        if options[centered] == True:
            # NOTE: this assumes that all method relying on 'planar'
            # symmetry must center the interface along the normal
            box = self.universe.dimensions[self.normal]
            translation = [0, 0, 0]
            translation[self.normal] = box / 2.
            self.universe.atoms.positions += np.array(translation)
            self.universe.atoms.pack_into_box(self.universe.dimensions[:3])
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
            if LooseVersion(self._MDAversion) > LooseVersion('0.15'):
                bondvalue = None
            else:
                bondvalue = False
            self.PDB[filename] = MDAnalysis.Writer(
                                    filename, multiframe=True,
                                    n_atoms=self.universe.atoms.n_atoms,
                                    bonds=bondvalue
                                 )
        self.PDB[filename].write(self.universe.atoms)

    def savepdb(self, filename='layers.pdb', centered='no', multiframe=True):
        """ An alias to :func:`writepdb`
        """
        self.writepdb(filename, centered, multiframe)

    def assign_radii(self, radii_dict):
        try:
            _groups = np.copy(self.extra_cluster_groups[:])
        except BaseException:
            _groups = []
        _groups.append(self.itim_group)
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
                    raise ValueError(self.UNDEFINED_RADIUS)
                del _radii
                del _types

    def triangulate_layer(self, layer=1):
        """Triangulate a layer.

        :param int layer:  (default: 1) triangulate this layer (on both sides\
                           of the interface)
        :return list triangulations:  a list of two Delaunay triangulations,\
                           which are also stored in self.surf_triang
        """
        if layer > len(self._layers[0]):
            raise ValueError(self.UNDEFINED_LAYER)

        box = self.universe.dimensions[:3]

        upper = self._layers[0][layer - 1]
        lower = self._layers[1][layer - 1]

        upperpos = self._generate_periodic_border_2d(upper)
        lowerpos = self._generate_periodic_border_2d(lower)

        self.surf_triang = [None, None]
        self.trimmed_surf_triangs = [None, None]
        self.triangulation_points = [None, None]
        self.surf_triang[0] = Delaunay(upperpos[:, 0:2])
        self.surf_triang[1] = Delaunay(lowerpos[:, 0:2])
        self.triangulation_points[0] = upperpos[:]
        self.triangulation_points[1] = lowerpos[:]
        self.trimmed_surf_triangs[0] = utilities.trim_triangulated_surface(
            self.surf_triang[0], box
        )
        self.trimmed_surf_triangs[1] = utilities.trim_triangulated_surface(
            self.surf_triang[1], box
        )
        return self.surf_triang

    def _initialize_distance_interpolator(self, layer):
        if self._interpolator is None:
            # we don't know if previous triangulations have been done on the
            # same layer, so just in case we repeat it here. This can be fixed
            # in principl with a switch
            self.triangulate_layer(layer)

            self._interpolator = [None, None]
            for layer in [0,1]:
                self._interpolator[layer] = LinearNDInterpolator(
                    self.surf_triang[layer],
                    self.triangulation_points[layer][:, 2])

    def interpolate_surface(self, positions, layer):
        self._initialize_distance_interpolator(layer)
        upper_set = positions[positions[:, 2] >= 0]
        lower_set = positions[positions[:, 2] < 0]
        # interpolated values of upper/lower_set on the upper/lower surface
        upper_int = self._interpolator[0](upper_set[:, 0:2])
        lower_int = self._interpolator[1](lower_set[:, 0:2])
        # copy everything back to one array with the correct order
        elevation = np.zeros(len(positions))
        elevation[np.where(positions[:, 2] >= 0)] = upper_int
        elevation[np.where(positions[:, 2] < 0)] = lower_int
        return elevation

    def _assign_normal(self, normal):
        if not (self.symmetry == 'planar'):
            raise ValueError(" wrong symmetry for normal assignement")
        if self.itim_group is None:
            raise TypeError(self.UNDEFINED_ITIM_GROUP)
        if normal == 'guess':
            self.normal = utilities.guess_normal(self.universe,
                                                 self.itim_group)
        else:
            if not (normal in self.directions_dict):
                raise ValueError(self.WRONG_DIRECTION)
            self.normal = self.directions_dict[normal]

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

    @abstractmethod
    def _assign_layers(self):
        pass

    @abstractproperty
    def layers(self):
        pass

    def _define_groups(self):
        # we first make sure cluster_cut is either None, or an array
        if isinstance(self.cluster_cut, (int, float)):
                self.cluster_cut = np.array([float(self.cluster_cut)])
        # same with extra_cluster_groups
        if not isinstance(self.extra_cluster_groups,
                           (list, tuple, np.ndarray,type(None))):
            self.extra_cluster_groups = [self.extra_cluster_groups]

        # fallback for itim_group
        if self.itim_group is None:
            self.itim_group = self.all_atoms

from pytim.itim import ITIM
from pytim.gitim import GITIM
from pytim.willard_chandler import WillardChandler
from pytim import observables, utilities, datafiles

#__all__ = [ 'itim' , 'gitim' , 'observables', 'datafiles', 'utilities']
