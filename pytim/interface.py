# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from abc import ABCMeta, abstractmethod
import numpy as np
from .properties import _create_property
from .writepdb import _writepdb
from . import messages
from . import utilities
from scipy.spatial import cKDTree


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
    autoassign, _autoassign =\
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
                if layer is None:
                    self._layers[uplow][nlayer] = self.universe.atoms[:0]
                else:
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
        try:
            self.universe.atoms.pack_into_box(self.universe.dimensions[:3])
        except ValueError:
            self.universe.atoms.pack_into_box(self.universe.dimensions)
        self.cluster_group = self.universe.atoms[:0]  # empty
        if (self.cluster_cut is not None):
            cluster_cut = float(self.cluster_cut[0])
            # we start by adding the atoms in the smaller clusters
            # of the opposit phase, if extra_cluster_groups are provided
            if (self.extra_cluster_groups is not None):
                for extra in self.extra_cluster_groups:
                    x_labels, x_counts, _ = utilities.do_cluster_analysis_dbscan(
                        extra, cluster_cut, self.cluster_threshold_density,
                        self.molecular)
                    x_labels = np.array(x_labels)
                    x_label_max = np.argmax(x_counts)
                    x_ids_other = np.where(x_labels != x_label_max)[0]

                    self.cluster_group += extra[x_ids_other]

            # next, we add the atoms belonging to the main phase
            self.cluster_group += self.itim_group

            # groups have been checked already in _sanity_checks()
            # self.cluster_group at this stage is composed of itim_group +
            # the smaller clusters of the other phase
            labels, counts, neighbors = utilities.do_cluster_analysis_dbscan(
                self.cluster_group, cluster_cut,
                self.cluster_threshold_density, self.molecular)
            labels = np.array(labels)

            # counts is not necessarily ordered by size of cluster.
            sorting = np.argsort(counts)[::-1]
            # labels for atoms in each cluster starting from the largest
            unique_labels = np.sort(np.unique(labels[labels > -1]))
            # by default, all elements of the cluster_group are in
            # single-molecule/atom clusters. We will update them right after.
            self.label_group(self.cluster_group, cluster=-1)
            # we go in reverse order to let smaller labels (bigger clusters)
            # overwrite larger labels (smaller cluster) when the molecular
            # option is used.
            for el in unique_labels[::-1]:
                # select a label
                cond = np.where(labels == el)
                if self.molecular is True:
                    g_ = self.cluster_group[cond].residues.atoms
                else:
                    g_ = self.cluster_group[cond]
                # probably we need an example here, say:
                # counts = [ 61, 1230, 34, 0, ...  0 ,0 ]
                # labels = [ 0, 1, 2, 1, -1  ....  -1 ]
                # we have three clusters, of 61, 1230 and 34 atoms.
                # There are 61 labels '0'
                #         1230 labels '1'
                #           34 labels '2'
                #         the remaining are '-1'
                #
                # sorting = [1,0,2,3,....] i.e. the largest element is in
                #     (1230) position 1, the next (61) is in position 0, ...
                # Say, g_ is now the group with label '1' (the biggest cluster)
                # Using argwhere(sorting==1) returns exactly 0 -> the right
                # ordered label for the largest cluster.
                self.label_group(g_, cluster=np.argwhere(sorting == el)[0, 0])
            # now that labels are assigned for each of the clusters,
            # we can restric the cluster group to the largest cluster.

            label_max = np.argmax(counts)
            ids_max = np.where(labels == label_max)[0]
            self.cluster_group = self.cluster_group[ids_max]
            self.n_neighbors = neighbors
        else:
            self.cluster_group = self.itim_group
            self.label_group(self.itim_group, cluster=1)
            self.label_group(self.cluster_group, cluster=0)

    def is_buried(self, pos):
        """ Checks wether an array of positions are located below
            the first interfacial layer """
        inter = self
        box = inter.universe.dimensions[:3]
        nonsurface = inter.cluster_group - inter.atoms[inter.atoms.layers == 1]
        # there are no inner atoms, distance is always > 0
        if len(nonsurface) == 0:
            return np.asarray([True] * len(pos))
        tree = cKDTree(nonsurface.positions, boxsize=box)
        neighs = tree.query_ball_point(pos, inter.alpha)
        condition = np.array([len(el) != 0 for el in neighs])
        return condition

    def reset_labels(self):
        self.label_group(
            self.universe.atoms, beta=0.0, layer=-1, cluster=-1, side=-1)

    @staticmethod
    def _attempt_shift(group, _pos_group, direction, halfbox_shift, _dir):
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

            Interface._attempt_shift(group, _pos_group, direction,
                                     halfbox_shift, _dir)

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
        try:
            universe.atoms.pack_into_box(universe.dimensions[:3])
        except ValueError:
            universe.atoms.pack_into_box(universe.dimensions)
            
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
        else:
            for xyz in [0, 1, 2]:
                try:
                    Interface._center(group, xyz, halfbox_shift=False)
                except ValueError:
                    pass
            try:
                group.universe.atoms.pack_into_box(group.universe.dimensions[:3])
            except ValueError:
                group.universe.atoms.pack_into_box(group.universe.dimensions)

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
                 multiframe=True,
                 tempfactors=None):
        """ Write the frame to a pdb file, marking the atoms belonging
            to the layers with different beta factors.

            :param str       filename:    the output file name
            :param str       centered:    'origin', 'middle', or 'no'
            :param AtomGroup group:       if 'all' is passed, use universe
            :param bool      multiframe:  append to pdb file if True
            :param ndarray   tempfactors: use this array as temp (beta) factors

            Example: save the positions (centering the interface in the cell)
                     without appending

            >>> import pytim
            >>> import pytim.datafiles
            >>> import MDAnalysis as mda
            >>> from pytim.datafiles import WATER_GRO
            >>> u = mda.Universe(WATER_GRO)
            >>> interface = pytim.ITIM(u)
            >>> interface.writepdb('layers.pdb',multiframe=False)

            Example: save the positions without centering the interface. This
                     will not shift the atoms from the original position
                     (still, they will be put into the basic cell).
                     The :obj:`multiframe` option set to :obj:`False` will
                     overwrite the file

            >>> interface.writepdb('layers.pdb',centered='no')

            Note that if :mod:`~pytim.gitim.GITIM` is used, and the
            :obj:`symmetry` option is different from :obj:`planar`,
            the :obj:`centered='origin'` option is equivalent to
            :obj:`centered='middle'`.
        """

        _writepdb(
            self,
            filename=filename,
            centered=centered,
            group=group,
            multiframe=multiframe,
            tempfactors=tempfactors)

    @staticmethod
    def _():
        """
        This is a collection of basic tests to check
        that code is running -- no test on the correctness
        of the output is performed here.

        >>> # TEST:0 loading the module
        >>> import pytim
        >>> pytim.ITIM._() ; # coverage

        >>> # TEST:1 basic functionality
        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import *
        >>> u = mda.Universe(WATER_GRO)
        >>> oxygens = u.select_atoms("name OW")
        >>> interface = pytim.ITIM(u, alpha=1.5, max_layers=4)
        >>> print (len(interface.layers[0,0]))
        786
        >>> del interface
        >>> interface = pytim.ITIM(u, alpha=1.5, max_layers=4, multiproc=False)
        >>> print (len(interface.layers[0,0]))
        786
        >>> del interface

        >>> # TEST:2 basic functionality
        >>> u=None
        >>> interface = pytim.GITIM(u)
        Traceback (most recent call last):
            ...
        Exception: Wrong Universe passed to ITIM class


        >>> interface = pytim.ITIM(u)
        Traceback (most recent call last):
            ...
        Exception: Wrong Universe passed to ITIM class

        >>> # TEST:3 large probe sphere radius
        >>> u = mda.Universe(WATER_GRO)
        >>> interface = pytim.ITIM(u, alpha=100000.0, max_layers=1,multiproc=False)
        Traceback (most recent call last):
            ...
        ValueError: parameter alpha must be smaller than the smaller box side

        >>> # TEST:3b no surface atoms
        >>> u = mda.Universe(GLUCOSE_PDB)
        >>> g = u.select_atoms('type C or name OW')
        >>> interface = pytim.GITIM(u,group=g, alpha=4.0)
        >>> print(interface.atoms)
        <AtomGroup []>

        >>> # TEST:4 interchangeability of Universe/AtomGroup
        >>> u = mda.Universe(WATER_GRO)
        >>> oxygens = u.select_atoms("name OW")
        >>> interface = pytim.ITIM(u, alpha=1.5,group=oxygens, max_layers=1,multiproc=False,molecular=False)
        >>> print (len(interface.layers[0,0]))
        262
        >>> interface = pytim.ITIM(oxygens, alpha=1.5,max_layers=1,multiproc=False,molecular=False)
        >>> print (len(interface.layers[0,0]))
        262


        >>> # PDB FILE FORMAT
        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import WATER_GRO
        >>> u = mda.Universe(WATER_GRO)
        >>> oxygens = u.select_atoms("name OW")
        >>> interface = pytim.ITIM(u, alpha=1.5, max_layers=4,molecular=True)
        >>> interface.writepdb('test.pdb',centered=False)
        >>> PDB =open('test.pdb','r').readlines()
        >>> line = list(filter(lambda l: 'ATOM     19 ' in l, PDB))[0]
        >>> beta = line[62:66] # PDB file format is fixed
        >>> print(beta)
        4.00


        >>> # mdtraj
        >>> try:
        ...     import mdtraj
        ...     try:
        ...         import numpy as np
        ...         import MDAnalysis as mda
        ...         import pytim
        ...         from pytim.datafiles import WATER_GRO,WATER_XTC
        ...         from pytim.datafiles import pytim_data,G43A1_TOP
        ...         # MDAnalysis
        ...         u = mda.Universe(WATER_GRO,WATER_XTC)
        ...         ref = pytim.ITIM(u)
        ...         # mdtraj
        ...         t = mdtraj.load_xtc(WATER_XTC,top=WATER_GRO)
        ...         # mdtraj manipulates the name of atoms, we need to set the
        ...         # radii by hand
        ...         _dict = { 'O':pytim_data.vdwradii(G43A1_TOP)['OW'],'H':0.0}
        ...         inter = pytim.ITIM(t, radii_dict=_dict)
        ...         ids_mda = []
        ...         ids_mdtraj = []
        ...         for ts in u.trajectory[0:2]:
        ...             ids_mda.append(ref.atoms.ids)
        ...         for ts in t[0:2]:
        ...             ids_mdtraj.append(inter.atoms.ids)
        ...         for fr in [0,1]:
        ...             if not np.all(ids_mda[fr] == ids_mdtraj[fr]):
        ...                 print ("MDAnalysis and mdtraj surface atoms do not coincide")
        ...         _a = u.trajectory[1] # we make sure we load the second frame
        ...         _b = t[1]
        ...         if not np.all(np.isclose(inter.atoms.positions[0], ref.atoms.positions[0])):
        ...             print("MDAnalysis and mdtraj atomic positions do not coincide")
        ...     except:
        ...         raise RuntimeError("mdtraj available, but a general exception happened")
        ... except:
        ...     pass


        >>> # check that using the biggest_cluster_only option without setting cluster_cut
        >>> # throws a warning and resets to biggest_cluster_only == False
        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from   pytim.datafiles import GLUCOSE_PDB
        >>>
        >>> u = mda.Universe(GLUCOSE_PDB)
        >>> solvent = u.select_atoms('name OW')
        >>> inter = pytim.GITIM(u, group=solvent, biggest_cluster_only=True)
        Warning: the option biggest_cluster_only has no effect without setting cluster_cut, ignoring it

        >>> print (inter.biggest_cluster_only)
        False

        >>> import pytim
        >>> import pytest
        >>> import MDAnalysis as mda
        >>> u = mda.Universe(pytim.datafiles.WATER_GRO)
        >>>
        >>> with pytest.raises(Exception):
        ...     pytim.ITIM(u,alpha=-1.0)

        >>> with pytest.raises(Exception):
        ...     pytim.ITIM(u,alpha=1000000)

        >>> pytim.ITIM(u,mesh=-1)
        Traceback (most recent call last):
        ...
        ValueError: parameter mesh must be positive


        >>> # check that it is possible to use two trajectories
        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import WATER_GRO, WATER_XTC
        >>> u = mda.Universe(WATER_GRO,WATER_XTC)
        >>> u2 = mda.Universe(WATER_GRO,WATER_XTC)
        >>> inter = pytim.ITIM(u,group=u.select_atoms('resname SOL'))
        >>> inter2 = pytim.ITIM(u2,group=u2.select_atoms('resname SOL'))
        >>> for ts in u.trajectory[::50]:
        ...     ts2 = u2.trajectory[ts.frame]

        """
        pass

    @staticmethod
    def __():
        """
        This is a collection of test to check
        that the algorithms are behaving properly if
        the interface is rotated in space.

        >>> # TEST:1, ITIM+GITIM, flat interface
        >>> import MDAnalysis as mda
        >>> import pytim
        >>> import numpy as np
        >>> from pytim.datafiles import WATER_GRO
        >>> pytim.ITIM.__() ; # coverage
        >>>
        >>> for method in [pytim.ITIM , pytim.GITIM] :
        ...     u = mda.Universe(WATER_GRO)
        ...     positions = np.copy(u.atoms.positions)
        ...     oxygens = u.select_atoms('name OW')
        ...     interface = method(u,group=oxygens,molecular=False,alpha=2.5,_noextrapoints=True)
        ...     #interface.writepdb(method.__name__+'.ref.pdb') ; # debug
        ...     ref_box = np.copy(u.dimensions)
        ...     ref_ind = np.sort(np.copy(interface.atoms.indices))
        ...     ref_pos = np.copy(interface.atoms.positions)
        ...
        ...     u.atoms.positions = np.copy(np.roll(positions,1,axis=1))
        ...     box = np.roll(ref_box[:3],1)
        ...     ref_box[:3] =  box
        ...     u.dimensions = ref_box
        ...     interface = method(u,group=oxygens,molecular=False,alpha=2.5,_noextrapoints=True)
        ...     ind = np.sort(interface.atoms.indices)
        ...     #interface.writepdb(method.__name__+'.pdb') ; # debug
        ...     cond = (ref_ind == ind )
        ...     if np.all(cond) ==  False:
        ...         miss1 = (np.in1d(ref_ind,ind)==False).sum()
        ...         miss2 = (np.in1d(ind,ref_ind)==False).sum()
        ...         percent = (miss1 + miss2)*0.5/len(ref_ind) * 100.
        ...         if percent > 2: # this should be 0 for ITIM, and < 5
        ...                         # for GITIM, with this config+alpha
        ...             print (miss1+miss2)
        ...             print ( " differences in indices in method",)
        ...             print ( method.__name__, " == ",percent," %")

        >>> del interface
        >>> del u

        >>> # TEST:2, GITIM, micelle
        >>> import MDAnalysis as mda
        >>> import pytim
        >>> import numpy as np
        >>> from pytim.datafiles import MICELLE_PDB
        >>>
        >>> for method in [pytim.GITIM] :
        ...     u = mda.Universe(MICELLE_PDB)
        ...     positions = np.copy(u.atoms.positions)
        ...     DPC = u.select_atoms('resname DPC')
        ...     interface = method(u,group=DPC,molecular=False,alpha=2.5,_noextrapoints=True)
        ...     #interface.writepdb(method.__name__+'.ref.pdb') ; # debug
        ...     ref_box = np.copy(u.dimensions)
        ...     ref_ind = np.sort(np.copy(interface.atoms.indices))
        ...     ref_pos = np.copy(interface.atoms.positions)
        ...
        ...     u.atoms.positions = np.copy(np.roll(positions,1,axis=1))
        ...     box = np.roll(ref_box[:3],1)
        ...     ref_box[:3] =  box
        ...     u.dimensions = ref_box
        ...     interface = method(u,group=DPC,molecular=False,alpha=2.5,_noextrapoints=True)
        ...     ind = np.sort(interface.atoms.indices)
        ...     #interface.writepdb(method.__name__+'.pdb') ; # debug
        ...     cond = (ref_ind == ind )
        ...     if np.all(cond) ==  False:
        ...         miss1 = (np.in1d(ref_ind,ind)==False).sum()
        ...         miss2 = (np.in1d(ind,ref_ind)==False).sum()
        ...         percent = (miss1 + miss2)*0.5/len(ref_ind) * 100.
        ...         if percent > 4 : # should be ~ 4 % for this system
        ...             print (miss1+miss2)
        ...             print ( " differences in indices in method",)
        ...             print ( method.__name__, " == ",percent," %")

        >>> del interface
        >>> del u

        """
        pass


#
