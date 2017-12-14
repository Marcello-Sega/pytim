#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

""" Module: itim
    ============
"""

from multiprocessing import Process, Queue
import numpy as np
from __builtin__ import zip as builtin_zip
from scipy.spatial import cKDTree
from pytim import utilities, surface
import pytim


class Surface(surface.Surface):

    def distance(self, inp):
        positions = utilities.extract_positions(inp)
        return self._distance_flat(positions)

    def interpolation(self, inp):
        positions = utilities.extract_positions(inp)
        upper_set = positions[positions[:, 2] >= 0]
        lower_set = positions[positions[:, 2] < 0]

        elevation = np.zeros(len(positions))

        self._initialize_distance_interpolator_flat(layer=self._layer)
        upper_interp = self._interpolator[0](upper_set[:, 0:2])
        lower_interp = self._interpolator[1](lower_set[:, 0:2])

        elevation[np.where(positions[:, 2] >= 0)] = upper_interp
        elevation[np.where(positions[:, 2] < 0)] = lower_interp
        return elevation

    def dump(self):
        pass

    def regular_grid(self):
        pass

    def triangulation(self, layer=0):
        return self.triangulate_layer_flat(layer)


class ITIM(pytim.PYTIM):
    """Identifies interfacial molecules at macroscopically flat interfaces.

        :param Universe universe: the MDAnalysis universe
        :param float mesh:        the grid spacing used for the testlines
        :param float alpha:       the probe sphere radius
        :param str normal:        the macroscopic interface normal direction\
                                  'x','y' or 'z' (by default is 'guess')
        :param AtomGroup group:   identify the interfacial molecules from\
                                     this group
        :param dict radii_dict:   dictionary with the atomic radii of the\
                                  elements in the group. If None is\
                                  supplied, the default one (from MDAnalysis)\
                                  will be used.
        :param int max_layers:    the number of layers to be identified
        :param float cluster_cut: cutoff used for neighbors or density-based\
                                  cluster search (default: None disables the\
                                  cluster analysis)
        :param float cluster_threshold_density: Number density threshold for\
                                  the density-based cluster search. 'auto'\
                                  determines the threshold automatically.\
                                  Default: None uses simple neighbors cluster\
                                  search, if cluster_cut is not None
        :param bool molecular:    Switches between search of interfacial\
                                  molecules / atoms (default: True)
        :param bool info:         print additional info
        :param bool multiproc:    parallel version (default: True. Switch off\
                                  for debugging)

        Example:

        >>> import MDAnalysis as mda
        >>> import numpy as np
        >>> import pytim
        >>> from pytim.datafiles import *
        >>>
        >>> u         = mda.Universe(WATER_GRO)
        >>> oxygens   = u.select_atoms("name OW")
        >>>
        >>> interface = pytim.ITIM(u, alpha=1.5, max_layers=4,molecular=True)

        >>> # atoms in the layers can be accesses either through
        >>> # the layers array:
        >>> print interface.layers
        [[<AtomGroup with 786 atoms> <AtomGroup with 681 atoms>
          <AtomGroup with 663 atoms> <AtomGroup with 651 atoms>]
         [<AtomGroup with 786 atoms> <AtomGroup with 702 atoms>
          <AtomGroup with 666 atoms> <AtomGroup with 636 atoms>]]


        >>> interface.layers[0,0] # upper side, first layer
        <AtomGroup with 786 atoms>

        >>> interface.layers[1,2] # lower side, third layer
        <AtomGroup with 666 atoms>

        >>> # or as a whole AtomGroup. This can include all atoms in all layers:
        >>> interface.atoms
        <AtomGroup with 5571 atoms>


        >>> selection = interface.atoms.sides == 0
        >>> interface.atoms[ selection ] # all atoms in the upper side layer
        <AtomGroup with 2781 atoms>
        >>> selection = np.logical_and(interface.atoms.layers == 2 , selection)
        >>> interface.atoms[ selection ] # upper side, second layer
        <AtomGroup with 681 atoms>

        >>> # the whole system can be quickly saved to a pdb file
        >>> # including the layer information, written in the beta field
        >>> # using:
        >>> interface.writepdb('system.pdb',centered=True)

        >>> # of course, the native interface of MDAnalysis can be used to write pdb files
        >>> # but the centering options are not available. Writing to other formats that
        >>> # do not support the beta factor will loose the information on the layers.
        >>> interface.atoms.write('only_layers.pdb')

    """

    @property
    def layers(self):
        """Access the layers as numpy arrays of AtomGroups.

        The object can be sliced as usual with numpy arrays, so, for example:

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import *
        >>>
        >>> u         = mda.Universe(WATER_GRO)
        >>> oxygens   = u.select_atoms("name OW")
        >>>
        >>> interface = pytim.ITIM(u, alpha=1.5, max_layers=4,molecular=True)
        >>> interface.layers[0,:]  # upper side (0), all layers
        array([<AtomGroup with 786 atoms>, <AtomGroup with 681 atoms>,
               <AtomGroup with 663 atoms>, <AtomGroup with 651 atoms>], \
dtype=object)

        >>> interface.layers[1,0]  # lower side (1), first layer (0)
        <AtomGroup with 786 atoms>


        >>> interface.layers[:,0:3] # 1st - 3rd layer (0:3), on both sides
        array([[<AtomGroup with 786 atoms>, <AtomGroup with 681 atoms>,
                <AtomGroup with 663 atoms>],
               [<AtomGroup with 786 atoms>, <AtomGroup with 702 atoms>,
                <AtomGroup with 666 atoms>]], dtype=object)


        >>> interface.layers[1,0:4:2] # side 1, layers 1-4 & stride 2 (0:4:2)
        array([<AtomGroup with 786 atoms>, <AtomGroup with 666 atoms>], \
dtype=object)


        """

        return self._layers

    def __init__(self, universe, mesh=0.4, alpha=1.5, normal='guess',
                 group=None, radii_dict=None, max_layers=1,
                 cluster_cut=None, cluster_threshold_density=None,
                 molecular=True, extra_cluster_groups=None, info=False,
                 multiproc=True, centered=False, warnings=False, **kargs):

        self.symmetry = 'planar'
        self.do_center = centered

        sanity = pytim.SanityCheck(self)
        sanity.assign_universe(
            universe, radii_dict=radii_dict, warnings=warnings)
        sanity.assign_alpha(alpha)
        sanity.assign_mesh(mesh)

        self.cluster_threshold_density = cluster_threshold_density
        self.max_layers = max_layers
        self._layers = np.empty([2, max_layers], dtype=self.universe.atoms[0].__class__)
        self._surfaces = np.empty(max_layers, dtype=type(Surface))
        self.info = info
        self.normal = None
        self.PDB = {}
        self.molecular = molecular

        sanity.assign_groups(group, cluster_cut, extra_cluster_groups)
        sanity.assign_normal(normal)
        sanity.assign_radii()

        self.grid = None
        self.use_threads = False
        self.use_kdtree = True
        self.use_multiproc = multiproc

        pytim.PatchTrajectory(self.universe.trajectory, self)

        self._assign_layers()

    def _assign_mesh(self):
        """ Mesh assignment method

            Based on a target value, determine a mesh size for the testlines
            that is compatible with the simulation box.
            Create the grid and initialize a cKDTree object with it to
            facilitate fast searching of the gridpoints touched by molecules.
        """
        box  = utilities.get_box(self.universe, self.normal)
        n, d = utilities.compute_compatible_mesh_params(self.target_mesh, box)
        self.mesh_nx = n[0]
        self.mesh_ny = n[1]
        self.mesh_dx = d[0]
        self.mesh_dy = d[1]
        if (self.use_kdtree == True):
            _x = np.linspace(0,box[0],num=self.mesh_nx, endpoint=False)
            _y = np.linspace(0,box[1],num=self.mesh_ny, endpoint=False)
            _X,_Y = np.meshgrid(_x,_y)
            self.meshpoints = builtin_zip(_X.ravel(), _Y.ravel())
            # cKDTree requires a box vetor with length double the dimension,
            _box = np.zeros(4)
            _box[:2] = box[:2]
            try:  # older scipy versions
                self.meshtree = cKDTree(self.meshpoints, boxsize=_box)
            except:
                self.meshtree = cKDTree(self.meshpoints, boxsize=_box[:2])

    def _touched_lines(self, atom, _x, _y, _z, _radius):
        return self.meshtree.query_ball_point(
            [_x[atom], _y[atom]], _radius[atom] + self.alpha)

    def _assign_one_side(self, uplow, sorted_atoms, _x, _y, _z,
                         _radius, queue=None):
        _layers = []
        for layer in range(0, self.max_layers):
            # this mask tells which lines have been touched.
            mask = self.mask[uplow][layer]
            _inlayer = []
            # atom here goes to 0 to #sorted_atoms, it is not a MDAnalysis
            # index/atom
            for atom in sorted_atoms:
                if self._seen[uplow][atom] != 0:
                    continue

                touched_lines = self._touched_lines(atom, _x, _y, _z, _radius)
                _submask = mask[touched_lines]

                if(len(_submask[_submask == 0]) == 0):
                    # no new contact, let's move to the next atom
                    continue

                # let's mark now:
                # 1) the touched lines
                mask[touched_lines] = 1

                # 2) the sorted atom
                # start counting from 1, 0 will be
                self._seen[uplow][atom] = layer + 1

                # 3) if all lines have been touched, create a group that
                # includes all atoms in this layer
                # NOTE that checking len(mask[mask==0])==0 is slower.
                if np.sum(mask) == len(mask):
                    _inlayer_indices = np.flatnonzero(
                        self._seen[uplow] == layer + 1)
                    _inlayer_group = self.cluster_group[_inlayer_indices]

                    if self.molecular == True:
                        # we first select the (unique) residues corresponding
                        # to _inlayer_group, and then we create  group of the
                        # atoms belonging to them, with
                        # _inlayer_group.residues.atoms
                        _tmp = _inlayer_group.residues.atoms
                        # and we copy it back to _inlayer_group
                        _inlayer_group = _tmp
                        # now we need the indices within the cluster_group,
                        # of the atoms in the molecular layer group;
                        # NOTE that from MDAnalysis 0.16, .ids runs from 1->N
                        # (was 0->N-1 in 0.15), we use now .indices
                        _indices = np.flatnonzero(
                            np.in1d(
                                self.cluster_group.atoms.indices,
                                _inlayer_group.atoms.indices))
                        # and update the tagged, sorted atoms
                        self._seen[uplow][_indices] = layer + 1

                    # one of the two layers (upper,lower) or both are empty
                    if not _inlayer_group:
                        raise Exception(self.EMPTY_LAYER)

                    _layers.append(_inlayer_group)
                    break
        if(queue is None):
            return _layers
        else:
            queue.put(_layers)

    def _sanity_checks(self):
        """ Basic checks to be performed after the initialization.

            We test them also here in the docstring:

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

        """

    def _assign_layers(self):
        """ Determine the ITIM layers.

            Note that the multiproc option is mainly for debugging purposes:
            >>> import MDAnalysis as mda
            >>> import pytim
            >>> u = mda.Universe(pytim.datafiles.WATER_GRO)
            >>> inter = pytim.ITIM(u,multiproc=True)
            >>> test1 = len(inter.layers[0,0])
            >>> inter = pytim.ITIM(u,multiproc=False)
            >>> test2 = len(inter.layers[0,0])
            >>> test1==test2
            True

        """

        self.label_group(self.universe.atoms, beta=0.0,
                         layer=-1, cluster=-1, side=-1)
        self._assign_mesh()
        up = 0
        low = 1
        self.layers_ids = [[], []]  # upper, lower
        size = (2, self.max_layers, self.mesh_nx * self.mesh_ny)
        self.mask = np.zeros(size, dtype=int)

        # this can be used later to shift back to the original shift
        self.original_positions = np.copy(self.universe.atoms.positions[:])

        self.universe.atoms.pack_into_box()
        # groups have been checked already in _sanity_checks()

        self._define_cluster_group()

        # we always (internally) center in ITIM
        self.center(planar_to_origin=True)

        # first we label all atoms in group to be in the gas phase
        self.label_group(self.itim_group.atoms, beta=0.5)
        # then all atoms in the largest group are labelled as liquid-like
        self.label_group(self.cluster_group.atoms, beta=0.0)

        _radius = self.cluster_group.radii
        self._seen = [[], []]
        size = len(self.cluster_group.positions)
        self._seen[up] = np.zeros(size, dtype=np.int8)
        self._seen[low] = np.zeros(size, dtype=np.int8)

        _x = utilities.get_x(self.cluster_group, self.normal)
        _y = utilities.get_y(self.cluster_group, self.normal)
        _z = utilities.get_z(self.cluster_group, self.normal)
        sort = np.argsort(_z + _radius * np.sign(_z))
        # NOTE: np.argsort returns the sorted *indices*

        if self.use_multiproc:
            # so far, it justs exploit a simple scheme splitting
            # the calculation between the two sides. Would it be
            # possible to implement easily 2d domain decomposition?
            proc = [None, None]
            queue = [Queue(), Queue()]
            proc[up] = Process(target=self._assign_one_side,
                               args=(up, sort[::-1], _x, _y, _z,
                                     _radius, queue[up]))
            proc[low] = Process(target=self._assign_one_side,
                                args=(low, sort[::], _x, _y, _z,
                                      _radius, queue[low]))

            for p in proc:
                p.start()
            for uplow in [up, low]:
                for index, group in enumerate(queue[uplow].get()):
                    # cannot use self._layers[uplow][index] = group, otherwise
                    # info about universe is lost (do not know why yet)
                    # must use self._layers[uplow][index] =
                    # self.universe.atoms[group.indices]
                    self._layers[uplow][index] =\
                        self.universe.atoms[group.indices]
            for p in proc:
                p.join()
        else:
            for index, group in enumerate(self._assign_one_side(
                    up, sort[::-1], _x, _y, _z, _radius)):
                self._layers[up][index] = group
            for index, group in enumerate(self._assign_one_side(
                    low, sort[::], _x, _y, _z, _radius)):
                self._layers[low][index] = group

        # Assign to all layers a label (tempfactor) that can be used in pdb files.
        # Additionally, set the new layers and sides
        for uplow in [up, low]:
            for nlayer, layer in enumerate(self._layers[uplow]):
                self.label_group(layer, beta=nlayer + 1.0,
                                 layer=nlayer + 1, side=uplow)

        for nlayer, layer in enumerate(self._layers[0]):
            self._surfaces[nlayer] = Surface(self, options={'layer': nlayer})

        if self.do_center == False:  # NOTE: do_center requires centering in the middle of the box
                                    # ITIM always centers internally in the
                                    # origin along the normal
            self.universe.atoms.positions = self.original_positions
        else:
            self._shift_positions_to_middle()

#
