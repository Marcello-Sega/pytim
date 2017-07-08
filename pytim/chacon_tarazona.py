#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

""" Module: chacon_tarazona
    =======================
"""

import numpy as np
from pytim import utilities, surface
import pytim


class Surface(surface.Surface):

    def dump(self):
        pass

    def regular_grid(self):
        pass

    def triangulation(self):
        pass

    def distance(self, inp):
        positions = utilities.extract_positions(inp)
        return self._distance_flat(positions)

    def interpolation(self, inp):
        positions = utilities.extract_positions(inp)
        upper_set = positions[positions[:, 2] >= 0]
        lower_set = positions[positions[:, 2] < 0]

        elevation = np.zeros(len(positions))

        upper_interp = self.surface_from_modes(upper_set, self.modes[0])
        lower_interp = self.surface_from_modes(lower_set, self.modes[1])

        elevation[np.where(positions[:, 2] >= 0)] = upper_interp
        elevation[np.where(positions[:, 2] < 0)] = lower_interp
        return elevation


class ChaconTarazona(pytim.PYTIM):
    """Identifies the dividing surface using the Chacon-Tarazona method
       (Chacón, E., and P. Tarazona. Phys. Rev. Lett. 91, 166103,2003)
       (Tarazona, P., and E. Chacón. Phys. Rev. B 70, 235407,2004)

    :param Universe universe: the MDAnalysis universe
    :param float alpha:       molecular scale cutoff
    :param float tau:         particles within this distance form the\
                              surface will be added durin the\
                              self-consistent procedure.
    :param bool molecular:    Switches between search of interfacial\
                              molecules / atoms (default: True)
    :param AtomGroup group:   compute the density using this group
    :param dict radii_dict:   dictionary with the atomic radii of the\
                              elements in the group. If None is\
                              supplied, the default one (from MDAnalysis)\
                              will be used.

    Example:

    >>> import MDAnalysis as mda
    >>> import numpy as np
    >>> import pytim
    >>> from pytim.datafiles import WATER_GRO
    >>> u = mda.Universe(WATER_GRO)
    >>> g = u.select_atoms('name OW')
    >>> interface = pytim.ChaconTarazona(u,alpha=2.,tau=1.5,group=g,info=False,molecular=False)
    >>> interface.writepdb('CT.pdb',centered=True)
    >>> print repr(interface.layers)
    array([[<AtomGroup with 175 atoms>],
           [<AtomGroup with 159 atoms>]], dtype=object)

    """
    _surface = None

    @property
    def layers(self):
        return self._layers

    def __init__(self, universe, alpha=2.0, tau=1.5, group=None,
                 radii_dict=None, max_layers=1, normal='guess', molecular=True,
                 info=True, mesh=None, centered=False, **kargs):

        self.symmetry = 'planar'
        self.do_center = centered

        sanity = pytim.SanityCheck(self)
        sanity.assign_universe(universe)

        self.target_mesh = mesh
        if mesh is not None:
            raise Exception("FFT-based version not implemented")
        self.info = info
        self.alpha = alpha
        self.tau = tau
        if(max_layers != 1):
            raise Exception("max_layers !=1 not implemented yet!")

        self.max_layers = max_layers
        self._layers = np.empty([2, max_layers], dtype=type(universe.atoms))
        self._surfaces = np.empty(max_layers, dtype=type(Surface))
        self.normal = None
        self.PDB = {}

        self.molecular = molecular

        # TODO implement cluster group
        sanity.assign_groups(group, None, None)
        sanity.assign_normal(normal)
        sanity.assign_radii(radii_dict)

        self.old_box = np.array([-1, -1, -1])
        self.sorted_indices = None
        self.surf = None
        self.modes = [None, None]

        pytim.PatchTrajectory(universe.trajectory, self)
        self._assign_layers()
        self._atoms = self.LayerAtomGroupFactory(
            self._layers[:].sum().indices, self.universe)

    def _points_next_to_surface(self, surf, modes, pivot):
        """ searches for points within a distance self.tau from the
            interface.
        """
        z_max = np.max(self.cluster_group[pivot].positions[::, 2])
        z_min = np.min(self.cluster_group[pivot].positions[::, 2])
        z_max += self.alpha * 2
        z_min -= self.alpha * 2
        positions = self.cluster_group.positions[::]
        # TODO other directions
        z = positions[::, 2]
        condition = np.logical_and(z > z_min, z < z_max)
        candidates = np.argwhere(condition)[::, 0]
        dists = surf.surface_from_modes(positions[candidates], modes)
        dists = dists - z[candidates]
        return candidates[dists * dists < self.tau**2]

    def _initial_pivots(self, sorted_ind):
        """ defines the initial pivots as the furthermost particles in 3x3
            regions
        """
        sectors = (np.array([0] * 9)).reshape(3, 3)
        pivot = []
        box = utilities.get_box(self.universe, normal=self.normal)
        pos = utilities.get_pos(self.cluster_group, normal=self.normal)
        for ind in sorted_ind:
            part = pos[ind]
            nx, ny = map(int, 2.999 * part[0:2] / box[0:2])
            if sectors[nx, ny] == 0:
                pivot.append(ind)
                sectors[nx, ny] = 1
            if np.sum(sectors) >= 9:
                break
        return pivot

    def _assign_one_side(self, side):
        # TODO add successive layers
        box = self.universe.dimensions[:3]
#        surf = self._surfaces[0]

        if side == 0:
            sorted_ind = self.sorted_indices[::-1]
        else:
            sorted_ind = self.sorted_indices

        pivot = np.sort(self._initial_pivots(sorted_ind))
        modes = None
        while True:
            surf = Surface(self, options={'layer': 0})
            surf._compute_q_vectors(box)
            modes = surf.surface_modes(self.cluster_group[pivot].positions)
            p = self.cluster_group[pivot].positions
            s = surf.surface_from_modes(p, modes.reshape(surf.modes_shape))
            d = p[::, 2] - s
            if self.info == True:
                print "side", side, "->", len(pivot), "pivots, msd=",\
                    np.sqrt(np.sum(d * d) / len(d))
            # TODO handle failure
            modes = modes.reshape(surf.modes_shape)
            new_pivot = np.sort(
                self._points_next_to_surface(surf, modes, pivot))
            if np.all(new_pivot == pivot):
                self.surf = surf
                self.modes[side] = modes
                _inlayer_group = self.cluster_group[pivot]
                if self.molecular == True:
                    _tmp = _inlayer_group.residues.atoms
                    _inlayer_group = _tmp
                return _inlayer_group

            else:
                pivot = new_pivot

    def _assign_layers(self):
        """ This function identifies the dividing surface and the atoms in the
        layers.

        """

        # TODO parallelize

        # this can be used later to shift back to the original shift
        self.original_positions = np.copy(self.universe.atoms.positions[:])
        self.universe.atoms.pack_into_box()

        box = self.universe.dimensions[:3]

        # groups have been checked already in _sanity_checks()

        self._define_cluster_group()

        self.centered_positions = None
        # we always (internally) center in Chacon-Tarazona
        self.center(planar_to_origin=True)
        # first we label all atoms in group to be in the gas phase
        self.label_group(self.itim_group.atoms, 0.5)
        # then all atoms in the largest group are labelled as liquid-like
        self.label_group(self.cluster_group.atoms, 0)

        self.old_box = box
        pos = self.cluster_group.positions
        self.sorted_indices = np.argsort(pos[::, self.normal])
        for side in [0, 1]:
            self._layers[side][0] = self._assign_one_side(side)
        for side in [0, 1]:
            for _nlayer, _layer in enumerate(self._layers[side]):
                self.label_group(_layer, _nlayer + 1)

        if self.do_center == False:
            self.universe.atoms.positions = self.original_positions
        else:
            self._shift_positions_to_middle()

#
