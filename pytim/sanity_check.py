# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
import numpy as np
import MDAnalysis
from . import datafiles
from . import utilities
from . import messages
from .properties import Layers, Clusters, Sides, _create_property
from .properties import guess_radii, _missing_attributes


class SanityCheck(object):
    def __init__(self, interface, warnings=False):

        self.interface = interface
        self.interface._MDAversion = MDAnalysis.__version__
        self.interface.warnings = warnings

    def assign_radii(self):
        try:
            groups = [g for g in self.interface.extra_cluster_groups[:]]
        except BaseException:
            groups = []
        groups.append(self.interface.analysis_group)
        total = self.interface.universe.atoms[0:0]  # empty group
        for g in groups:
            if g is not None:
                guess_radii(self.interface, group=g)
                radii = np.copy(g.radii)
                total += g
        vals = list(self.interface.radii_dict.values())
        vals = np.array(vals).astype(float)
        avg = round(np.average(vals), 3)

        nantypes = total.types[np.isnan(total.radii)]
        radii = np.copy(total.radii)
        radii[np.isnan(radii)] = avg
        for nantype in nantypes:
            self.interface.guessed_radii.update({nantype: avg})
        total.radii = radii
        try:
            gr = self.interface.guessed_radii
            if gr and self.interface.warnings:
                print("guessed radii: ", gr, end=' ')
                print("You can override this by using, e.g.: pytim.", end='')
                print(self.interface.__class__.__name__, end=' ')
                print("(u,radii_dict={ '", end='')
                print(gr.keys()[0] + "':1.2 , ... } )")
        except BaseException:
            pass

    def assign_mesh(self, mesh):
        interface = self.interface
        box = interface.universe.dimensions[:3]
        interface.target_mesh = mesh
        if not isinstance(interface.target_mesh, (int, float)):
            raise TypeError(messages.MESH_NAN)
        if interface.target_mesh <= 0:
            raise ValueError(messages.MESH_NEGATIVE)
        if interface.target_mesh >= np.amin(box) / 2.:
            raise ValueError(messages.MESH_LARGE)

        try:
            np.arange(int(self.interface.alpha / self.interface.target_mesh))
        except BaseException:
            print(("Error while initializing ITIM: alpha ({0:f}) too large or\
                  mesh ({1:f}) too small".format(self.interface.alpha,
                                                 self.interface.target_mesh)))
            raise ValueError

    def assign_normal(self, normal):
        interface = self.interface
        if not (interface.symmetry == 'planar'):
            raise ValueError(" wrong symmetry for normal assignement")
        if interface.analysis_group is None:
            raise TypeError(messages.UNDEFINED_ANALYSIS_GROUP)
        if normal == 'guess':
            interface.normal = utilities.guess_normal(interface.universe,
                                                      interface.analysis_group)
        else:
            dirdict = {'x': 0, 'y': 1, 'z': 2}
            if not (normal in interface.directions_dict):
                raise ValueError(messages.WRONG_DIRECTION)
            interface.normal = dirdict[interface.directions_dict[normal]]

    def _define_groups(self):
        # we first make sure cluster_cut is either None, or an array
        if isinstance(self.interface.cluster_cut, (int, float)):
            self.interface.cluster_cut = np.array(
                [float(self.interface.cluster_cut)])
        # same with extra_cluster_groups
        if not isinstance(self.interface.extra_cluster_groups,
                          (list, tuple, np.ndarray, type(None))):
            self.interface.extra_cluster_groups = \
                [ self.interface.extra_cluster_groups ]

    def _check_group(self, input_obj):
        """ Check whether input_obj is one of the following,
            and act accordingly:
            - MDAnalysis.core.universe.Universe
            - MDAnalysis.core.groups.AtomGroup
            - mdtraj.core.trajectory.Trajectory
            - simtk.openmm.app.simulation.Simulation
        """

        if isinstance(input_obj, MDAnalysis.core.universe.Universe):
            self.interface.universe = input_obj
            self.interface.analysis_group = None
            return 'MDAnalysis'

        if isinstance(input_obj, MDAnalysis.core.groups.AtomGroup):
            self.interface.universe = input_obj.universe
            self.interface.analysis_group = input_obj
            return 'MDAnalysis'

        try:
            import os
            import tempfile
            import mdtraj
            if isinstance(input_obj, mdtraj.core.trajectory.Trajectory):
                _file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.pdb', delete=False)
                _file.close()
                input_obj[0].save_pdb(_file.name)
                self.interface.universe = MDAnalysis.Universe(_file.name)
                PatchMDTRAJ(input_obj, self.interface.universe)
                os.remove(_file.name)
                return 'mdtraj'
        except ImportError:
            pass
        try:
            import os
            from simtk.openmm.app.simulation import Simulation
            from simtk.openmm.app import pdbfile
            from simtk.unit import angstrom as openmm_AA

            if isinstance(input_obj, Simulation):
                _file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.pdb', delete=False)
                top = input_obj.topology
                context = input_obj.context
                pos = context.getState(getPositions=True).getPositions(
                    asNumpy=True).value_in_unit(openmm_AA)
                pdbfile.PDBFile.writeFile(
                    topology=top, positions=pos, file=_file)
                _file.close()
                self.interface.universe = MDAnalysis.Universe(_file.name)
                PatchOpenMM(input_obj, self.interface)
                os.remove(_file.name)
                return 'openmm'
        except ImportError:
            pass
        return None

    def assign_universe(self, input_obj, radii_dict=None):
        """ Tweak the details of the universe:

            - Compare input_obj against the possible classes. This
              makes possible the use of an AtomGroup in place of a
              Universe.
            - Load the radii from file or from radii_dict
            - Check for missing attributes
        """

        self.interface._mode = self._check_group(input_obj)
        if self.interface._mode is None:
            raise Exception(messages.WRONG_UNIVERSE)

        self.interface.all_atoms = self.interface.universe.select_atoms('all')

        if radii_dict is not None:
            self.interface.radii_dict = radii_dict.copy()
        else:
            self.interface.radii_dict = datafiles.pytim_data.vdwradii(
                datafiles.G43A1_TOP).copy()
            self.patch_radii_dict()

        _missing_attributes(self.interface, self.interface.universe)

    def patch_radii_dict(self):
        """ Fix here by hand common problems with radii assignment """
        self.interface.radii_dict['D'] = 0.0
        self.interface.radii_dict['M'] = 0.0
        self.interface.radii_dict['HW'] = 0.0
        self.interface.radii_dict['Me'] = self.interface.radii_dict['CMet']

    def assign_alpha(self, alpha):
        try:
            box = self.interface.universe.dimensions[:3]
        except BaseException:
            raise Exception("Cannot set alpha before having a simulation box")
        if alpha < 0:
            raise ValueError(messages.ALPHA_NEGATIVE)
        if alpha >= np.amin(box):
            raise ValueError(messages.ALPHA_LARGE)
        self.interface.alpha = alpha

    def assign_groups(self, analysis_group, cluster_cut, extra_cluster_groups):
        elements = 0
        extraelements = -1
        ag = analysis_group

        if self.interface.analysis_group is None:
            if ag is None:
                self.interface.analysis_group = self.interface.all_atoms
            elif isinstance(ag, int):
                self.interface.analysis_group = self.interface.universe.atoms[ag:ag + 1]
            elif isinstance(ag, list) or isinstance(ag, np.ndarray):
                self.interface.analysis_group = self.interface.universe.atoms[ag]
            else:
                self.interface.analysis_group = ag

        if (len(self.interface.analysis_group) == 0):
            raise RuntimeError(messages.UNDEFINED_ANALYSIS_GROUP)

        self.interface.cluster_cut = cluster_cut

        self.interface.extra_cluster_groups = extra_cluster_groups

        self._define_groups()
        interface = self.interface

        if (interface.cluster_cut is not None):
            elements = len(interface.cluster_cut)
        if (interface.extra_cluster_groups is not None):
            extraelements = len(interface.extra_cluster_groups)

        if not (elements == 1 or elements == 1 + extraelements):
            raise RuntimeError(messages.MISMATCH_CLUSTER_SEARCH)

    def check_multiple_layers_options(self):
        try:
            if self.interface.biggest_cluster_only is True:
                if self.interface.cluster_cut is None:
                    self.interface.biggest_cluster_only = False
                    print("Warning: the option biggest_cluster_only", end=' ')
                    print("has no effect without setting", end=' ')
                    print("cluster_cut, ignoring it")
        except BaseException:
            pass
