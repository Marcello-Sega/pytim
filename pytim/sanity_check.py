# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

from distutils.version import LooseVersion
import importlib
import numpy as np
import MDAnalysis
from . import datafiles
from . import utilities
from pytim.properties import Layers,Clusters,Sides,_create_property
from difflib import get_close_matches
from . import messages

class SanityCheck(object):

    def __init__(self, interface):

        self.interface = interface
        self.interface._MDAversion = MDAnalysis.__version__
        self.V016 = LooseVersion('0.16')

    def assign_radii(self):
        try:
            groups = [g for g in self.interface.extra_cluster_groups[:]]
        except BaseException:
            groups = []
        groups.append(self.interface.itim_group)
        total = self.interface.universe.atoms[0:0]  # empty group
        for g in groups:
            if g is not None:
                self.guess_radii(group=g)
                radii = np.copy(g.radii)
                total += g
        avg = round(np.average(self.interface.radii_dict.values()), 3)

        nantypes = total.types[np.isnan(total.radii)]
        radii = np.copy(total.radii)
        radii[np.isnan(radii)] = avg
        for nantype in nantypes:
            self.guessed_radii.update({nantype: avg})
        total.radii = radii
        try:
            if self.guessed_radii != {} and self.interface.warnings == True:
                print "guessed radii: ", self.guessed_radii,
                print "You can override this by using, e.g.: pytim." + self.interface.__class__.__name__,
                print "(u,radii_dict={ '" + self.guessed_radii.keys()[0] + "':1.2 , ... } )"
        except BaseException:
            pass

    def assign_mesh(self, mesh):
        self.interface.target_mesh = mesh
        if not isinstance(self.interface.target_mesh, (int, float)):
            raise TypeError(messages.MESH_NAN)
        if self.interface.target_mesh <= 0:
            raise ValueError(messages.MESH_NEGATIVE)
        if self.interface.target_mesh >= np.amin(self.interface.universe.dimensions[:3]) / 2.:
            raise ValueError(messages.MESH_LARGE)

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
            raise TypeError(messages.UNDEFINED_ITIM_GROUP)
        if normal == 'guess':
            self.interface.normal = utilities.guess_normal(self.interface.universe,
                                                           self.interface.itim_group)
        else:
            dirdict = {'x': 0, 'y': 1, 'z': 2}
            if not (normal in self.interface.directions_dict):
                raise ValueError(messages.WRONG_DIRECTION)
            self.interface.normal = dirdict[self.interface.directions_dict[normal]]

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
        self.topologyattrs = importlib.import_module(
            'MDAnalysis.core.topologyattrs'
        )
        guessers = MDAnalysis.topology.guessers
        self._check_missing_attribute('names', 'Atomnames', universe.atoms,
                                      universe.atoms.ids.astype(str), universe)
        # NOTE _check_missing_attribute() relies on radii being set to np.nan
        # if the attribute radii is not present
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
        # we add here the new layer, cluster and side information
        # they are not part of MDAnalysis.core.topologyattrs
        if 'layers' not in dir(universe.atoms):
            layers = np.zeros(len(universe.atoms), dtype=np.int) - 1
            universe.add_TopologyAttr(Layers(layers))

        if 'clusters' not in dir(universe.atoms):
            clusters = np.zeros(len(universe.atoms), dtype=np.int) - 1
            universe.add_TopologyAttr(Clusters(clusters))

        if 'sides' not in dir(universe.atoms):
            sides = np.zeros(len(universe.atoms), dtype=np.int) - 1
            universe.add_TopologyAttr(Sides(sides))

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
            if isinstance(value, np.ndarray) or isinstance(value, list):
                if len(value) == len(group):
                    values = np.array(value)
                else:
                    raise RuntimeError("improper array/list length")
            else:
                values = np.array([value] * len(group))
            universe.add_TopologyAttr(missing_class(values))
            if name == 'elements':
                types = MDAnalysis.topology.guessers.guess_types(group.names)
                # is there an inconsistency in the way 'element' is defined
                # different modules in MDA?
                n0 = {'number': 0} 
                # Note: the second arg in .get() is the default.
                group.elements = np.array(
                    [utilities.atoms_maps.get(t, n0)['number'] for t in types])
            if name == 'radii':
                self.guess_radii()

    def weighted_close_match(self, string, dictionary):
        # increase weight of the first letter
        # this fixes problems with atom names like CH12
        _wdict = {}
        _dict = dictionary
        _str = string[0] + string[0] + string
        for key in _dict.keys():
            _wdict[key[0] + key[0] + key] = _dict[key]
        m = get_close_matches(_str, _wdict.keys(), n=1, cutoff=0.1)[0]
        return m[2:]

    def guess_radii(self, group=None):
        # NOTE: this code depends on the assumption that not-set radii,
        # have the value np.nan (see _missing_attributes() ), so don't change it
        # let's test first which information is available
        guessed = {}
        try:
            self.guessed_radii.update({})
        except AttributeError:
            self.guessed_radii = {}

        universe = self.interface.universe
        if group is None:
            group = self.interface.universe.atoms

        nans = np.isnan(group.radii)
        # if no radius is nan, no need to guess anything
        if not (np.any(np.equal(group.radii, None)) or np.any(nans)):
            return
        nones = np.equal(group.radii, None)
        group.radii[nones] = np.array([np.nan] * len(group.radii[nones]))
        #group.radii = group.radii.astype(np.float32)

        group = group[np.isnan(group.radii)]

        have_masses = ('masses' in dir(group))

        have_types = False
        if 'types' in dir(group):
            have_types = True
            try:
                # When atom types are str(ints), e.g. lammps ,
                # we cannot use them to guess radii

                # trying to convert strings to integers:
                group.types.astype(int)
                have_types = False
            except ValueError:
                pass

        # We give precedence to atom names, then to types
        if have_types:
            radii = np.copy(group.radii)

            _dict = self.interface.radii_dict
            for aname in np.unique(group.names):
                try:
                    matching_type = self.weighted_close_match(aname, _dict)
                    radii[group.names == aname] = _dict[matching_type]
                    guessed.update({aname: _dict[matching_type]})
                except:
                    try:
                        atype = group.types[group.names == aname][0]
                        matching_type = self.weighted_close_match(atype, _dict)
                        radii[group.types == atype] = _dict[matching_type]
                        guessed.update({atype: _dict[matching_type]})
                    except:
                        pass

            group.radii = np.copy(radii)
        # We fill in the remaining ones using masses information

        group = group[np.isnan(group.radii)]

        if have_masses:
            radii = np.copy(group.radii)
            masses = group.masses
            types = group.types
            unique_masses = np.unique(masses)
            # Let's not consider atoms with zero mass.
            unique_masses = unique_masses[unique_masses > 0]
            d = utilities.atoms_maps
            for target_mass in unique_masses:
                atype, mass = min(d.items(), key=lambda (
                    _, entry): abs(entry['mass'] - target_mass))
                try:
                    matching_type = get_close_matches(
                        atype, self.interface.radii_dict.keys(), n=1, cutoff=0.1
                    )

                    radii[masses == target_mass] = self.interface.radii_dict[matching_type[0]]

                    [guessed.update({t: self.interface.radii_dict[matching_type[0]]})
                     for t in types[masses == target_mass]]
                except:
                    pass
            group.radii = radii
        self.guessed_radii.update(guessed)

    def _apply_patches(self, input_obj):

        if isinstance(input_obj, MDAnalysis.core.universe.Universe):
            self.interface.universe = input_obj
            self.interface.itim_group = None
            return 'MDAnalysis'

        if isinstance(input_obj, MDAnalysis.core.groups.AtomGroup):
            self.interface.universe = input_obj.universe
            self.interface.itim_group = input_obj
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
                pos = input_obj.context.getState(getPositions=True).getPositions(
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

    def assign_universe(self, input_obj, radii_dict=None, warnings=False):

        self.interface._mode = self._apply_patches(input_obj)
        if self.interface._mode is None:
            raise Exception(messages.WRONG_UNIVERSE)

        self.interface.all_atoms = self.interface.universe.select_atoms('all')
        #self.interface.radii_dict = tables.vdwradii.copy()
        self.interface.radii_dict = datafiles.pytim_data.vdwradii(
            datafiles.G43A1_TOP).copy()
        self.patch_radii_dict()
        self.interface.warnings = warnings
        if radii_dict is not None:
            self.interface.radii_dict = radii_dict.copy()

        self._missing_attributes(self.interface.universe)

    def patch_radii_dict(self):
        # fix here by hand common problems with radii assignment
        self.interface.radii_dict['D'] = 0.0
        self.interface.radii_dict['M'] = 0.0
        self.interface.radii_dict['HW'] = 0.0
        self.interface.radii_dict['Me'] = self.interface.radii_dict['CMet']

    def assign_alpha(self, alpha):
        try:
            box = self.interface.universe.dimensions[:3]
        except:
            raise Exception("Cannot set alpha before having a simulation box")
        if alpha < 0:
            raise ValueError(messages.ALPHA_NEGATIVE)
        if alpha >= np.amin(box):
            raise ValueError(messages.ALPHA_LARGE)
        self.interface.alpha = alpha

    def wrap_group(self, inp):
        if inp is None:
            return None
        if isinstance(inp, int):
            return self.interface.universe.atoms[inp:inp + 1]
        if isinstance(inp, list) or isinstance(inp, np.ndarray):
            return self.interface.universe.atoms[inp]
        return inp

    def assign_groups(self, itim_group, cluster_cut, extra_cluster_groups):
        elements = 0
        extraelements = -1

        if self.interface.itim_group is None:
            self.interface.itim_group = self.wrap_group(itim_group)

        self.interface.cluster_cut = self.wrap_group(cluster_cut)
        self.interface.extra_cluster_groups = self.wrap_group(
            extra_cluster_groups)

        self._define_groups()
        if(len(self.interface.itim_group) == 0):
            raise StandardError(messages.UNDEFINED_ITIM_GROUP)
        interface = self.interface

        if(interface.cluster_cut is not None):
            elements = len(interface.cluster_cut)
        if(interface.extra_cluster_groups is not None):
            extraelements = len(interface.extra_cluster_groups)

        if not (elements == 1 or elements == 1 + extraelements):
            raise StandardError(messages.MISMATCH_CLUSTER_SEARCH)

        return True

    def check_multiple_layers_options(self):
        try:
            if self.interface.biggest_cluster_only == True and self.interface.cluster_cut == None:
                self.interface.biggest_cluster_only = False
                print "Warning: the option biggest_cluster_only has no effect without setting cluster_cut, ignoring it"
        except:
            pass


