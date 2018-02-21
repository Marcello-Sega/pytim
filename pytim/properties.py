# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
from abc import abstractproperty
import MDAnalysis
import importlib
import numpy as np
from .atoms_maps import atoms_maps
from difflib import get_close_matches


class Layers(MDAnalysis.core.topologyattrs.AtomAttr):
    """Layers for each atom"""
    attrname = 'layers'
    singular = 'layer'
    per_object = 'atom'


class Clusters(MDAnalysis.core.topologyattrs.AtomAttr):
    """Clusters for each atom"""
    attrname = 'clusters'
    singular = 'cluster'
    per_object = 'atom'


class Sides(MDAnalysis.core.topologyattrs.AtomAttr):
    """Sides for each atom"""
    attrname = 'sides'
    singular = 'side'
    per_object = 'atom'


def _create_property(property_name,
                     docstring=None,
                     readonly=False,
                     required=False):
    def getter(self):
        return self.__getattribute__('_' + property_name)

    def setter(self, value):
        self.__setattr__('_' + property_name, value)

    if readonly is True:
        setter = None
    if required is False:
        absprop = None
    else:
        absprop = abstractproperty(None)

    return property(fget=getter, fset=setter, doc=docstring), absprop


def _missing_attributes(interface, universe):
    interface._topologyattrs = importlib.import_module(
        'MDAnalysis.core.topologyattrs')
    _check_missing_attribute(interface, 'names', 'Atomnames', universe.atoms,
                             universe.atoms.ids.astype(str))
    # NOTE _check_missing_attribute() relies on radii being set to np.nan
    # if the attribute radii is not present
    _check_missing_attribute(interface, 'radii', 'Radii', universe.atoms,
                             np.nan)
    _check_missing_attribute(interface, 'tempfactors', 'Tempfactors',
                             universe.atoms, 0.0)
    _check_missing_attribute(interface, 'bfactors', 'Bfactors', universe.atoms,
                             0.0)
    _check_missing_attribute(interface, 'altLocs', 'AltLocs', universe.atoms,
                             ' ')
    _check_missing_attribute(interface, 'icodes', 'ICodes', universe.residues,
                             ' ')
    _check_missing_attribute(interface, 'occupancies', 'Occupancies',
                             universe.atoms, 1)
    _check_missing_attribute(interface, 'elements', 'Elements', universe.atoms,
                             1)
    _extra_attributes(interface, universe)


def _extra_attributes(interface, universe):
    # we add here the new layer, cluster and side information
    # they are not part of MDAnalysis.core.topologyattrs
    attr = {'layers': Layers, 'clusters': Clusters, 'sides': Sides}
    for key in attr.keys():
        if key not in dir(universe.atoms):
            vals = np.zeros(len(universe.atoms), dtype=np.int) - 1
            universe.add_TopologyAttr(attr[key](vals))


def _check_missing_attribute(interface, name, classname, group, value):
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
    universe = interface.universe
    if name not in dir(universe.atoms):
        missing_class = getattr(interface._topologyattrs, classname)
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
                [atoms_maps.get(t, n0)['number'] for t in types])
        if name == 'radii':
            guess_radii(interface)


def weighted_close_match(string, dictionary):
    # increase weight of the first letter
    # this fixes problems with atom names like CH12
    _wdict = {}
    _dict = dictionary
    _str = string[0] + string[0] + string
    for key in _dict.keys():
        _wdict[key[0] + key[0] + key] = _dict[key]
    m = get_close_matches(_str, _wdict.keys(), n=1, cutoff=0.1)[0]
    return m[2:]


def _guess_radii_from_masses(interface, group, guessed):
    radii = np.copy(group.radii)
    masses = group.masses
    types = group.types
    unique_masses = np.unique(masses)
    # Let's not consider atoms with zero mass.
    unique_masses = unique_masses[unique_masses > 0]
    d = atoms_maps
    for target_mass in unique_masses:
        atype, _ = min(
            d.items(),
            key=lambda __entry: abs(__entry[1]['mass'] - target_mass))
        try:
            match_type = get_close_matches(
                atype, interface.radii_dict.keys(), n=1, cutoff=0.1)
            rd = interface.radii_dict
            radii[masses == target_mass] = rd[match_type[0]]
            for t in types[masses == target_mass]:
                guessed.update({t: rd[match_type[0]]})
        except BaseException:
            pass
        group.radii = radii


def _guess_radii_from_types(interface, group, guessed):
    radii = np.copy(group.radii)

    _dict = interface.radii_dict
    for aname in np.unique(group.names):
        try:
            matching_type = weighted_close_match(aname, _dict)
            radii[group.names == aname] = _dict[matching_type]
            guessed.update({aname: _dict[matching_type]})
        except (KeyError, IndexError):
            try:
                atype = group.types[group.names == aname][0]
                matching_type = weighted_close_match(atype, _dict)
                radii[group.types == atype] = _dict[matching_type]
                guessed.update({atype: _dict[matching_type]})
            except (KeyError, IndexError):
                pass
    group.radii = np.copy(radii)


def guess_radii(interface, group=None):
    # NOTE: this code depends on the assumption that not-set radii,
    # have the value np.nan (see _missing_attributes() ), so don't change it
    # let's test first which information is available
    guessed = {}
    try:
        interface.guessed_radii.update({})
    except AttributeError:
        interface.guessed_radii = {}

    if group is None:
        group = interface.universe.atoms

    nans = np.isnan(group.radii)
    # if no radius is nan, no need to guess anything
    if not (np.any(np.equal(group.radii, None)) or np.any(nans)):
        return
    nones = np.equal(group.radii, None)
    group.radii[nones] = np.array([np.nan] * len(group.radii[nones]))

    group = group[np.isnan(group.radii)]

    # We give precedence to atom names, then to types
    try:
        # this test failes wither if no 'type' property
        # is available, or if it is, but the values are
        # integers (like in lammps) and thus cannot be
        # used to guess the type (in this code)
        group.types.astype(int)
    except AttributeError:  # no types at all
        pass  # will try with masses
    except ValueError:  # types are there, and are not integers
        _guess_radii_from_types(interface, group, guessed)

    # We fill in the remaining ones using masses information
    group = group[np.isnan(group.radii)]

    if ('masses' in dir(group)):
        _guess_radii_from_masses(interface, group, guessed)

    interface.guessed_radii.update(guessed)
