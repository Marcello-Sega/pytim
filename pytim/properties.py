# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis

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


