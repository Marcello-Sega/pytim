# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import numpy as np
from .interface import Interface
from .sanity_check import SanityCheck
from .surface import SurfaceFlatInterface
from .surface import SurfaceGenericInterface


class SimpleInterface(Interface):
    """ This simple interface is designed to allow the user to define
        its own interface by supplying one or more groups, and be able to
        use all other analysis tools of pytim, which require an interface,
        also when it is not advisable/possible to use any of the other
        classes

        :param Universe universe: The MDAnalysis_ Universe.
        :param AtomGroup group:   The AtomGroup with the interfacial atoms.
                                  (only for symmetry='generic'). See the upper
                                  and lower groups options below.
        :param float alpha:       The probe sphere radius (used for handling
                                  the periodicity in the surface layer
                                  triangulation only)
        :param str normal:        The macroscopic interface normal direction
                                  'x','y', or 'z'(default)
        :param str symmetry:      Either 'grneric' (default) or 'planar'.
                                  Selects the type of interpolation/distance
                                  calculation. If 'planar' two groups have to
                                  be also passed (upper and lower)
        :param AtomGroup upper:   The upper surface group (if symmetry='planar')
        :param AtomGroup lower:   The lower surface group (if symmetry='planar')


        Example: compute an intrinsic profile by interpolating the surface
        position from the P atoms of a DPPC bilayer

        >>> import pytim
        >>> import numpy as np
        >>> from pytim.datafiles import DPPC_GRO
        >>> import MDAnalysis as mda
        >>> u = mda.Universe(DPPC_GRO)
        >>> p = u.select_atoms('name P8')
        >>>
        >>> dppc  = u.select_atoms('resname DPPC')
        >>> water = u.select_atoms('resname SOL')
        >>>
        >>> box = u.dimensions[:3]
        >>> # here we want to use as 'surface' atoms only the P atoms
        >>> # and compute an intrinsic profile with respect to the
        >>> # interpolated surface.
        >>> # We need to distinguish the upper and lower leaflet by hand!
        >>> upper = p[p.positions[:,2]>box[2]/2.]
        >>> lower = p - upper
        >>>
        >>> # if symmetry=='planar', supply an upper and a lower group
        >>> # if symmetry=='generic', supply only group
        >>>
        >>> # here we need a large value of alpha (for pbc reconstruction
        >>> # of the triangulation used for the interpolation)
        >>> inter = pytim.SimpleInterface(u,symmetry='planar', upper=upper, lower=lower,alpha=5.)
        >>>
        >>> profile1 = pytim.observables.Profile(interface=inter)
        >>> profile2 = pytim.observables.Profile(interface=inter)
        >>> np.random.seed(1)
        >>> profile1.sample(dppc)
        >>> profile2.sample(water)
        >>>
        >>> lo,up,av1 = profile1.get_values(binwidth=.5)
        >>> lo,up,av2 = profile2.get_values(binwidth=.5)
        >>> np.set_printoptions(8)
        >>> print (av2[64:70])
        [0.04282182 0.04154116 0.04147182        inf 0.04651406 0.05234671]


        .. _MDAnalysis: http://www.mdanalysis.org/

    """

    def __init__(self,
                 universe,
                 group=None,
                 alpha=1.5,
                 symmetry='generic',
                 normal='z',
                 upper=None,
                 lower=None):

        self.symmetry = symmetry
        self.universe = universe
        self.group = group
        self.alpha = alpha
        self.upper = upper
        self.lower = lower
        emptyg = universe.atoms[0:0]
        if self.group is None:
            self.group = universe.atoms

        sanity = SanityCheck(self)
        sanity.assign_universe(universe, radii_dict=None, warnings=False)
        sanity.assign_alpha(alpha)
        sanity.assign_radii()
        if normal in [0, 1, 2]:
            self.normal = normal
        else:
            dirdict = {'x': 0, 'y': 1, 'z': 2, 'X': 0, 'Y': 1, 'Z': 2}
            self.normal = dirdict[normal]

        if self.symmetry == 'planar':
            if self.upper is None or self.lower is None:
                raise RuntimeError('cannot initialize a planar surface' +
                                   'without both the upper and lower groups')
            self._layers = np.empty(
                [2, 1], dtype=self.universe.atoms[0].__class__)
            self._layers[0, 0] = self.upper
            self._layers[1, 0] = self.lower
            for uplow in [0, 1]:
                self.label_group(self._layers[uplow][0], beta=1., layer=1)
            self.label_planar_sides()
            self._surfaces = np.empty(1, dtype=type(SurfaceFlatInterface))
            self._surfaces[0] = SurfaceFlatInterface(
                self, options={'layer': 0})

        else:
            self._layers = np.empty(
                [1, 1], dtype=self.universe.atoms[0].__class__)
            self._layers[0, 0] = self.group
            self.label_group(self._layers[0], beta=1., layer=1)
            self._surfaces = np.empty(1, dtype=type(SurfaceGenericInterface))
            self._surfaces[0] = SurfaceGenericInterface(
                self, options={'layer': 0})

    def _(self):
        """ additional tests
        >>> import pytim
        >>> import numpy as np
        >>> from pytim.datafiles import WATER_GRO
        >>> import MDAnalysis as mda
        >>> u = mda.Universe(WATER_GRO)
        >>>
        >>> u.atoms[:8].positions=np.array([[0.,0,5],[0,1,5],[1,0,5],[1,1,5],\
                [0.,0.,-5.],[0,1,-5],[1,0,-5],[1,1,-5]])
        >>> upper,lower=u.atoms[:4],u.atoms[4:8]
        >>> inter = pytim.SimpleInterface(u,symmetry='planar', upper=upper,\
                lower=lower,alpha=5.)
        >>> g = u.atoms[8:12]
        >>> g.atoms.positions=np.asarray([[.5,.5,6],[.5,.5,4],[.5,.5,-4],\
                [.5,.5,-6]])
        >>> print(pytim.observables.IntrinsicDistance(inter).compute(g))
        [ 1. -1. -1.  1.]



        """

    def _assign_layers(self):
        pass


#
