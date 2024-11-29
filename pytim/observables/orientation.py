# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: Orientation
    ===================
"""

from __future__ import print_function
from . import Observable
import numpy as np
from MDAnalysis.core.groups import Atom, AtomGroup


class Orientation(Observable):
    """Orientation of a group of points.

    :param str options: optional string. If `normal` is passed, the
                        orientation of the normal vectors is computed
                        If the option 'molecular' is passed at initialization
                        the coordinates of the second and third atoms are
                        folded around those of the first.

    """

    def __init__(self, universe, options=''):
        self.u = universe
        self.options = options

    def compute(self, inp, kargs=None):
        """Compute the observable.

        :param ndarray inp:  the input atom group. The length must be a multiple
                             of three
        :returns: the orientation vectors

        For each triplet of positions A1,A2,A3, computes the unit vector
        between A2-A1 and  A3-A1 or, if the option 'normal' is passed at
        initialization, the unit vector normal to the plane spanned by the
        three vectors.


        """

        if isinstance(inp, AtomGroup) and len(inp) != 3 * len(inp.residues):
            inp = inp.residues
        if 'molecular' in self.options:
            pos = self.fold_around_first_atom_in_residue(inp)
        else:
            pos = inp.positions
        flat = pos.flatten()
        pos = flat.reshape(len(flat) // 3, 3)
        a = pos[1::3] - pos[0::3]
        b = pos[2::3] - pos[0::3]

        if 'normal' in self.options:
            v = np.cross(a, b)
        else:
            v = np.array(a + b)
        v = np.array([el / np.sqrt(np.dot(el, el)) for el in v])
        return v

class BivariateAngles(Observable):
    """ Two angles characterizing the orientation next to a surface of an axisymmetric molecule
        like water.

        The angles are computed according to Jedlovszky et al., Phys. Chem. Chem. Phys., 2004, 6, 1874–1879

        :param: bool molecular:  if True, uses the option "molecular=True" in the calculation of the
                                 molecular axes (using pytim.observables.Orientation)
    """

    def __init__(self, universe, molecular=False):
        options=['']
        if molecular: options=['molecular']
        self.molecular_plane_vector  = Orientation(universe,options=options+['normal'])
        self.dipole_vector = Orientation(universe,options=options)

    def compute(self, inp, **kargs):
        """Compute the observable.

        :param ndarray inp: the input atom group. The length must
                be a multiple of three
        :param **kwargs:   See below

        :Keyword Arguments:
          * *normal* (``int`` or ``str``)  -- Overrides the surface
            normal as determined by pytim via one of its methods. Use
            it when this is not available. Default: [0,0,1]

        :returns: two arrays, each the same length as the input
          group, one with the cosine of the angle between
          the molecular symmetry axis and the surface normal,
          and the the second with the angle (in rad) between
          the projection of the surface normal onto the
          plane defined by the molecular symmetry axiss and
          the molecular plane normal.

        Example:

        >>> import MDAnalysis as mda
        >>> import numpy as np
        >>> import pytim
        >>> from pytim.datafiles import WATER_GRO
        >>> from pytim.observables import BivariateAngles
        >>>
        >>> u = mda.Universe(WATER_GRO)
        >>> g = u.select_atoms('name OW')
        >>> inter = pytim.ITIM(g, cluster_cut=3.5, molecular=True)
        >>> biv = BivariateAngles(u,molecular=True)
        >>> condition = np.logical_and(u.atoms.sides==0,u.atoms.layers==1)
        >>> group = u.atoms[condition]
        >>> costheta, phi = biv.compute(group)
        >>> print(all(np.isclose([costheta[0],phi[0]], [0.6533759236335754, 0.10778185716460659])))
        True
    """

        v_normal = self.molecular_plane_vector.compute(inp)
        v_dipole = self.dipole_vector.compute(inp)
        try:
            v_surface = np.asarray(kargs['normal'])
        except:
            try:
                v_surface = u.inter.normal
            except:
                v_surface = np.asarray([0.,0.,1.])

        costheta = np.dot(v_dipole, v_surface)
        # the surface normal projected on the plane with normal v_dipole
        v = (v_surface-v_dipole*costheta[:,np.newaxis])
        v = v / np.linalg.norm(v,axis=1)[:,np.newaxis]
        normal_component = np.sum(v*v_normal,axis=1)
        # avoid numerical errors
        normal_component[normal_component>1.0]=1.0
        normal_component[normal_component<-1.0]=-1.0
        orthogonal_component = np.sqrt(1-normal_component**2)
        phi = np.mod(np.arctan2(normal_component, orthogonal_component),np.pi/2)
        return costheta,phi
