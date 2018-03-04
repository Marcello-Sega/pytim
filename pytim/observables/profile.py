# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: Profile
    ===============
"""
from __future__ import print_function
from .basic_observables import Number
from .intrinsic_distance import IntrinsicDistance
import numpy as np
from scipy import stats
from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup


class Profile(object):
    """Calculates the profile (normal, or intrinsic) of a given observable
    across the simulation box.

    :param Observable observable:   :class:`Number <pytim.observables.Number>`,
                                    :class:`Mass <pytim.observables.Mass>`, or
                                    any other observable:
                                    calculate the profile of this quantity. If
                                    None is supplied, it defaults to the number
                                    density. The number density is always
                                    calculated on a per atom basis.
    :param ITIM       interface:    if provided, calculate the intrinsic
                                    profile with respect to the first layers
    :param str        direction:    'x','y', or 'z' : calculate the profile
                                    along this direction. (default: 'z' or
                                    the normal direction of the interface,
                                    if provided.
    :param bool       MCnorm:       if True (default) use a simple Monte Carlo
                                    estimate the effective volumes of the bins.

    Example (non-intrinsic, total profile + first 4 layers ):

    >>> import numpy as np
    >>> import MDAnalysis as mda
    >>> import pytim
    >>> from   pytim.datafiles import *
    >>> from   pytim.observables import Profile
    >>>
    >>> u = mda.Universe(WATER_GRO,WATER_XTC)
    >>> g = u.select_atoms('name OW')
    >>> # here we calculate the profiles of oxygens only (note molecular=False)
    >>> inter = pytim.ITIM(u,group=g,max_layers=4,centered=True, molecular=False)
    >>>
    >>> # We create a list of 5 profiles, one for the total and 4 for the first
    >>> # 4 layers.
    >>> # Note that by default Profile() uses the number of atoms as an observable
    >>> Layers = []
    >>> for n in range(5):
    ...     Layers.append(Profile())
    >>>
    >>> # Go through the trajectory, center the liquid slab and sample the profiles
    >>> for ts in u.trajectory[::50]:
    ...         # this shifts the system so that the center of mass of the liquid slab
    ...         # is in the middle of the box
    ...         inter.center()
    ...
    ...         Layers[0].sample(g)
    ...         Layers[1].sample(u.atoms[u.atoms.layers == 1 ])
    ...         Layers[2].sample(u.atoms[u.atoms.layers == 2 ])
    ...         Layers[3].sample(u.atoms[u.atoms.layers == 3 ])
    ...         Layers[4].sample(u.atoms[u.atoms.layers == 4 ])
    >>>
    >>> density=[]
    >>> for L in Layers:
    ...     low,up,avg = L.get_values(binwidth=0.5)
    ...     density.append(avg)
    >>>
    >>> # (low + up )/2 is the middle of the bin
    >>> np.savetxt('profile.dat',list(zip(low,up,density[0],density[1],density[2],density[3],density[4])))

    This results in the following profile (sampling more often and zooming close to the interface border)

    .. image:: nonintrinsic_water.png
        :width: 50%

    Example: the intrinsic profile of a LJ liquid/vapour interface:

    >>> import numpy as np
    >>> import MDAnalysis as mda
    >>> import pytim
    >>> from   pytim.datafiles import LJ_GRO
    >>> from   pytim.observables import Profile

    >>> u = mda.Universe(LJ_GRO,LJ_SHORT_XTC)
    >>> g = u.select_atoms('all')
    >>>
    >>> inter = pytim.ITIM(u,alpha=2.5,cluster_cut=4.5)
    >>> profile = Profile(interface=inter)
    >>>
    >>> for ts in u.trajectory:
    ...     profile.sample(g)
    >>>
    >>> low, up, avg = profile.get_values(binwidth=0.5)
    >>> np.savetxt('profile.dat',list(zip(low,up,avg)))


    This results in the following profile (sampling a longer trajectory):

    .. image:: intrinsic_lj.png
        :width: 50%

    Note the missing point at position = 0, this is the delta-function contirbution.
    Negative positions are within the liquid phase, while positive ones are in the vapour
    phase.

    """

    def __init__(self,
                 direction=None,
                 observable=None,
                 interface=None,
                 symmetry='default',
                 MCnorm=True):

        _dir = {'x': 0, 'y': 1, 'z': 2}
        if direction is None:
            try:
                self._dir = interface.normal
            except:
                self._dir = 2
        else:
            self._dir = _dir[direction]
        self.interface = interface
        self._MCnorm = MCnorm
        if symmetry == 'default' and interface is not None:
            self.symmetry = self.interface.symmetry
        else:
            self.symmetry = symmetry

        if observable is None:
            self.observable = Number()
        else:
            self.observable = observable
        self.binsize = 0.01  # this is used for internal calculations, the
        # output binsize can be specified in
        # self.get_values()

        self.sampled_bins = None
        self.sampled_values = None
        self._range = None
        self._counts = 0
        self._totvol = []

    def sample(self, group):
        # TODO: implement progressive averaging to handle very long trajs
        # TODO: implement memory cleanup
        if not isinstance(group, AtomGroup):
            raise TypeError("The first argument passed to "
                            "Profile.sample() must be an AtomGroup.")

        box = group.universe.trajectory.ts.dimensions[:3]
        if self._range is None:
            _range = [0., box[self._dir]]
            nbins = int(box[self._dir] / self.binsize)
            # we need to make sure that the number of bins is odd, so that the
            # central one encompasses zero (to make the delta-function
            # contribution appear always in this bin)
            if (nbins % 2 > 0):
                nbins += 1
            self._nbins = nbins
            if self.interface is not None:
                _range -= box[self._dir] / 2.
            self._range = _range
        v = np.prod(box)
        self._totvol.append(v)

        if self.interface is None:
            pos = group.positions[::, self._dir]
        else:
            deltabin = 1 + (self._nbins - 1) // 2
            pos = IntrinsicDistance(
                self.interface, symmetry=self.symmetry).compute(group)

            if self._MCnorm is False:
                rnd_accum = np.ones(self._nbins)

            else:
                rnd_accum = np.array(0)
                size = 5 * len(group.universe.atoms)
                rnd = np.random.random((size, 3))
                rnd *= self.interface.universe.dimensions[:3]
                rnd_pos = IntrinsicDistance(
                    self.interface, symmetry=self.symmetry).compute(rnd)
                rnd_accum, bins, _ = stats.binned_statistic(
                    rnd_pos,
                    np.ones(len(rnd_pos)),
                    range=self._range,
                    statistic='sum',
                    bins=self._nbins)

        values = self.observable.compute(group)
        accum, bins, _ = stats.binned_statistic(
            pos, values, range=self._range, statistic='sum', bins=self._nbins)
        accum[~np.isfinite(accum)] = 0.0
        if self.interface is not None:
            accum[deltabin] = np.inf

        if self.sampled_values is None:
            self.sampled_values = accum.copy()
            if self.interface is not None:
                self.sampled_rnd_values = rnd_accum.copy()
            # stores the midpoints
            self.sampled_bins = bins[1:] - self.binsize / 2.
        else:
            self.sampled_values += accum
            if self.interface is not None:
                self.sampled_rnd_values += rnd_accum
        self._counts += 1

    def get_values(self, binwidth=None, nbins=None):
        if self.sampled_values is None:
            print("Warning no profile sampled so far")
        # we use the largest box (largest number of bins) as reference.
        # Statistics will be poor at the boundaries, but like that we don't
        # loose information
        max_bins = len(self.sampled_bins)
        max_size = max_bins * self.binsize

        if binwidth is not None:  # overrides nbins
            nbins = max_size / binwidth
        if nbins is None:  # means also binwidth must be none
            nbins = max_bins

        if (nbins % 2 > 0):
            nbins += 1

        vals = self.sampled_values.copy()

        if self.interface is not None:
            _vol = self.sampled_rnd_values * np.average(self._totvol)
            _vol /= np.sum(self.sampled_rnd_values)
            deltabin = np.where(~np.isfinite(vals))[0]
            vals[deltabin] = 0.0
            vals[_vol > 0] /= _vol[_vol > 0]
            vals[_vol <= 0] *= 0.0
            vals[deltabin] = np.inf
        else:
            vals /= (np.average(self._totvol) / self._nbins)

        vals /= self._counts

        avg, bins, _ = stats.binned_statistic(
            self.sampled_bins,
            vals,
            range=self._range,
            statistic='mean',
            bins=nbins)
        return [bins[0:-1], bins[1:], avg]

    @staticmethod
    def _():
        """
        >>> # this doctest checks that the same profile is
        >>> # obtained after rotating the system
        >>> import MDAnalysis as mda
        >>> import numpy as np
        >>> import pytim
        >>> pytim.observables.Profile._()
        >>> from pytim.datafiles import WATERSMALL_GRO
        >>> from matplotlib import pyplot as plt
        >>> u = mda.Universe(WATERSMALL_GRO)
        >>> inter = pytim.ITIM(u,cluster_cut=3.5,alpha=2.5)
        >>> print(inter.normal)
        2

        >>> prof = pytim.observables.Profile(interface=inter)
        >>> prof.sample(u.atoms)
        >>> sv = prof.sampled_values
        >>> u.atoms.positions=np.roll(u.atoms.positions,1,axis=1)
        >>> box = u.dimensions[:]
        >>> box[0]=box[2]
        >>> box[2]=box[1]
        >>> u.dimensions = box
        >>> inter = pytim.ITIM(u,cluster_cut=3.5,alpha=2.5)
        >>> print(inter.normal)
        0

        >>> prof = pytim.observables.Profile(interface=inter)
        >>> prof.sample(u.atoms)
        >>> sv2 = prof.sampled_values
        >>> print(np.all(sv==sv2))
        True

        """
        pass
