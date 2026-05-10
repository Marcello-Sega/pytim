# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: Profile
    ===============
"""
from __future__ import print_function
from .basic_observables import Number
from .intrinsic_distance import IntrinsicDistance
import numpy as np
import warnings
from scipy import stats
from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup


class Profile(object):
    r"""Calculates the profile (normal, or intrinsic) of a given observable
    across the simulation box.

    :param Observable observable:    :class:`Number <pytim.observables.Number>`,
                                     :class:`Mass <pytim.observables.Mass>`, or
                                     any other observable:
                                     calculate the profile of this quantity. If
                                     None is supplied, it defaults to the number
                                     density. The number density is always
                                     calculated on a per atom basis.
    :param str        direction:     'x','y', or 'z' : calculate the profile
                                     along this direction. (default: 'z' or
                                     the normal direction of the interface,
                                     if provided.
    :param ITIM       interface:     if provided, calculate the intrinsic
                                     profile with respect to the first layers
    :param bool       MCnorm:        if True (default) use a simple Monte Carlo
                                     estimate the effective volumes of the bins
    :param bool       reduced_units: if True, compute the histogram as a function
                                     of the position normalized by the instantanous
                                     box length. This can be used for NpT simulations.

    :Keyword Arguments:
        * MCpoints (int) --
          number of points used for MC normalization (default, 10x the number
          of atoms in the universe)

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
    >>> from   pytim.datafiles import LJ_GRO, LJ_SHORT_XTC
    >>> from   pytim.observables import Profile

    >>> u = mda.Universe(LJ_GRO,LJ_SHORT_XTC)
    >>>
    >>> inter = pytim.ITIM(u,alpha=2.5,cluster_cut=4.5)
    >>> profile = Profile(interface=inter)
    >>>
    >>> for ts in u.trajectory:
    ...     profile.sample(u.atoms)
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
                 observable=None,
                 direction=None,
                 interface=None,
                 symmetry='default',
                 mode='default',
                 reduced_units=False,
                 MCnorm=True,
                 keep_missing=False,
                 **kargs):

        _dir = {'x': 0, 'y': 1, 'z': 2}
        if direction is None:
            try:
                self._dir = interface.normal
            except:
                self._dir = 2
        else:
            self._dir = _dir[direction]
        self.mode = mode
        self.reduced_units = reduced_units
        self.interface = interface
        self._MCnorm = MCnorm
        self.keep_missing = keep_missing
        self.kargs = kargs
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

    def _determine_range(self, box):
        upper = np.min(box)
        if self.reduced_units:
            r = np.array([0.,1.])
        else:
            if self._MCnorm:
                upper = np.max(box)
                r = np.array([0., upper])
            if self._dir is not None:
                r = np.array([0., box[self._dir]])
            else:
                r = np.array([0., upper])

        if self.interface is not None:
            r -= r[1] / 2.
        self._range = r

    def _determine_bins(self):
        nbins = int((self._range[1] - self._range[0]) / self.binsize)
        # we need to make sure that the number of bins is odd, so that the
        # central one encompasses zero (to make the delta-function
        # contribution appear always in this bin)
        if (nbins % 2 > 0):
            nbins += 1
        self._nbins = nbins

    def _sample_random_distribution(self,
                                    group,
                                    upper_surface=None,
                                    lower_surface=None):
        box = group.universe.dimensions[:3]
        rnd_accum = np.array(0)
        try:
            size = self.kargs['MCpoints']
        except:
            # assume atomic volumes of ~ 30 A^3 and sample
            # 10 points per atomic volue as a rule of thumb
            size1 = int(np.prod(box) / 3.)
            # just in case 'unphysical' densities are used:
            size2 = 10 * len(group.universe.atoms)
            size = np.max([size1, size2])
        rnd = np.random.random((size, 3))
        rnd *= self.interface.universe.dimensions[:3]
        rnd_pos = IntrinsicDistance(
            self.interface, symmetry=self.symmetry).compute(
                rnd,
                upper_surface=upper_surface,
                lower_surface=lower_surface)
        # the interpolator can return NaNs in some cases
        rnd_pos = rnd_pos[np.isfinite(rnd_pos)]
        rnd_accum, bins, _ = stats.binned_statistic(
            rnd_pos,
            np.ones(len(rnd_pos)),
            range=self._range,
            statistic='sum',
            bins=self._nbins)
        return rnd_accum, bins

    def sample(self,
               group,
               interface = None,
               upper_surface=None,
               lower_surface=None,
               **kargs):
        # TODO: implement progressive averaging to handle very long trajs
        # TODO: implement memory cleanup
        if not isinstance(group, AtomGroup):
            raise TypeError("The first argument passed to "
                            "Profile.sample() must be an AtomGroup.")
        if interface is not None: self.interface = interface
        if upper_surface is not None or lower_surface is not None:
            if upper_surface is None or lower_surface is None:
                raise ValueError("upper_surface and lower_surface must be provided together")
            if interface.symmetry != "planar":
                raise ValueError("upper_surface/lower_surface are only supported for planar symmetry. "
                                 "Instantiate the interface with symmetry='planar'.")

        box = group.universe.trajectory.ts.dimensions[:3]

        if self._range is None:
            self._determine_range(box)
            self._determine_bins()

        if self.interface is None:
            pos = group.positions[::, self._dir]
        else:
            pos = IntrinsicDistance(
                self.interface, symmetry=self.symmetry, mode=self.mode).compute(
                    group,
                    upper_surface=upper_surface,
                    lower_surface=lower_surface)

            if self._MCnorm is False:
                rnd_accum = np.ones(self._nbins)

            else:
                rnd_accum, bins = self._sample_random_distribution(
                    group,
                    upper_surface=upper_surface,
                    lower_surface=lower_surface)

        values = self.observable.compute(group, **kargs)
        # the interpolator can return NaNs in some cases
        finite = np.isfinite(pos)
        n_total,n_finite = len(pos), np.count_nonzero(finite)
        cond = np.isfinite(pos)
        values, pos = values[cond], pos[cond]
        if n_finite != n_total:
            msg = f"Profile.sample(): interpolation failed for {n_total-n_finite} out of {n_total} atoms."
            if not self.keep_missing:
                warnings.warn(msg+' Skipping this frame (set keep_missing=True to override)', RuntimeWarning)
                return False
            else:
                warnings.warn(msg, RuntimeWarning)

        v = np.prod(box)
        if self.reduced_units:
            pos/= box[self._dir]
            v/= box[self._dir]
        accum, bins, _ = stats.binned_statistic(
            pos,
            values,
            range=tuple(self._range),
            statistic='sum',
            bins=self._nbins)
        accum[~np.isfinite(accum)] = 0.0

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
        self._totvol.append(v)
        self._counts += 1
        return True

    def get_values(self, binwidth=None, nbins=None, density=True):
        if self.sampled_values is None:
            print("Warning no profile sampled so far")
            return [None]*3

        max_bins = len(self.sampled_bins)
        max_size = max_bins * self.binsize

        if binwidth is not None:
            nbins = int(round(max_size / binwidth))
        if nbins is None:
            nbins = max_bins
        nbins = int(nbins)

        if nbins % 2 > 0:
            nbins += 1

        counts = self.sampled_values.copy()

        if self.interface is not None:
            fine_deltabin = int(1 + (self._nbins - 1) // 2)
            counts[fine_deltabin] = 0.0

        avg_counts, bins, _ = stats.binned_statistic(
            self.sampled_bins,
            counts,
            range=self._range,
            statistic="sum",
            bins=nbins,
        )

        avg_counts /= self._counts

        if not density:
            avg = avg_counts
        elif self.interface is None:
            bin_volume = np.average(self._totvol) / nbins
            avg = avg_counts / bin_volume
        else:
            mc_counts, _, _ = stats.binned_statistic(
                self.sampled_bins,
                self.sampled_rnd_values,
                range=self._range,
                statistic="sum",
                bins=nbins,
            )

            total_mc = np.sum(self.sampled_rnd_values)
            avg_vol = np.average(self._totvol) * mc_counts / total_mc

            avg = np.zeros_like(avg_counts)
            mask = avg_vol > 0.0
            avg[mask] = avg_counts[mask] / avg_vol[mask]

            out_deltabin = int(1 + (nbins - 1) // 2)
            avg[out_deltabin] = np.inf

        return bins[:-1], bins[1:], avg


    @staticmethod
    def _():
        """
        >>> # this doctest checks that the same profile is
        >>> # obtained after rotating the system, and that
        >>> # it is consistent through versions
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
        >>> np.random.seed(1) # for the MC normalization
        >>> stdprof = pytim.observables.Profile()
        >>> stdprof.sample(u.atoms)
        >>> vals = stdprof.get_values(binwidth=0.5)[2]
        >>> print(np.around(vals[:6],decimals=3))
        [0.092 0.11  0.081 0.11  0.098 0.098]


        >>> prof = pytim.observables.Profile(interface=inter)
        >>> prof.sample(u.atoms)
        >>> vals = prof.get_values(binwidth=0.5)[2]
        >>> print(np.around(vals[len(vals)//2-3:len(vals)//2+3],decimals=3))
        [0.073 0.043 0.028   inf 0.    0.   ]



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

        >>> # We check now the profile computed with GITIM
        >>> u = mda.Universe(WATERSMALL_GRO)
        >>> g = u.select_atoms('name OW')
        >>> inter = pytim.GITIM(u,group=g,alpha=2.5)
        >>> print(inter.normal)
        None

        >>> np.random.seed(1) # for the MC normalization
        >>> stdprof = pytim.observables.Profile()
        >>> stdprof.sample(u.atoms)
        >>> vals = stdprof.get_values(binwidth=0.5)[2]
        >>> print(np.around(vals[:6],decimals=3))
        [0.092 0.11  0.081 0.11  0.098 0.098]

        >>> prof = pytim.observables.Profile(interface=inter)
        >>> prof.sample(u.atoms)
        >>> vals = prof.get_values(binwidth=1.0)[2]
        >>> print(np.around(vals[len(vals)//2-4:len(vals)//2+2],decimals=3))
        [0.096 0.098 0.056 0.      inf 0.   ]

        """

        pass
