# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: FreeVolume
    ==================
"""
from __future__ import print_function
import numpy as np
from scipy.spatial import cKDTree
from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup


class FreeVolume(object):
    """ Calculates the fraction of free volume in the system, or its profile.

        Note that this does not fit in the usual observable class as it can not be
        expressed as a property of particles, and needs some kind of gridding to be calculated.

        :param Universe universe:  the universe
        :param int npoints: number of Monte Carlo sampling points (default: 10x the number of atoms in the universe)

        Returns:
        :(free volume, error) tuple : A tuple with the free volume and the estimated error

        Examples:
        >>> import MDAnalysis as mda
        >>> import numpy as np
        >>> import pytim
        >>> from pytim.datafiles import CCL4_WATER_GRO, _TEST_BCC_GRO

        >>> u = mda.Universe(CCL4_WATER_GRO)
        >>> inter = pytim.ITIM(u) # just to make sure all radii are set
        >>> np.random.seed(1) # ensure reproducibility of test
        >>> FV = pytim.observables.FreeVolume(u)
        >>> bins, prof ,err = FV.compute_profile()

        >>> free, err = FV.compute()
        >>> print ('{:0.3f} +/- {:0.3f}'.format(free,err))
        0.431 +/- 0.001

        >>> # strict test on bcc volume fraction
        >>> u = mda.Universe(_TEST_BCC_GRO)
        >>> # we add some random gitter to avoid singular matrices
        >>> u.atoms.positions += np.random.random(u.atoms.positions.shape)*1e-5
        >>> inter = pytim.GITIM(u,radii_dict={'C':10.*np.sqrt(3.)/4.})
        >>> nsamples = int(1e5)
        >>> FV = pytim.observables.FreeVolume(u,npoints = nsamples)
        >>> np.random.seed(1) # ensure reproducibility of test
        >>> free, err = FV.compute()
        >>> np.isclose(free,1.0-0.6802,rtol=1e-3)
        True
        >>> np.random.seed(1) # ensure reproducibility of test
        >>> lst, _ = FV._compute()
        >>> np.isclose(free,1.0-len(lst)*1.0/nsamples, rtol=1e-6)
        True

    """

    def __init__(self, universe, npoints=None):
        self.u = universe
        if npoints is None:
            npoints = 10 * len(universe.atoms)
        self.npoints = npoints

    def _compute(self, inp=None):
        res = np.array(0)
        _box = self.u.dimensions.copy()
        box = _box[:3]
        try:  # older scipy versions
            tree = cKDTree(
                np.random.random((self.npoints, 3)) * box, boxsize=_box[:6])
        except:
            tree = cKDTree(
                np.random.random((self.npoints, 3)) * box, boxsize=_box[:3])
        if inp is None:
            inp = self.u.atoms
        if not isinstance(inp, AtomGroup):
            raise RuntimeError(self.__class__.__name__ +
                               'compute needs AtomGroup as an input')
        # np.unique here avoids counting contributions from overlapping spheres
        radii = np.unique(inp.radii)

        for radius in radii:
            where = np.where(np.isclose(inp.radii, radius))
            lst = [
                e for l in tree.query_ball_point(inp.positions[where], radius)
                for e in l
            ]
            res = np.append(res, lst)
        return np.unique(res), tree.data

    def compute_profile(self, inp=None, nbins=30, direction=2):
        """ Compute a profile of the free volume fraction

            :param AtomGroup inp:  compute the volume fraction of this group, None selects the complete universe
            :param int nbins: number of bins, by default 30
            :param int direction: direction along wich to compute the the profile, z (2) by default

            :returns bins,fraction,error: the left limit of the bins, the free volume fraction in each bin, the associated std deviation
        """
        box = self.u.dimensions[:3].copy()

        slabwidth = box[direction] / nbins

        bins = np.arange(nbins + 1) * slabwidth

        histo = []
        error = []
        res, data = self._compute(inp)
        for i in range(nbins):
            condition = np.logical_and(data[:, direction] >= bins[i],
                                       data[:, direction] < bins[i + 1])
            in_slab = np.where(condition)[0]
            n_in_slab = np.sum(condition * 1.0)
            if n_in_slab == 0:
                histo.append(0.0)
                error.append(0.0)
            else:
                ratio = np.sum(np.isin(res, in_slab) * 1.0) / \
                    n_in_slab  # occupied volume
                histo.append(1. - ratio)
                error.append(np.sqrt(ratio * (1. - ratio) / n_in_slab))
        return bins, np.array(histo), np.array(error)

    def compute(self, inp=None):
        """ Compute the total free volume fraction in the simulation box

            :param AtomGroup inp:  compute the volume fraction of this group, None selects the complete universe
            :param int nbins: number of bins, by default 30

            :returns fraction, error: the free volume fraction and associated error

        """
        _, free, err = self.compute_profile(inp, nbins=1)
        return free[0], err[0]
