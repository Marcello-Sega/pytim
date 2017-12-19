# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: correlator
    ==================
"""
from __future__ import print_function

import numpy as np
from pytim import utilities
from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup


class Correlator(object):
    """ Computes the (self) correlation of an observable (scalar or vector)

    :param Observable observable: compute the autocorrelation of this observable
    :param bool reduced: when the observable is a vector, average over all spatial direction if reduced==True (default)
    :param bool normalize: normalize the correlation to 1 at t=0
    :param AtomGroup reference: if the group passed to the sample() function changes its composition along the trajectory \
                                (such as a layer group), a reference group that includes all atoms that could appear in the \
                                variable group must be passed, in order to provide a proper normalization. This follows the \
                                convention in J. Phys. Chem. B 2017, 121, 5582-5594, (DOI: 10.1021/acs.jpcb.7b02220). See\
                                the example below.
    :param double memory_warn: if not None, print a warning once this threshold of memory (in Mb) is passed.


    Example:

    >>> import pytim
    >>> import MDAnalysis as mda
    >>> import numpy as np
    >>> from pytim.datafiles import WATERSMALL_GRO
    >>> from pytim.utilities import lap
    >>> #  tmpdir here is specified only for travis
    >>> import  os
    >>> WATERSMALL_TRR = pytim.datafiles.pytim_data.fetch('WATERSMALL_LONG_TRR',tmpdir='./')
    checking presence of a cached copy... not found. Fetching remote file... done.

    >>> u = mda.Universe(WATERSMALL_GRO,WATERSMALL_TRR)
    >>> g = u.select_atoms('name OW')

    >>> velocity = pytim.observables.Velocity()
    >>> corr = pytim.observables.Correlator(observable=velocity)
    >>> for t in u.trajectory[1:]:
    ...     corr.sample(g)
    >>> vacf = corr.correlation()

    This produces the following (steps of 1 fs):

    .. plot::

        import pytim
        import MDAnalysis as mda
        import numpy as np
        from pytim.datafiles import WATERSMALL_GRO
        from pytim.utilities import lap
        WATERSMALL_TRR=pytim.datafiles.pytim_data.fetch('WATERSMALL_LONG_TRR')

        u = mda.Universe(WATERSMALL_GRO,WATERSMALL_TRR)
        g = u.select_atoms('name OW')

        velocity = pytim.observables.Velocity()
        corr = pytim.observables.Correlator(observable=velocity)
        for t in u.trajectory[1:]:
            corr.sample(g)

        vacf = corr.correlation()

        from matplotlib import pyplot as plt
        plt.plot(vacf[:1000])
        plt.plot([0]*1000)
        plt.show()


    In order to compute the correlation for variable groups, one should proceed
    as follows:

    >>> corr = pytim.observables.Correlator(observable=velocity,reference=g)
    >>> # notice the molecular=False switch, in order for the
    >>> # layer group to be made of oxygen atoms only and match
    >>> # the reference group
    >>> inter = pytim.ITIM(u,group=g,alpha=2.0,molecular=False)
    >>> for t in u.trajectory[1:10]: # example only: sample the whole trajectory
    ...     corr.sample(inter.atoms)
    >>> os.unlink('./'+WATERSMALL_TRR) # cleanup
    >>> layer_vacf = corr.correlation()


    """

    def __init__(self, universe=None, observable=None, reduced=True, normalize=True, reference=None, memory_warn=None):
        pass  # wrong implementation
        name = self.__class__.__name__
        self.observable = observable
        self.reference = reference
        self.timeseries = []
        self.maskseries = []
        self.reduced = reduced
        self.normalize = normalize
        self.shape = None
        self.mem_usage = 0.0
        self.warned = False
        if memory_warn is None:
            self.warned = True
            self.memory_warn = 0.0
        else:
            self.memory_warn = memory_warn
        if reference is not None:
            if isinstance(reference, Atom):
                # in case just a single atom has been passed,
                # and not a group with one atom
                self.reference = u.atoms[reference.index:reference.index + 1]
            if not isinstance(reference, AtomGroup):
                raise RuntimeError(
                    name + ': reference must be eiter an Atom or an AtomGroup')
            self.reference_obs = observable.compute(reference) * 0.0
            if len(self.reference_obs.shape) > 2:
                raise RuntimeError(
                    name + ' works only with scalar and vectors')

    def sample(self, group):
        if self.reference is not None:
            sampled = self.reference_obs.copy()
            mask = np.isin(self.reference, group)
            self.maskseries.append(list(mask))
            obs = self.observable.compute(group)
            sampled[mask] = obs
        else:
            sampled = self.observable.compute(group)

        self.timeseries.append(list(sampled.flatten()))
        self.mem_usage += sampled.nbytes / 1024.0 / 1024.0  # in Mb
        if self.mem_usage > self.memory_warn and self.warned == False:
            print("Warning: warning threshold of", end=' ')
            print(self.memory_warn, "Mb exceeded")
            self.warned = True

        if self.shape == None:
            self.shape = sampled.shape

    def correlation(self):
        corr = utilities.correlate(self.timeseries)

        if self.reference is not None:
            mask = utilities.correlate(self.maskseries)
            if self.shape[1] > 1:
                tmp_mask = mask.copy()
                for i in range(1, self.shape[1]):
                    mask = np.hstack((mask, tmp_mask))
            corr[mask > 0] = corr[mask > 0] / mask[mask > 0]
        if self.reduced == True:
            corr = np.sum(corr, axis=1)
        if self.normalize == True:
            corr = corr / corr[0]
        return corr
