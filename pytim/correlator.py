# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#!/usr/bin/python
import numpy as np


def correlate(a1=np.ndarray(0), a2=None):
    """ correlate 1D data series using numpy fft. The function calculates correlation
        or cross-correlation.

        :param ndarray a1:   first data set to correlate
        :param ndarray a2:   (optional) second data set, to compute cross-correlation

        Example: time autocorrelation of the number of atoms in the outermost layer

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim import observables
        >>> from pytim.datafiles import *
        >>>
        >>> u = mda.Universe(WATER_GRO,WATER_XTC)
        >>> inter = pytim.ITIM(u)
        >>>
        >>> size=[]
        >>> time=[]
        >>> # sample the size of the first layer on the upper
        >>> # side
        >>> for ts in u.trajectory[:50]:
        ...     time.append(ts.time)
        ...     size.append(len(inter.layers[0,0]))
        >>>
        >>> # we need to subtract the average value
        >>> corr =  observables.correlate(size-np.mean(size)) 
        >>> corr = corr/corr[0] # normalize to 1


        This will produce (sampling the whole trajectory), the following:

        .. plot::

            import numpy as np
            import MDAnalysis as mda
            import pytim
            from   pytim.datafiles import *
            from   pytim import observables
            from matplotlib import pyplot as plt

            u = mda.Universe(WATER_GRO,WATER_XTC)
            inter = pytim.ITIM(u)

            size=[]
            time=[]
            for ts in u.trajectory[:]:
                time.append(ts.time)
                size.append(len(inter.layers[0,0]))

            corr =  observables.correlate(size-np.mean(size))
            plt.plot(time,corr/corr[0])
            plt.plot(time,[0]*len(time))
            plt.gca().set_xlabel("time/ps")

            plt.show()

    """

    # TODO: is there a faster way?
    size = len(a1)
    norm = np.arange(size)[::-1] + 1
    a1 = np.append(a1, a1[:] * 0)  # zero padding
    fa1 = np.fft.fft(a1)

    if not isinstance(a2, type(None)):  # do cross-corr
        a2 = np.append(a2, a2[:] * 0)
        fa2 = np.fft.fft(a2)
    else:                             # do auto-corr
        a2 = a1             # pointer to data in a1
        fa2 = fa1

    # Doing fft on a1 and a2
    return np.real((np.fft.fft(fa2 * np.conj(fa1))[:size] + np.fft.fft(fa1 * np.conj(fa2))[:size]) / norm / len(fa1)) / 2.
