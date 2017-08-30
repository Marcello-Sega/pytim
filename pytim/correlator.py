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
        >>> print corr
        [ 1.          0.2854253   0.18059321  0.19296861  0.42821504  0.22823323
          0.32167236  0.19506167  0.34336377  0.18111761  0.05521241  0.15146141
          0.3059723   0.03549702 -0.06320951  0.08207338 -0.02442951 -0.01383818
         -0.04637258  0.0533662  -0.10312512 -0.0085912  -0.37878377 -0.26352859
         -0.06200694  0.01444449 -0.44058268 -0.36078218 -0.35199886 -0.16273729
         -0.24969988 -0.55350561 -0.3740507  -0.01228043 -0.67140082 -0.78662433
         -0.28146374 -0.37563115 -0.68283012 -0.70017332 -0.48424531 -0.56197533
         -0.65147349 -0.7446905  -0.16783918 -0.43809782 -2.04122294 -1.25494069
          0.2705082   5.35673624]



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

    size = len(a1)
    norm = np.arange(size)[::-1] + 1
    fa1 = np.fft.fft(a1,size*2)

    if not isinstance(a2, type(None)):  # do cross-corr
        fa2 = np.fft.fft(a2,size*2)
        return ( (np.fft.fft(fa2 * np.conj(fa1))[:size] + np.fft.fft(fa1 * np.conj(fa2))[:size]).real.T / norm).T / len(fa1) / 2.
    else:                             # do auto-corr
        return ( (np.fft.fft(fa1 * np.conj(fa1))[:size] ).real.T / norm).T / len(fa1) 


