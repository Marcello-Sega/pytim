#!/usr/bin/python
import numpy as np
from timeit import default_timer as timer

class Correlator(object):
    """ Correlation class for 1D data series using
        numpy fft. The class calculates correlation
        or cross-correlation.

        :param numpy.array a1:   first data set to correlate
        :param numpy.array a2:   second data set to correlate


        :attribute str mode:     query the mode for autocorrelation or crosscorrelation
    """

    #TODO: error handling is non-existent
    #TODO: add different methods for correlation (direct, fft)
    def __init__(self, a1=np.ndarray(0), a2=None):
        self.tic = timer()

        self.a1 = a1
        self.a2 = a2
        if self.a2 is not None:
            self.mode = 'cross-correlation'
        else:
            self.mode = 'auto-correlation'


    def append(self):
        #TODO: append to list is faster than append to numpy.ndarray
        #TODO: need typechecking
        pass


    def correlate(self):
        """ Calculate the symmetrized, averaged and normalized correlation
            on dataset a1,a2 or autocorrelation on a1.

            Example:
            >>> a = np.array([1,0,1,0,1])
            >>> b = np.array([0,2,0,1,0])
            >>> c = Correlator(b,a)
            >>> print c.mode
            cross-correlation

            >>> print c.correlate()
            [-0.    0.75  0.    0.75  0.  ]

            >>> c = Correlator(b)
            >>> print c.mode
            auto-correlation

            >>> print c.correlate(b)
            [ 1.      0.      0.6667 -0.      0.    ]

        """

        #TODO: is there a faster way?
        size=len(self.a1)
        norm = np.arange(size)[::-1]+1
        self.a1 = np.append(self.a1,self.a1[:]*0)  # zero padding
        fa1 = np.fft.fft(self.a1)

        if not isinstance(self.a2, type(None)): # do cross-corr
            self.a2 = np.append(self.a2,self.a2[:]*0)
            fa2 = np.fft.fft(self.a2)
        else:                             # do auto-corr
            self.a2 = self.a1             # pointer to data in a1
            fa2     = fa1

        # Doing fft on a1 and a2
        return np.real(  (np.fft.fft(fa2*np.conj(fa1))[:size] + np.fft.fft(fa1*np.conj(fa2))[:size]) / norm / len(fa1) ) / 2.


    def lap(self):
        """ Return time elapsed after last invocation.
            On first invocation, it is the time elapsed
            since __init__

            Example:

            >>> from time import sleep
        >>> c2 = Correlator()
        >>> sleep(2)
        >>> print c2.lap()
        2.00517892838

        >>> c2 = Correlator()
        >>> sleep(2)
        >>> c2.lap()
        >>> sleep(3)
        >>> print c2.lap()
        3.00523400307

    """
    toc=timer()
    dt = toc-self.tic
    self.tic=toc
    return dt


if __name__ == "__main__":
    a = np.array([1,1,1,3,1])
    b = np.array([3,2,5,1,4])
    print 'a:', a
    print 'b:', b
    c1 = Correlator(b,a)
    c2 = Correlator(b)

    print c1.mode
    print np.round(c1.correlate(),4)
    print c2.mode
    print np.round(c2.correlate(),4)
