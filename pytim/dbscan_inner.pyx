# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Fast inner loop for DBSCAN.
# Author: Lars Buitinck
# License: 3-clause BSD

# slightly modified to provide cluster sizes

cimport cython
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
cimport numpy as np
import numpy as np


# Work around Cython bug: C++ exceptions are not caught unless thrown within
# a cdef function with an "except +" declaration.
cdef inline void push(vector[np.npy_intp] &stack, np.npy_intp i) except +:
    stack.push_back(i)


@cython.boundscheck(False)
@cython.wraparound(False)
def dbscan_inner(np.ndarray[np.uint8_t, ndim=1, mode='c'] is_core,
                 np.ndarray[object, ndim=1] neighborhoods,
                 np.ndarray[np.npy_intp, ndim=1, mode='c'] labels,
                 np.ndarray[np.npy_intp, ndim=1, mode='c'] counts):
    cdef np.npy_intp i, label_num = 0, v
    cdef np.ndarray[np.npy_intp, ndim=1] neighb
    cdef vector[np.npy_intp] stack
    cdef cset[np.npy_intp] seen

    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points.
        # This is very similar to the classic algorithm for computing connected
        # components, the difference being that we label non-core points as
        # part of a cluster (component), but don't expand their neighborhoods.
        while True:
            if labels[i] == -1:
                labels[i] = label_num
                counts[label_num] += 1 
                if is_core[i]:
                    neighb = neighborhoods[i]
                    for i in range(neighb.shape[0]):
                        v = neighb[i]
                        if labels[v] == -1 and seen.count(v) == 0:
                            seen.insert(v)
                            push(stack, v)

            if stack.size() == 0:
                break
            i = stack.back()
            stack.pop_back()

        seen.clear()
	
        label_num += 1

