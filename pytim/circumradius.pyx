#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

cimport cython
from cython.view cimport array as cvarray
cimport scipy.linalg.cython_lapack as cython_lapack
from libc.math cimport sqrt 
    
cimport numpy as np 
import numpy as np
    

ctypedef np.float64_t npreal

cdef _solve(double * M, double * b, int * info):
    """ solves M x = b for 2 rhs vectors """ 
    cdef int n = 3
    cdef int nrhs = 2 
    cdef int ipiv[3]
    cython_lapack.dgesv(&n,&nrhs,M,&n,ipiv,b,&n,info)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
cdef  _circumradius(npreal [:,:] points, npreal [:] radii, int [:,:] simplices ):
    cdef size_t i, j,pid,simplex
    cdef int info
    cdef npreal arg
    cdef npreal r[4][3]
    cdef npreal rad[4]
    cdef npreal rad2[4]
    cdef npreal s[3]
    cdef npreal ds[2][3]
    cdef npreal M[3][3]
    cdef npreal d[3]
    cdef npreal r2[4]
    cdef npreal vi,u2,v2,uv,R1,R2,A,B,C,R_1,R_2
    R = np.zeros(simplices.shape[0])
    cdef npreal[:] Rv = R
            
    for simplex in range(simplices.shape[0]):
        r2[0]=r2[1]=r2[2]=r2[3]=0.0
        for i in range(4):
            pid = simplices[simplex,i]
            rad[i]  = radii[pid]
            rad2[i] = rad[i]*rad[i]
            for j in range(3):
                r[i][j] = points[pid,j]
                r2[i]   = r2[i] + r[i][j]*r[i][j]
        
        for i in range(3):
            d[i] = rad[0] - rad[i+1]
            s[i] = (r2[0] - r2[i+1] - rad2[0] + rad2[i+1]) * 0.5
            ds[0][i] = d[i]
            ds[1][i] = s[i]
            for j in range(3):
                # lapack uses fortran ordering
                M[j][i] = r[0][j] - r[i+1][j]
   
        _solve(&(M[0][0]),&(ds[0][0]),&info)

        if (info> 0):
            Rv[simplex]=0.0
            continue
        u2 = 0.0
        v2 = 0.0
        uv = 0.0
        for i in range(3):
            vi = r[0][i]-ds[1][i]
            v2 += vi*vi
            u2 += ds[0][i]*ds[0][i]
            uv += ds[0][i]*vi
        A = rad[0] - uv
        arg = A*A - (u2 -1) * (v2 -rad2[0])
        if arg < 0.0:
            Rv[simplex]=0.0
            continue
            
        B = sqrt(arg)
        C = u2 - 1
        R_1 = (A+B)/C
        R_2 = (A-B)/C
        if R_1 < 0.0:
            if R_2 < 0:
                Rv[simplex] = 0.0
            else:
                Rv[simplex] = R_2
        else:
            if R_2 < 0:
                Rv[simplex] = R_1
            elif R_2 < R_1:
                Rv[simplex] = R_2
            else:
                Rv[simplex] = R_1
    return R

def circumradius(points, radii, simplices):
    return _circumradius(points, radii, simplices)


