cdef extern from "gtools.h":
  double sphere(int idx0, int idx1, int idx2, int idx3, const double* points, const double* weights, double* center)

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
# adapter for call for one tetrahedon
def tsphere(int idx0, int idx1, int idx2, int idx3, 
            double[:,::1] points, double[:] weights, double[:] center):
  return sphere(idx0, idx1, idx2, idx3, &points[0,0],&weights[0],&center[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def next_tetrahedon(int[:] triangle, int[:] nearpoints, int numberOfNearpoints,
                    double[:] oldcenter, 
                    double[:,::1] points, double[:] weights,double[:] center):
  cdef int i
  cdef double minCenter[3]  # C-array for output of coordinations of centers
  cdef double min_distance2 = 1e300
  cdef int min_index = -1
  cdef double radius=0
  cdef double minRadius=0
  for i in range(numberOfNearpoints):
    radius=sphere(triangle[0], triangle[1], triangle[2], nearpoints[i], 
           &points[0,0], &weights[0], &center[0])
    distance2 = ((oldcenter[0]-center[0])*(oldcenter[0]-center[0]) +
                 (oldcenter[1]-center[1])*(oldcenter[1]-center[1]) +
                 (oldcenter[2]-center[2])*(oldcenter[2]-center[2]))
        
    if distance2 < min_distance2:
       min_distance2 = distance2		
       min_index = i
       minCenter[0]=center[0]
       minCenter[1]=center[1]
       minCenter[2]=center[2]
       minRadius=radius
  
  center[0]=minCenter[0]
  center[1]=minCenter[1]
  center[2]=minCenter[2]
      
  return [min_index,minRadius]
