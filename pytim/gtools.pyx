from libc.stdlib cimport malloc,free
cimport numpy as np
import numpy as np
cimport cython

cdef extern from "gtools.h":
  cdef double sphere(int idx0, int idx1, int idx2, int idx3, const double* points, const double* weights, double* center)

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

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)

def makeClusters(double[:,::1] points, double alpha, long[:,::1] tetrahedrons,
long[:,::1] neighbors, float[:] box, long[:] extraids, long[:] isDense, cluster_group, Cluster):
    cdef int n=len(tetrahedrons)
    cdef int n_Nei=len(neighbors[0])
    clusters=[]
    cdef double rij[3]
    cdef int vec[3]
    cdef int vec1d[3]
    cdef int vec2d[3]
    cdef double *inCl=<double*> malloc(n*sizeof(double))
    #cdef double (*rInCl)[3] = <double (*)[3]>malloc(n*3*sizeof(double))
    cdef np.ndarray[np.float64_t,ndim=2] rInCl=np.zeros((n,3),dtype=np.float64)
    #cdef double *tInCl=<double*> malloc(n*sizeof(double))
    cdef np.ndarray[np.int64_t,ndim=1] tInCl=np.zeros(n,dtype=np.int64)
    cdef int nCl=-1
    cdef int dim=0
    cdef int j,jindex,k,kindex

    cdef int i=0
    for i in range(0,n): inCl[i]=-1
    for i in range(0,n):
        if(inCl[i]==-1): #i-th tetrahedron isn't used in any cluster
            nCl+=1
            clusters.append(Cluster(cluster_group,box))
            clusters[nCl].clusterDensity=isDense[i]
            clusters[nCl].clusterDimension=0
            dim=0
            vec1d[0]=0;vec1d[1]=0;vec1d[2]=0;
            vec2d[0]=0;vec2d[1]=0;vec2d[2]=0;
            clusters[nCl].appendTetrahedron(i,tetrahedrons[i],extraids,points)
            inCl[i]=nCl
            tInCl[i]=nCl
            rInCl[i,0]=points[extraids[tetrahedrons[i,0]],0]
            rInCl[i,1]=points[extraids[tetrahedrons[i,0]],1]
            rInCl[i,2]=points[extraids[tetrahedrons[i,0]],2]
            
            #for j in clusters[nCl].tetrahedrons: #loop over tetrahedra already added to the cluster
            jindex=0
            while jindex < len(clusters[nCl].tetrahedrons):
                j=clusters[nCl].tetrahedrons[jindex]
                jindex+=1
                for kindex in range(0,n_Nei):
                    k=neighbors[j,kindex]
                    if k<0: continue
                    rij[0]=points[extraids[tetrahedrons[k,0]],0]-rInCl[j,0]#t.points[extraids[tetrahedrons[j][0]],:]
                    rij[1]=points[extraids[tetrahedrons[k,0]],1]-rInCl[j,1]
                    rij[2]=points[extraids[tetrahedrons[k,0]],2]-rInCl[j,2]
                    PBC(rij,box)
                    rij[0]+=rInCl[j,0];rij[1]+=rInCl[j,1];rij[2]+=rInCl[j,2]
                    if(inCl[k]==-1 and isDense[k]==isDense[i]):
                        rInCl[k,0]=rij[0];rInCl[k,1]=rij[1];rInCl[k,2]=rij[2]
                        clusters[nCl].appendTetrahedron(k,tetrahedrons[k],extraids,points)
                        inCl[k]=nCl
                        tInCl[k]=nCl
                    elif(isDense[k]==isDense[i]):
                        vec[0]=<int>(((rij[0]-rInCl[k,0])/box[0])+0.5)
                        vec[1]=<int>(((rij[1]-rInCl[k,1])/box[1])+0.5)
                        vec[2]=<int>(((rij[2]-rInCl[k,2])/box[2])+0.5)
                        if(vec[0]>0 or vec[1]>0 or vec[2]>0):
                            if(dim==0):
                                dim=1
                                vec1d[0]=vec[0];vec1d[1]=vec[1];vec1d[2]=vec[2];
                                clusters[nCl].clusterDimension=dim
                            elif(dim==1):
                                cross(vec,vec1d,vec2d)
                                if(vec2d[0]>0 or vec2d[1]>0 or vec2d[2]>0):
                                    dim=2
                                    clusters[nCl].clusterDimension=dim
                            elif(dim==2):
                                if(dot(vec,vec2d)!=0):
                                    dim=3
                                    clusters[nCl].clusterDimension=dim
   
    free(inCl)
    
    return (clusters,tInCl)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def makeClustersFromInterfaces(long [:,::1] neighbors, interface,int maxInterfaces):
    compactInterfaces=[]
    cdef double rij[3]
    cdef n=len(interface)
    cdef int *inCl=<int*> malloc(n*sizeof(int))
    cdef int nCl=-1
    cdef int i,jindex,k,j
   
    for i in range(0,n):inCl[i]=-1
     
    for i in range(0,n):
        if(len(compactInterfaces)>=maxInterfaces):break
        if(inCl[i]==-1):
            compactInterfaces.append([])
            nCl+=1
            compactInterfaces[nCl].append(i)
            inCl[i]=nCl
            jindex=0
            while jindex < len(compactInterfaces[nCl]):
                j=compactInterfaces[nCl][jindex]
                jindex+=1
                for kindex in range(0,len(neighbors[j])):
                    k=neighbors[j,kindex]
                    if(k<0):continue
                    if(inCl[k]<0):
                        compactInterfaces[nCl].append(k)
                        inCl[k]=nCl
    free(inCl)
    return compactInterfaces
   


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int dot(int* vec1,int* vec2):
    return vec1[0]*vec1[0]+vec1[1]*vec1[1]+vec1[2]*vec1[2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cross(int* vec1,int* vec2, int* vec3):
    vec3[0]=vec1[1]*vec2[2]-vec1[2]*vec2[1];vec3[1]=vec1[2]*vec2[0]-vec1[0]*vec2[2]; vec3[2]=vec1[0]*vec2[1]-vec1[1]*vec2[0]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)    
cdef PBC(double *vector,float[:] box):
    if(vector[0]<-box[0]/2.0):vector[0]+=box[0]
    if(vector[0]>box[0]/2.0):vector[0]-=box[0]
    if(vector[1]<-box[1]/2.0):vector[1]+=box[1]
    if(vector[1]>box[1]/2.0):vector[1]-=box[1]
    if(vector[2]<-box[2]/2.0):vector[2]+=box[2]
    if(vector[2]>box[2]/2.0):vector[2]-=box[2]
