# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
import numpy as np

def PBC(vector,box):
    cond = np.where(vector<-box/2.)
    vector[cond]+=box[cond]
    cond = np.where(vector>box/2.)
    vector[cond]-=box[cond]

def squareDistancePBC(r1,r2,box):
    rij=r2-r1
    PBC(rij, box)
    return sum(rij[:]**2)

def isTooLarge(a,b,c,d,box):
    d1=squareDistancePBC(a,b,box)
    d2=squareDistancePBC(a,c,box)
    d3=squareDistancePBC(a,d,box)
    d4=squareDistancePBC(b,c,box)
    d5=squareDistancePBC(b,d,box)
    d6=squareDistancePBC(c,d,box)
    boxx2=(box[0]/2.0)**2
    boxy2=(box[1]/2.0)**2
    boxz2=(box[2]/2.0)**2
    
    sqd =  np.array([d1,d2,d3,d4,d5,d6])
    if np.any(sqd > boxx2):
        return True
    if np.any(sqd > boxy2):
        return True
    if np.any(sqd > boxz2):
        return True
    return False

def isUnique(indices,tHash):
    ind1=np.sort(indices)
    if(str(ind1) in tHash):
        return False
    return True

def findPBCNeighboringSimplices(triangulation,originalIndex,finalSimplices,tHash,extraids):
    pbcNeighbors=[]
    t=triangulation
    neighbor=0
    for index in range(0,len(originalIndex)):
        pbcNeighbors.append([])
        for i in range(0,4):
            neighbor=t.neighbors[originalIndex[index],i]
            vec1=np.array([extraids[t.simplices[neighbor,0]], extraids[t.simplices[neighbor,1]], extraids[t.simplices[neighbor,2]], extraids[t.simplices[neighbor,3]]])
            vec1=np.sort(vec1)
            if(str(vec1) in tHash):
                pbcNeighbors[index].append(tHash[str(vec1)])
            else:
                pbcNeighbors[index].append(-1)
    #print("ukaz souseda: ",t.neighbors[1,0])
    return pbcNeighbors
    

def clearPBCtriangulation(triangulation,extrapoints,extraids,box):
    t=triangulation
    finalSimplices=[]
    originalIndex=[]
    outx=np.zeros(4)
    outy=np.zeros(4)
    outz=np.zeros(4)
    tHash={}
    index=0
    for simplex in t.simplices:
       # if (isTooLarge(t.points[extraids[simplex[0]]],t.points[extraids[simplex[1]]],
        #               t.points[extraids[simplex[2]]],t.points[extraids[simplex[3]]],box)):
        #    continue
        vec1=np.array([extraids[simplex[0]], extraids[simplex[1]], extraids[simplex[2]], extraids[simplex[3]]])
        #je uvnitr domény?
        outx[:]=0
        outy[:]=0
        outz[:]=0
        for i in range(0,4):
            if(t.points[simplex[i],0]>box[0] or t.points[simplex[i],0]<0):
                outx[i]=1
            if(t.points[simplex[i],1]>box[1] or t.points[simplex[i],1]<0):
                outy[i]=1
            if(t.points[simplex[i],2]>box[2] or t.points[simplex[i],2]<0):
                outz[i]=1
        if(sum(outx)<=2 and sum(outy)<=2 and sum(outz)<=2):
            if(isUnique(vec1,tHash)):
                finalSimplices.append(simplex)
                originalIndex.append(index)
                #print("test",len(finalSimplices),len(t.simplices))
                vec1=np.array([extraids[simplex[0]], extraids[simplex[1]], extraids[simplex[2]], extraids[simplex[3]]])
                vec1=np.sort(vec1)
                tHash[str(vec1)]=len(finalSimplices)-1
        index+=1
    neighbors=findPBCNeighboringSimplices(triangulation,originalIndex,finalSimplices,tHash,extraids)
    return [finalSimplices,neighbors]
