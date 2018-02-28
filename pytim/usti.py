#!/usr/bin/python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: usti
    ============
"""
from __future__ import print_function
import numpy as np
import datetime
from scipy.spatial import distance
from gtools import tsphere

from . import utilities
from .sanity_check import SanityCheck
try:
    from pytetgen import Delaunay
except ImportError:
    from scipy.spatial import Delaunay

from .Interface import Interface
from .patches import PatchTrajectory, PatchOpenMM, PatchMDTRAJ

from . import quasiTriangulation

class USTI(Interface):
    """ Identifies interfacial molecules at curved interfaces using
        the method TODO add description

        * TODO: add reference *

        :param Object universe:     The MDAnalysis Universe, MDTraj trajectory
                                    or OpenMM Simulation objects.
        :param Object group:        An AtomGroup, or an array-like object with
                                    the indices of the atoms in the group. Will
                                    identify the interfacial molecules from
                                    this group
        :param float alpha:         The probe sphere radius
        :param str normal:          'x','y,'z' or 'guess'
                                    (for planar interfaces only)
        :param bool molecular:      Switches between search of interfacial
                                    molecules / atoms (default: True)
        :param int max_layers:      The number of layers to be identified
        :param int max_interfaces:  The maximal number of interfaces to be identified
        :param dict radii_dict:     Dictionary with the atomic radii of the
                                    elements in the group. If None is supplied,
                                    the default one (from GROMOS 43a1) will be
                                    used.
        :param float cluster_cut:   Cutoff used for neighbors or density-based
                                    cluster search (default: None disables the
                                    cluster analysis)
        :param float cluster_threshold_density: Number density threshold for
                                    the density-based cluster search. 'auto'
                                    determines the threshold automatically.
                                    Default: None uses simple neighbors cluster
                                    search, if cluster_cut is not None
        :param Object extra_cluster_groups: Additional groups, to allow for
                                    mixed interfaces
        :param bool biggest_cluster_only: Tag as surface atoms/molecules only
                                    those in the largest cluster. Need to
                                    specify also a :py:obj:`cluster_cut` value.
        :param str symmetry:        Gives the code a hint about the topology
                                    of the interface: 'generic' (default)
                                    or  'planar'
        :param bool centered:       Center the  :py:obj:`group`
        :param bool info:           Print additional info
        :param bool warnings:       Print warnings
        :param bool autoassign:     If true (default) detect the interface
                                    every time a new frame is selected.
        :param np.array boolean periodicity: specify in which dimensions PBC will be applyied
        :param str trianType:       Type of triangulation: Delaunay, regular, quasi
    """

    def __init__(self,
                 universe,
                 group=None,
                 alpha=2.0,
                 normal='guess',
                 molecular=True,
                 max_layers=1,
                 max_interfaces=1,
                 radii_dict=None,
                 cluster_cut=None,
                 cluster_threshold_density=None,
                 extra_cluster_groups=None,
                 biggest_cluster_only=False,
                 symmetry='generic',
                 centered=False,
                 info=False,
                 warnings=False,
                 autoassign=True,
                 _noextrapoints=False,
                 periodicity=np.ones(3),
                 trianType='Delaunay', #'quasi', 'regular'
                 **kargs):

        # this is just for debugging/testing
        self._noextrapoints = _noextrapoints
        self.autoassign = autoassign

        self.do_center = centered

        self.biggest_cluster_only = biggest_cluster_only
        sanity = SanityCheck(self)
        sanity.assign_universe(
            universe, radii_dict=radii_dict, warnings=warnings)
        sanity.assign_alpha(alpha)

        self.cluster_threshold_density = cluster_threshold_density
        self._clusters=[]
        self.max_layers = max_layers
        self.max_interfaces = max_interfaces
        self._layers = np.empty([max_layers], dtype=type(universe.atoms))
        self.trianType=trianType
        self.periodicity=periodicity
        self.info = info
        self.normal = None
        self.PDB = {}
        self.molecular = molecular
        sanity.assign_groups(group, cluster_cut, extra_cluster_groups)
        sanity.check_multiple_layers_options()
        sanity.assign_radii()

        self._assign_symmetry(symmetry)

        PatchTrajectory(self.universe.trajectory, self)

        if self.info: t1 = datetime.datetime.now()
        self._assign_layers()
        if self.info:
            t2 = datetime.datetime.now()
            print("time of USTI: ",t2-t1)

        self._layers=self._layers[0,0,0:self.max_layers]
        

    def _sanity_checks(self):
        """ Basic checks to be performed after the initialization.

        """

    def assignTetrahedrons(self,tetrahedrons,alpha):
        isDense=np.zeros(len(tetrahedrons))
        center=np.zeros(3)
        weights=np.zeros(len(self.triangulation.points))
        i=0
        for simplex in tetrahedrons:
            if(self.trianType=='quasi'):
                isDense[i]=1 if self.triangulation.touchingRadii[i]< alpha else 0
            elif(self.trianType=='Delaunay'):
                isDense[i]=1 if tsphere(simplex[0],simplex[1],simplex[2],simplex[3], self.triangulation.points, weights, center)<alpha else 0
            elif(self.trianType=='regular'):
                weights=self.triangulation.weights
                isDense[i]=1 if tsphere(simplex[0],simplex[1],simplex[2],simplex[3], self.triangulation.points, weights, center)<alpha else 0
            i+=1
        return isDense


    def makeClusters(self,alpha,tetrahedrons,neighbors,box,extraids):
        isDense=self.assignTetrahedrons(tetrahedrons,alpha) #It assigns tetrahedrons to the dilute or dense phase
        t = self.triangulation
        n=len(tetrahedrons)
        clusters=[]
        rij=np.zeros(3)
        vec=np.zeros(3)
        vec1d=np.zeros(3)
        vec2d=np.zeros(3)
        inCl=np.zeros(n)
        rInCl=np.zeros((n,3),dtype=np.float64)
        tInCl=np.zeros(n)
        nCl=-1
        inCl[:]=-1
        dim=0
        
        for i in range(0,n):
            if(inCl[i]==-1): #i-th tetrahedron isn't used in any cluster
                nCl+=1
                clusters.append(Cluster(self.cluster_group))
                clusters[nCl].clusterDensity=isDense[i]
                clusters[nCl].clusterDimension=0
                dim=0
                vec1d[:]=0
                vec2d[:]=0
                clusters[nCl].appendTetrahedron(i,tetrahedrons[i],extraids)
                inCl[i]=nCl
                tInCl[i]=nCl
                rInCl[i,:]=t.points[extraids[tetrahedrons[i][0]],:]
                for j in clusters[nCl].tetrahedrons: #loop over tetrahedra already added to the cluster
                    for k in neighbors[j]:
                        if k<0: continue
                        rij[:]=t.points[extraids[tetrahedrons[k][0]],:]-rInCl[j,:]#t.points[extraids[tetrahedrons[j][0]],:]
                        utilities.PBC(rij,box)
                        rij[:]+=rInCl[j,:]
                        if(inCl[k]==-1 and isDense[k]==isDense[i]):
                            rInCl[k,:]=rij[:]
                            clusters[nCl].appendTetrahedron(k,tetrahedrons[k],extraids)
                            inCl[k]=nCl
                            tInCl[k]=nCl
                        elif(isDense[k]==isDense[i]):
                            vec[0]=round((rij[0]-rInCl[k,0])/box[0])
                            vec[1]=round((rij[1]-rInCl[k,1])/box[1])
                            vec[2]=round((rij[2]-rInCl[k,2])/box[2])
                            if(any(vec)):
                                if(dim==0):
                                    dim=1
                                    vec1d[0]=vec[0];vec1d[1]=vec[1];vec1d[2]=vec[2];
                                    clusters[nCl].clusterDimension=dim
                                elif(dim==1):
                                    vec2d=np.cross(vec,vec1d)
                                    if(any(vec2d)):
                                        dim=2
                                        clusters[nCl].clusterDimension=dim
                                elif(dim==2):
                                    if(np.dot(vec,vec2d)!=0):
                                        dim=3
                                        clusters[nCl].clusterDimension=dim
        return [clusters, tInCl]
    
    def findInterfaces(self,tInCl,clusters,tNeighbors,simplices):
        interfaces=[]#np.empty((len(clusters),),dtype=object)
        for i in range(0,len(clusters)):
            interfaces.append([])
            for item in clusters[i].tetrahedrons:
                if((tNeighbors[item][0]<0) or i!=tInCl[tNeighbors[item][0]]):
                    interfaces[i].append(Triangle(simplices[item][1],simplices[item][2],simplices[item][3]))
                if(tNeighbors[item][1]<0 or i!=tInCl[tNeighbors[item][1]]):
                    interfaces[i].append(Triangle(simplices[item][0],simplices[item][2],simplices[item][3]))
                if(tNeighbors[item][2]<0 or i!=tInCl[tNeighbors[item][2]]):
                    interfaces[i].append(Triangle(simplices[item][0],simplices[item][1],simplices[item][3]))
                if(tNeighbors[item][3]<0 or i!=tInCl[tNeighbors[item][3]]):
                    interfaces[i].append(Triangle(simplices[item][0],simplices[item][1],simplices[item][2]))
                    
                if(len(tNeighbors[item])>4):
                    if((tNeighbors[item][4]>-2) and i!=tInCl[tNeighbors[item][4]]):
                        interfaces[i].append(Triangle(simplices[item][1],simplices[item][2],simplices[item][3]))
                    if(tNeighbors[item][5]>-2 and i!=tInCl[tNeighbors[item][5]]):
                        interfaces[i].append(Triangle(simplices[item][0],simplices[item][2],simplices[item][3]))
                    if(tNeighbors[item][6]>-2 and i!=tInCl[tNeighbors[item][6]]):
                        interfaces[i].append(Triangle(simplices[item][0],simplices[item][1],simplices[item][3]))
                    if(tNeighbors[item][7]>-2 and i!=tInCl[tNeighbors[item][7]]):
                        interfaces[i].append(Triangle(simplices[item][0],simplices[item][1],simplices[item][2]))
        return interfaces
    
    
    def findNeighboringTriangles(self,interface,extraids):
        neighbors=np.ones((len(interface),27),dtype=np.int)*(-1)
        neighborsCount=np.zeros(len(interface),dtype=np.int)
        for i in range(0,len(interface)-1):
            for j in range(i+1,len(interface)):
                if(interface[i].isNeighborOf(interface[j],extraids)):
                    if(not j in neighbors[i]):
                        neighbors[i][neighborsCount[i]]=j
                        neighborsCount[i]+=1
                    if(not i in neighbors[j]):
                        neighbors[j][neighborsCount[j]]=i
                        neighborsCount[j]+=1
        return neighbors
    
    
    def makeClustersFromInterfaces(self,neighbors,interface):
        compactInterfaces=[]
        rij=np.array(3)
        inCl=np.zeros(len(interface))
        inCl[:]=-1
        nCl=-1
        
        for i in range(0,len(interface)):
            if(inCl[i]==-1):
                compactInterfaces.append([])
                nCl+=1
                compactInterfaces[nCl].append(i)
                inCl[i]=nCl
                for j in compactInterfaces[nCl]:
                    for k in neighbors[j]:
                        if(k<0):continue
                        if(inCl[k]<0):
                            compactInterfaces[nCl].append(k)
                            inCl[k]=nCl
        return compactInterfaces
   
    def getInterfaces(self,tInCl,clusters,tNeighbors,tetrahedrons,extraids):

        interfaces=[]
        interface=self.findInterfaces(tInCl,clusters,tNeighbors,tetrahedrons)
        if self.info:
            print('findInterfaces done')
        for i in range(0,len(interface)):
            neighbors=self.findNeighboringTriangles(interface[i],extraids)
            interfaces.append(self.makeClustersFromInterfaces(neighbors,interface[i]))
        return [interface,interfaces]

    def findLayers(self,cluster,clusterInterface,individualInterface,extraids): 
        """ return list of indices of molecules in individual layers for the required cluster
        """
        layers=[]
        layers.append([])
        isUsed={}
        index1=0
        next=False
        for intf in individualInterface:
            index1=extraids[clusterInterface[intf].A]
            if(not index1 in isUsed):
                layers[0].append(index1)
                isUsed[index1]=0
            index1=extraids[clusterInterface[intf].B]
            if(not index1 in isUsed):
                layers[0].append(index1)
                isUsed[index1]=0
            index1=extraids[clusterInterface[intf].C]
            if(not index1 in isUsed):
                layers[0].append(index1)
                isUsed[index1]=0
               
        i=0 
        for lay in layers:
            next=False
            for item in lay:
                if(not next):
                    layers.append([])
                    next=True
                for n in cluster.neighboringAtoms[item]:
                    if(not n in isUsed):
                        layers[i+1].append(n)
                        isUsed[n]=i+1
            i+=1
            
        return layers[0:i-1]

    def getLayers(self,clusters,interface,interfaces,extraids): 
        """ collect layers from each interfaces in each cluster
        """
        layers=[]
        for i in range(0,len(interface)):
            layers.append([])
            for infc in interfaces[i]:
                layers[i].append(self.findLayers(clusters[i],interface[i],infc,extraids))
        return layers    

    def alpha_shape(self, alpha):
        box = self.universe.dimensions[:3]
        delta = 2. * self.alpha + 1e-6
        points = self.cluster_group.positions[:]
        nrealpoints = len(points)
        np.random.seed(0)  # pseudo-random for reproducibility
        gitter = (np.random.random(3 * 8).reshape(8, 3)) * 1e-9
        if self._noextrapoints == False:
            extrapoints, extraids = utilities.generate_periodic_border_for_usti(
                points, box, delta,self.periodicity)

        extrapoints = np.asarray(extrapoints,dtype=np.float)

        if self.info:
            t1 = datetime.datetime.now()
        weights=np.zeros(len(extrapoints))
        if(self.trianType=="Delaunay"):
            self.triangulation = Delaunay(extrapoints)
        else:
            for i in range(0,len(extrapoints)):
                weights[i]=self.cluster_group.radii[extraids[i]]
            if(self.trianType=='quasi'):
                self.triangulation=quasiTriangulation.QuasiTriangulation(extrapoints,weights,box+2.0*delta)
            elif(self.trianType=='regular'):
                self.triangulation = Delaunay(extrapoints,weights=weights)

        if self.info:
            print(len(self.triangulation.simplices))
            t2 = datetime.datetime.now()
            print("time of triangulation: ",t2-t1)
        [tetrahedrons,neighbors]=utilities.clearPBCtriangulation(self.triangulation,extrapoints,extraids,box)
        if self.info:
            t3 = datetime.datetime.now()
            print("PBC smoothing: ",t3-t2)
        [self._clusters,tInCl]=self.makeClusters(alpha,tetrahedrons,neighbors,box,extraids)
        if self.info:
            t4 = datetime.datetime.now()
            print("clusters: ",t4-t3)
        [interface,interfaces]=self.getInterfaces(tInCl,self._clusters,neighbors,tetrahedrons,extraids)
        if self.info:
            t5 = datetime.datetime.now()
            print("interfaces: ",t5-t4)
        if(len(interface)==1 and len(interface[0])==0):
            raise RuntimeError("No interfaces found! Please check the value of threshold parameter!")

        layers=self.getLayers(self._clusters,interface,interfaces,extraids)
        if self.info:
            t6 = datetime.datetime.now()
            print("layers: ",t6-t5)
        self._layers = np.empty([len(self._clusters),self.max_interfaces, self.max_layers], dtype=type(self.universe.atoms))
        i=0
        maxvalue=0
        for c in self._clusters:
             c.interfaces=interfaces[i]
             if(len(c.tetrahedrons)>maxvalue):
                maxvalue=len(c.tetrahedrons)
                self.largestCluster=i
             i+=1
        
        return layers

    def _assign_layers(self):
        """Determine the USTI layers."""
        self.reset_labels()
        # this can be used later to shift back to the original shift
        self.original_positions = np.copy(self.universe.atoms.positions[:])
        self.universe.atoms.pack_into_box()

        self._define_cluster_group()

        self.centered_positions = None
        if self.do_center:
            self.center()

        # first we label all atoms in group to be in the gas phase
        self.label_group(self.itim_group.atoms, 0.5)
        # then all atoms in the larges group are labelled as liquid-like
        self.label_group(self.cluster_group.atoms, 0.0)

        size = len(self.cluster_group.positions)
        alpha_ids = self.alpha_shape(self.alpha)

        max_usedLayers=0

        #for each cluster "c" and its each interface "i" asign layer "l"
        for c in range(0,len(alpha_ids)):
            for i in range(0,len(alpha_ids[c])):
                if(i<len(alpha_ids[c])and i<self.max_interfaces):
                    for l in range(0,len(alpha_ids[c][i])):
                        if(l<len(alpha_ids[c][i]) and l<self.max_layers):
                            if self.molecular:
                                self._layers[c,i,l] = self.cluster_group[alpha_ids[c][i][l]].residues.atoms
                            else:
                                self._layers[c,i,l] = self.cluster_group[alpha_ids[c][i][l]]
                            if(c==0 and i==0 and self._layers[c,i,l] is not None):
                                max_usedLayers+=1
            
        for c in range(0,len(alpha_ids)):
            for i in range(0,len(alpha_ids[c])):
                if(i<len(alpha_ids[c])and i<self.max_interfaces):
                    for l in range(0,len(alpha_ids[c][i])):
                        if(l<len(alpha_ids[c][i]) and l<self.max_layers):
                            #self.label_group(self._layers[c,i,l], str(c)+'00'+str(i+1+l/10.0))
                            self.label_group(self._layers[c,i,l], beta=1. * (l + 1), layer=(l + 1))
            self.clusters[c].layers=self.layers[c]


        if(max_usedLayers<self.max_layers):
            print("Warning: the system contains fewer layers than required")
            print("requiered: ", self.max_layers, " found: ",max_usedLayers)
            self.max_layers=max_usedLayers


    @property
    def layers(self):
        """Access the layers as numpy arrays of AtomGroups.

        The object can be sliced as usual with numpy arrays.
        Differently from ITIM, there are no sides. Example:
        TODO: check this example

        >>> import MDAnalysis as mda
        >>> import pytim
        >>> from pytim.datafiles import MICELLE_PDB
        >>>
        >>> u = mda.Universe(MICELLE_PDB)
        >>> micelle = u.select_atoms('resname DPC')
        >>> inter = pytim.USTI(u, group=micelle, max_layers=3,molecular=False)
        >>> inter.layers  #all layers
        array([<AtomGroup with 909 atoms>, <AtomGroup with 301 atoms>,
               <AtomGroup with 164 atoms>], dtype=object)
        >>> inter.layers[0]  # first layer (0)
        <AtomGroup with 909 atoms>

        """
        return self._layers

    @property
    def clusters(self):
        return self._clusters

class Cluster():
    def __init__(self,clusterGroup):
        self.tetrahedrons=[]
        self.atomIndices=[]
        self.neighboringAtoms={}
        self.__interfaces=[]
        self.__layers=[]
        self.clusterGroup=clusterGroup

    def appendTetrahedron(self,index,tetrahedron,extraids):
        self.tetrahedrons.append(index)
        for i in range(0,4):
            index1=extraids[tetrahedron[i]]
            if(not index1 in self.neighboringAtoms):
                self.atomIndices.append(index1)
                self.neighboringAtoms[index1]=[]     
            self.appendAtomNeighbors(index1,extraids[tetrahedron])
            
    def appendTetrahedronOld(self,index,tetrahedron,extraids):
        self.tetrahedrons.append(index)
        for i in range(0,4):
            index1=extraids[tetrahedron[i]]
            if(not index1 in self.atomIndices):
                self.atomIndices.append(index1)
                if(not index1 in self.neighboringAtoms):
                    self.neighboringAtoms[index1]=[]     
            self.appendAtomNeighbors(index1,extraids[tetrahedron])
            
    def appendAtomNeighbors(self,index1,index2):
        for i in index2:
            if i==index1:
                continue
            if(not i in self.neighboringAtoms[index1]):
                self.neighboringAtoms[index1].append(i)
    @property
    def clusterDimension(self):
        return self._clusterDimension
    
    @clusterDimension.setter
    def clusterDimension(self,dimension):
        self._clusterDimension=dimension
        
    @property
    def clusterDensity(self):
        return self._clusterDensity
    
    @clusterDensity.setter
    def clusterDensity(self,_density):
        self._clusterDensity=_density
        
    @property
    def interfaces(self):
        return self.__interfaces
    
    @interfaces.setter
    def interfaces(self,value):
        for i in value:
            self.__interfaces.append(i)
    
    @property
    def layers(self):
        return self.__layers
    
    @layers.setter
    def layers(self,value):
        for i in value: #loop over interfaces
            self.__layers.append(i)
        
class Triangle():
    def __init__(self,A,B,C):
        self.A=A
        self.B=B
        self.C=C
        
    def isNeighborOf(self,triangle,extraids):
        if(int(extraids[self.A]==extraids[triangle.A] or extraids[self.A]==extraids[triangle.B] or extraids[self.A]==extraids[triangle.C])+
            int(extraids[self.B]==extraids[triangle.A] or extraids[self.B]==extraids[triangle.B] or extraids[self.B]==extraids[triangle.C])+
            int(extraids[self.C]==extraids[triangle.A] or extraids[self.C]==extraids[triangle.B] or extraids[self.C]==extraids[triangle.C])==2):
            return True
        return False
