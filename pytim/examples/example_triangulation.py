# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import numpy as np 
import pytim
from pytim import *
from pytim.datafiles import *

interface = pytim.ITIM(mda.Universe(WATER_GRO))
surface   = observables.LayerTriangulation(interface)

interface.assign_layers()

stats, tri =  surface.compute()

print("The total triangulated surface has an area of {:04.1f} Angstrom^2".format(stats[0]))

#subset=oxygens[::100]
#print subset
#tri, index = pytim.distance_from_planar_set(subset,layer)
#print tri
#print index,len(index)
#
#pytim.lap()
#
#import matplotlib.pyplot as plt
#plt.triplot(layer.positions[:,0], layer.positions[:,1],tri.simplices.copy())
#plt.plot(subset.positions[:,0], subset.positions[:,1], 'o')
#plt.triplot(layer.positions[:,0], layer.positions[:,1],tri.simplices[index].copy(),lw=4)
#plt.show()
    


