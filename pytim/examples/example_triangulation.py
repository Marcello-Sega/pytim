# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import numpy as np
import pytim
from pytim import observables
from pytim.datafiles import *

interface = pytim.ITIM(mda.Universe(WATER_GRO))
box = interface.universe.dimensions[:3]

# triangulate the surface
surface = observables.LayerTriangulation(interface)

# obtain : statistics on the surface, two Delaunay objects,
#          the points belonging to the surfaces,
#          the triangles points clipped to the simulation box
stats, tri, points, trim = surface.compute()

print("The total triangulated surface has an area of {:04.1f} Angstrom^2".format(
    stats[0]))


# plot the triangulation using matplotlib
try:
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtplt_tri
except:
    print ("mpl_toolkits is needed for the graphical part of this example, you might need to upgrade matplotlib to v. 2.0")
    exit()
try:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim([0, box[0]])
    ax.set_ylim([0, box[1]])

    for surf in [0, 1]:
        # we need to clip the triangulation to the basic cell, but ax.set_xlim()
        # does not work for 3d plots in matplotlib.
        for axis in [0, 1]:
            points[surf][:, 2][points[surf][:, axis] < 0] = np.nan
            points[surf][:, 2][points[surf][:, axis] > box[axis]] = np.nan
        # create a matplotlib Triangulation (mtmplt_tri does not accept Delaunay
        # objects, but his owns)
        triang = mtplt_tri.Triangulation(
            x=points[surf][:, 0], y=points[surf][:, 1], triangles=tri[surf].simplices)
        # plot the triangulation of each of the two surfaces
        ax.plot_trisurf(triang, points[surf][:, 2],
                        linewidth=0.2, antialiased=True)

    try:
        # nice latex labels for publication-ready figures, in case
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax.set_xlabel(r'$x/\AA$')
        ax.set_ylabel(r'$y/\AA$')
        ax.set_zlabel(r'$z/\AA$')
    except:
        pass

    # save to pdf and visualize interactively
    plt.savefig("surfaces.pdf")
    print("surface triangulation saved in surfaces.pdf")
    plt.show()
except:
    print "this is not run for code coverage"
