# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-

# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: utilities
    =================
"""
from __future__ import print_function
from timeit import default_timer as timer
from itertools import product
import numpy as np
from sys import stderr

from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup

from .utilities_geometry import trim_triangulated_surface
from .utilities_geometry import triangulated_surface_stats
from .utilities_geometry import polygonalArea, fit_sphere, EulerRotation
from .utilities_geometry import find_surface_triangulation
from .utilities_geometry import pbc_compact, pbc_wrap
from .utilities_pbc import generate_periodic_border, rebox
from .gaussian_kde_pbc import gaussian_kde_pbc
from .utilities_dbscan import do_cluster_analysis_dbscan

from .atoms_maps import atoms_maps
from .utilities_mesh import compute_compatible_mesh_params
from .utilities_mesh import generate_grid_in_box


def lap(show=False):
    """ Timer function

        :param bool show: (optional) print timer information to stderr
    """

    if not hasattr(lap, "tic"):
        lap.tic = timer()
    else:
        toc = timer()
        dt = toc - lap.tic
        lap.tic = toc
        if show:
            stderr.write("LAP >>> " + str(dt) + "\n")
        return dt


def correlate(a1, a2=None, _normalize=True):
    """
      correlate data series using numpy fft. The function calculates \
      correlation or cross-correlation.

      :param ndarray a1: first data set to correlate
      :param ndarray a2: (optional) second data set, to compute the
                         cross-correlation

      Example: time autocorrelation of the number of atoms in the outermost
               layer

      >>> import MDAnalysis as mda
      >>> import pytim
      >>> import numpy as np
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
      >>> np.set_printoptions(precision=3,threshold=1000,suppress=True)
      >>> corr = pytim.utilities.correlate(size-np.mean(size))
      >>> corr = corr/corr[0] # normalize to 1
      >>> print (corr)
      [ 1.     0.153  0.104  0.17   0.365  0.115  0.171  0.104  0.342  0.24
       -0.021  0.097  0.265  0.004 -0.169  0.088  0.022 -0.008 -0.113  0.003
       -0.139  0.051 -0.287 -0.279  0.027 -0.019 -0.223 -0.43  -0.157 -0.285
        0.048 -0.704 -0.26   0.13  -0.31  -0.883 -0.12  -0.323 -0.388 -0.64
       -0.295 -0.177 -0.165 -0.81  -0.321 -0.031 -1.557 -1.296 -0.305  6.974]

      This will produce (sampling the whole trajectory), the following:

      .. plot::

          import numpy as np
          import MDAnalysis as mda
          import pytim
          from   pytim.datafiles import *
          from matplotlib import pyplot as plt

          u = mda.Universe(WATER_GRO,WATER_XTC)
          inter = pytim.ITIM(u)

          size=[]
          time=[]
          for ts in u.trajectory[:]:
              time.append(ts.time)
              size.append(len(inter.layers[0,0]))

          corr =  pytim.utilities.correlate(size-np.mean(size))
          plt.plot(time,corr/corr[0])
          plt.plot(time,[0]*len(time))
          plt.gca().set_xlabel("time/ps")

          plt.show()

    """
    reshaped = False
    a1 = np.asarray(a1)
    size = a1.shape[0]
    if len(a1.shape) == 1:
        reshaped = True
        a1 = a1.reshape(a1.shape[0], 1)
        if a2 is not None:
            a2 = a2.reshape(a2.shape[0], 1)

    if _normalize is True:
        norm = (np.arange(size)[::-1] + 1.).reshape(size, 1)
    else:
        norm = 1.0

    fa1 = np.fft.fft(a1, axis=0, n=size * 2)

    if a2 is None:  # do auto-cross
        corr = (
            np.fft.fft(fa1 * fa1.conj(), axis=0)[:size]).real / norm / len(fa1)
    else:  # do cross-corr
        fa2 = np.fft.fft(a2, axis=0, n=size * 2)
        corr = (np.fft.fft(fa2 * fa1.conj() + fa1 * fa2.conj(),
                           axis=0)[:size]).real / norm / len(fa1) / 2.

    if reshaped is True:
        corr = corr.reshape(corr.shape[0], )

    return corr


def extract_positions(inp):
    if isinstance(inp, np.ndarray):
        positions = inp
    if isinstance(inp, Atom):
        positions = inp.position
    if isinstance(inp, AtomGroup):
        positions = inp.positions
    return positions


def get_box(universe, normal=2):
    box = universe.coord.dimensions[0:3]
    return np.roll(box, 2 - normal)


def get_pos(group, normal=2):
    return np.roll(group.positions, normal - 2, axis=-1)


def get_coord(coord, group, normal=2):
    return group.positions[:, (coord + 1 + normal) % 3]


def get_x(group, normal=2):
    return get_coord(0, group=group, normal=normal)


def get_y(group, normal=2):
    return get_coord(1, group=group, normal=normal)


def get_z(group, normal=2):
    return get_coord(2, group=group, normal=normal)


def centerbox(universe,
              x=None,
              y=None,
              z=None,
              vector=None,
              center_direction=2,
              halfbox_shift=True):
    # in ITIM, the system is always centered at 0 along the normal direction (halfbox_shift==True)
    # To center to the middle of the box along all directions, set
    # halfbox_shift=False
    dim = universe.coord.dimensions
    stack = False
    dirdict = {'x': 0, 'y': 1, 'z': 2}
    if center_direction in dirdict:
        center_direction = dirdict[center_direction]
    if not (center_direction in [0, 1, 2]):
        raise ValueError("Wrong direction supplied to centerbox")

    shift = np.array([0., 0., 0.])
    if halfbox_shift is True:
        shift[center_direction] = dim[center_direction] / 2.
    # we rebox the atoms in universe, and not a vector
    if x is None and y is None and z is None and vector is None:
        stack = True
        x = get_x(universe.atoms)
        y = get_y(universe.atoms)
        z = get_z(universe.atoms)

    if x is None and y is None and z is None and vector is not None:
        vector = rebox(vector, dim[center_direction], shift[center_direction])

    if x is not None or y is not None or z is not None:
        for index, val in enumerate((x, y, z)):
            try:
                # let's just try to rebox all directions. Will succeed only
                # for those which are not None. The >= convention is needed
                # for cKDTree
                val = rebox(val, dim[index], shift[index])
            except TypeError:
                pass
    if stack:
        universe.coord.positions = np.column_stack((x, y, z))


def guess_normal(universe, group):
    """
    Guess the normal of a liquid slab

    """
    universe.atoms.pack_into_box()
    dim = universe.coord.dimensions

    delta = []
    for direction in range(0, 3):
        histo, _ = np.histogram(
            group.positions[:, direction],
            bins=5,
            range=(0, dim[direction]),
            density=True)
        max_val = np.amax(histo)
        min_val = np.amin(histo)
        delta.append(np.sqrt((max_val - min_val)**2))

    if np.max(delta) / np.min(delta) < 5.0:
        print("Warning: the result of the automatic normal detection (",
              np.argmax(delta), ") is not reliable")
    return np.argmax(delta)


def density_map(pos, grid, sigma, box):
    values = np.vstack([pos[::, 0], pos[::, 1], pos[::, 2]])
    kernel = gaussian_kde_pbc(values, bw_method=sigma / values.std(ddof=1))
    kernel.box = box
    kernel.sigma = sigma
    return kernel, values.std(ddof=1)


def _NN_query(kdtree, position, qrange):
    return kdtree.query_ball_point(position, qrange, n_jobs=-1)


def consecutive_filename(universe, basename, extension):
    frame = universe.trajectory.frame
    filename = basename + '.' + str(frame) + '.' + extension
    return filename


def generate_cube_vertices(box, delta=0.0, jitter=False, dim=3):
    """ Generate points at the vertices of a rectangular box

        :param array_like box:   dimensions of the box
        :param array_like delta: add buffer around the box
        :param bool jitter:      create general linear positions
                                 by minute displacements of the
                                 vertices
        :param int dim:          dimension of the rectangular box,
                                 the default is 3

    """
    cube_vertices = np.array(list(product((0., 1.), repeat=dim)))
    vertices = np.multiply(cube_vertices, box + 2*delta) \
        + np.multiply(cube_vertices-1, 2*delta)
    if jitter:
        state = np.random.get_state()
        np.random.seed(0)  # pseudo-random for reproducibility
        vertices += (np.random.random(3 * 8).reshape(8, 3)) * 1e-9
        np.random.set_state(state)

    return vertices


