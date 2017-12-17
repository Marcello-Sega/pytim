# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: utilities
    =================
"""
from timeit import default_timer as timer
import numpy as np
from sys import stderr

from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup

from pytim.utilities_geometry import trim_triangulated_surface
from pytim.utilities_geometry import triangulated_surface_stats
from pytim.utilities_geometry import polygonalArea, fit_sphere, EulerRotation
from pytim.utilities_pbc import generate_periodic_border, rebox
from pytim.gaussian_kde_pbc import gaussian_kde_pbc
from pytim.utilities_dbscan import do_cluster_analysis_DBSCAN

from atoms_maps import atoms_maps
from pytim.utilities_mesh import compute_compatible_mesh_params
from pytim.utilities_mesh import generate_grid_in_box


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


def correlate(a1=np.ndarray(0), a2=None):
    """ correlate data series using numpy fft. The function calculates correlation
        or cross-correlation.

        :param ndarray a1:   first data set to correlate
        :param ndarray a2:   (optional) second data set, to compute cross-correlation

        Example: time autocorrelation of the number of atoms in the outermost layer

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
        >>> corr = pytim.utilities.correlate(size-np.mean(size))
        >>> corr = corr/corr[0] # normalize to 1
        >>> print corr
        [ 1.          0.1420121   0.10364119  0.14718647  0.37093981  0.09908694
          0.16514898  0.0946748   0.33824381  0.2187186  -0.02084513  0.08711942
          0.24537069 -0.0102749  -0.1934566   0.10323017  0.02911581 -0.00939353
         -0.11041383  0.01191062 -0.13293405  0.05622434 -0.2826456  -0.27631805
          0.0351999  -0.01167737 -0.21058736 -0.42930886 -0.13241366 -0.26325648
          0.07229366 -0.70028015 -0.23617053  0.13629839 -0.24335089 -0.87832556
         -0.12957699 -0.32853026 -0.3863053  -0.65227527 -0.2672419  -0.18756502
         -0.22565105 -0.78979698 -0.28407306 -0.02037816 -1.5120148  -1.31553408
         -0.18836842  7.55135513]

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

    size = len(a1)
    norm = np.arange(size)[::-1] + 1
    fa1 = np.fft.fft(a1, axis=0, n=size * 2)

    if not isinstance(a2, type(None)):  # do cross-corr
        fa2 = np.fft.fft(a2, axis=0, n=size * 2)
        return ((np.fft.fft(fa2 * np.conj(fa1) + fa1 * np.conj(fa2), axis=0)[:size]).real.T / norm).T / len(fa1) / 2.
    else:                               # do auto-corr
        return ((np.fft.fft(fa1 * np.conj(fa1), axis=0)[:size]).real.T / norm).T / len(fa1)


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

def centerbox(universe, x=None, y=None, z=None, vector=None,
              center_direction=2, halfbox_shift=True):
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
    if halfbox_shift == True:
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
            except Exception:
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
        histo, _ = np.histogram(group.positions[:, direction], bins=5,
                                range=(0, dim[direction]),
                                density=True);
        max_val = np.amax(histo)
        min_val = np.amin(histo)
        delta.append(np.sqrt((max_val - min_val)**2))

    if np.max(delta) / np.min(delta) < 5.0:
        print "Warning: the result of the automatic normal detection (", np.argmax(delta), ") is not reliable"
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
#
