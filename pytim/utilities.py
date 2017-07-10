# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from timeit import default_timer as timer
import numpy as np
import itertools
from pytim_dbscan import dbscan_inner
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from scipy.cluster import vq

try:  # MDA >=0.16
    from MDAnalysis.core.groups import Atom, AtomGroup, Residue, ResidueGroup
except BaseException:
    from MDAnalysis.core.AtomGroup import Atom, AtomGroup, Residue,\
        ResidueGroup


def lap(show=False):
    if not hasattr(lap, "tic"):
        lap.tic = timer()
    else:
        toc = timer()
        dt = toc - lap.tic
        lap.tic = toc
        if show:
            print("LAP >>> " + str(dt))
        return dt


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


def compute_compatible_mesh_params(mesh, box):
    """ given a target mesh size and a box, return the number of grid elements
        and spacing in each direction, which are commensurate with the box
    """
    n = map(int, np.ceil(box / mesh))
    d = box / n
    return n, d


def rebox(pos, edge, shift):
    """ rebox a vector along one dimension
        :param ndarray pos: the array of components to be reboxed
        :param float edge: the simulation box edge
        :param float shift: additional shift
    """
    condition = pos >= edge - shift
    pos[condition] -= edge

    condition = pos < 0 - shift
    pos[condition] += edge

    return pos


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
    return np.argmax(delta)


def trim_triangulated_surface(tri, box):
    """ Reduce a surface triangulation that has been extended to\
        allow for periodic boundary conditions\
            to the primary cell.

            The simplices within the primary cell are those with at\
            least two vertices within the cell boundaries.

        :param Delaunay tri: a 2D triangulation
        :param ndarray  box: box cell parameters
        :returns ndarray simplices: the simplices within the primary cell.
    """
    condition = np.logical_and(tri.points[tri.simplices] <= box[0:2],
                               tri.points[tri.simplices] > [0, 0])
    sum1 = condition.sum(axis=2)
    return tri.simplices[np.where((sum1 >= 2).sum(axis=1) >= 2)]


def triangulated_surface_stats(tri2d, points3d):
    """ Return basic statistics about a surface triangulation

        Implemented statistics are: surface area

        :param ndarray tri2d : indices of triangles vertices
        :param ndarray points3d : the heigth of each vertex along the third\
                                  dimension
        :returns list stats : the statistics :  [surface_area]
    """

    # NOTE: would it be possible to write a more efficient routine ?
    # some advanced indexing here...
    # points3d[reduced] is an array of shape (x,3,3)
    # we need to subtract the first of the three vectors
    # from each of them. This uses numpy.newaxis
    v = points3d[tri2d] - points3d[tri2d][:, 0][:, None]
    # then we need to make the cross product of the two
    # non-zero vectors of each triplet
    area = np.linalg.norm(np.cross(v[:, 1], v[:, 2]), axis=1).sum() / 2.
    return [area]


def generate_grid_in_box(box, npoints, order='zxy'):
    """generate an homogenous grid of npoints^3 points that spans the
       complete box.

       :param ndarray box: the simulation box edges
       :param ndarray npoints: the number of points along each direction

    """

    x_ = np.linspace(0., box[0], npoints[0])
    y_ = np.linspace(0., box[1], npoints[1])
    z_ = np.linspace(0., box[2], npoints[2])
    if order == 'zyx':
        z, y, x = np.meshgrid(z_, y_, x_, indexing='ij')
    else:
        x, y, z = np.meshgrid(z_, y_, x_, indexing='ij')

    grid = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)
    grid = np.append(grid, z.reshape(-1, 1), axis=1)
    return grid.T


def consecutive_filename(universe, basename, extension):
    frame = universe.trajectory.frame
    filename = basename + '.' + str(frame) + '.' + extension
    return filename


class gaussian_kde_pbc(gaussian_kde):

    def evaluate_pbc(self, points):
        """ PBC-enabled version of scipy.stats.gaussian_kde.evaluate()
        """

        points = np.atleast_2d(points)
        box = self.box
        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (
                    d, self.d)
                raise ValueError(msg)

        result = np.zeros((m,), dtype=float)

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:, i, np.newaxis] - points
                diff = diff.T
                diff -= (diff > box / 2.) * box
                diff += (diff < -box / 2.) * box
                diff = diff.T
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result = result + np.exp(-energy)
        else:
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:, i, np.newaxis]
                diff = diff.T
                diff -= (diff > box / 2.) * box
                diff += (diff < -box / 2.) * box
                diff = diff.T
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result[i] = np.sum(np.exp(-energy), axis=0)

        result = result / self._norm_factor

        return result


def density_map(pos, grid, sigma, box):
    values = np.vstack([pos[::, 0], pos[::, 1], pos[::, 2]])
    kernel = gaussian_kde_pbc(values, bw_method=sigma / values.std(ddof=1))
    kernel.box = box
    return kernel, values.std(ddof=1)


def _NN_query(kdtree, position, qrange):
    return kdtree.query_ball_point(position, qrange, n_jobs=-1)


def generate_periodic_border(points, box, delta,method='3d'):
    """ Selects the pparticles within a skin depth delta from the
        simulation box, and replicates them to mimic periodic
        boundary conditions. Returns all points (original +
        periodic copies) and the indices of the original particles
    """
    extrapoints = np.copy(points)

    if method is '2d':
        shifts = np.array([el+(0,) for el in list(itertools.product([1, -1, 0],
                                                  repeat=2))])
    else:
        shifts = np.array(list(itertools.product([1, -1, 0], repeat=3)))

    extraids = np.arange(len(points), dtype=np.int)
    for shift in shifts:
        if(np.sum(shift * shift)):  # avoid [0,0,0]
            # this needs some explanation:
            # if shift ==0  -> the condition is always true
            # if shift ==1  -> the condition is x > box - delta
            # if shift ==-1 -> the condition is -x > 0 - delta -> x <delta
            # Requiring np.all() to be true makes the logical and returns
            # (axis=1) True for all indices whose atoms satisfy the
            # condition
            selection = np.all(shift * points >= shift * shift *
                               ((box + shift * box) / 2. - delta),
                               axis=1)
            # add the new points at the border of the box
            extrapoints = np.append(
                extrapoints, points[selection] - shift * box, axis=0)
            # we keep track of the original ids.
            extraids = np.append(extraids, np.where(selection)[0])
    return extrapoints, extraids


def do_cluster_analysis_DBSCAN(
        group, cluster_cut, box, threshold_density=None, molecular=True):
    """ Performs a cluster analysis using DBSCAN

        :returns [labels,counts]: lists of the id of the cluster to which\
                                  every atom is belonging to, and of the\
                                  number of elements in each cluster.

        Uses a slightly modified version of DBSCAN from sklearn.cluster
        that takes periodic boundary conditions into account (through
        cKDTree's boxsize option) and collects also the sizes of all
        clusters. This is on average O(N log N) thanks to the O(log N)
        scaling of the kdtree.

    """
    if isinstance(threshold_density, type(None)):
        min_samples = 2
    if isinstance(threshold_density, (float, int)):
        min_samples = threshold_density * 4. / 3. * np.pi * cluster_cut**3
        if min_samples < 2:
            min_samples = 2

    # NOTE: extra_cluster_groups are not yet implemented
    points = group.atoms.positions[:]

    tree = cKDTree(points, boxsize=box[:6])
    neighborhoods = np.array([np.array(neighbors)
                              for neighbors in tree.query_ball_point(
        points, cluster_cut, n_jobs=-1)]
    )
    if len(neighborhoods.shape) != 1:
        raise ValueError("Error in do_cluster_analysis_DBSCAN(), the cutoff\
                          is probably too small")
    if molecular == False:
        n_neighbors = np.array([len(neighbors)
                                for neighbors in neighborhoods])
    else:
        n_neighbors = np.array([len(np.unique(group[neighbors].resids))
                                for neighbors in neighborhoods])

    if isinstance(threshold_density, str):
        if not (threshold_density == 'auto'):
            raise ValueError("Wrong value of 'threshold_density' passed\
                              to do_cluster_analysis_DBSCAN() ")
        modes = 2
        centroid, _ = vq.kmeans2(n_neighbors * 1.0, modes, iter=10,
                                 check_finite=False)
        # min_samples   = np.mean(centroid)
        min_samples = np.max(centroid)

    labels = -np.ones(points.shape[0], dtype=np.intp)
    counts = np.zeros(points.shape[0], dtype=np.intp)

    core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
    dbscan_inner(core_samples, neighborhoods, labels, counts)
    return labels, counts, n_neighbors


def fit_sphere(points):
    """ least square fit of a sphere through a set of points.

        :param ndarray points: a (N,3) array
        :returns list: radius, center_x, center_y, center_z

    """

    px = points[::, 0]
    py = points[::, 1]
    pz = points[::, 2]
    f = np.sum(points * points, axis=1)
    A = np.zeros((len(points), 4))
    A[:, 0] = 2. * px
    A[:, 1] = 2. * py
    A[:, 2] = 2. * pz
    A[:, 3] = 1.
    C = np.dot(np.linalg.pinv(A), f)
    radius = np.sqrt(C[0] * C[0] + C[1] * C[1] + C[2] * C[2] + C[3])
    return radius, C[0], C[1], C[2]


# colormap from http://jmol.sourceforge.net/jscolors/
colormap = {
    'H':  [255, 255, 255],
    'He': [217, 255, 255],
    'Li': [204, 128, 255],
    'Be': [194, 255, 0],
    'B':  [255, 181, 181],
    'C':  [144, 144, 144],
    'N':  [48, 80, 248],
    'O':  [255, 13, 13],
    'F':  [144, 224, 80],
    'Ne': [179, 227, 245],
    'Na': [171, 92, 242],
    'Mg': [138, 255, 0],
    'Al': [191, 166, 166],
    'Si': [240, 200, 160],
    'P':  [255, 128, 0],
    'S':  [255, 255, 48],
    'Cl': [31, 240, 31],
    'Ar': [128, 209, 227],
    'K':  [143, 64, 212],
    'Ca': [61, 255, 0],
    'Sc': [230, 230, 230],
    'Ti': [191, 194, 199],
    'V':  [166, 166, 171],
    'Cr': [138, 153, 199],
    'Mn': [156, 122, 199],
    'Fe': [224, 102, 51],
    'Co': [240, 144, 160],
    'Ni': [80, 208, 80],
    'Cu': [200, 128, 51],
    'Zn': [125, 128, 176],
    'Ga': [194, 143, 143],
    'Ge': [102, 143, 143],
    'As': [189, 128, 227],
    'Se': [255, 161, 0],
    'Br': [166, 41, 41],
    'Kr': [92, 184, 209],
    'Rb': [112, 46, 176],
    'Sr': [0, 255, 0],
    'Y':  [148, 255, 255],
    'Zr': [148, 224, 224],
    'Nb': [115, 194, 201],
    'Mo': [84, 181, 181],
    'Tc': [59, 158, 158],
    'Ru': [36, 143, 143],
    'Rh': [10, 125, 140],
    'Pd': [0, 105, 133],
    'Ag': [192, 192, 192],
    'Cd': [255, 217, 143],
    'In': [166, 117, 115],
    'Sn': [102, 128, 128],
    'Sb': [158, 99, 181],
    'Te': [212, 122, 0],
    'I':  [148, 0, 148],
    'Xe': [66, 158, 176],
    'Cs': [87, 23, 143],
    'Ba': [0, 201, 0],
    'La': [112, 212, 255],
    'Ce': [255, 255, 199],
    'Pr': [217, 255, 199],
    'Nd': [199, 255, 199],
    'Pm': [163, 255, 199],
    'Sm': [143, 255, 199],
    'Eu': [97, 255, 199],
    'Gd': [69, 255, 199],
    'Tb': [48, 255, 199],
    'Dy': [31, 255, 199],
    'Ho': [0, 255, 156],
    'Er': [0, 230, 117],
    'Tm': [0, 212, 82],
    'Yb': [0, 191, 56],
    'Lu': [0, 171, 36],
    'Hf': [77, 194, 255],
    'Ta': [77, 166, 255],
    'W':  [33, 148, 214],
    'Re': [38, 125, 171],
    'Os': [38, 102, 150],
    'Ir': [23, 84, 135],
    'Pt': [208, 208, 224],
    'Au': [255, 209, 35],
    'Hg': [184, 184, 208],
    'Tl': [166, 84, 77],
    'Pb': [87, 89, 97],
    'Bi': [158, 79, 181],
    'Po': [171, 92, 0],
    'At': [117, 79, 69],
    'Rn': [66, 130, 150],
    'Fr': [66, 0, 102],
    'Ra': [0, 125, 0],
    'Ac': [112, 171, 250],
    'Th': [0, 186, 255],
    'Pa': [0, 161, 255],
    'U':  [0, 143, 255],
    'Np': [0, 128, 255],
    'Pu': [0, 107, 255],
    'Am': [84, 92, 242],
    'Cm': [120, 92, 227],
    'Bk': [138, 79, 227],
    'Cf': [161, 54, 212],
    'Es': [179, 31, 212],
    'Fm': [179, 31, 186],
    'Md': [179, 13, 166],
    'No': [189, 13, 135],
    'Lr': [199, 0, 102],
    'Rf': [204, 0, 89],
    'Db': [209, 0, 79],
    'Sg': [217, 0, 69],
    'Bh': [224, 0, 56],
    'Hs': [230, 0, 46],
    'Mt': [235, 0, 38]
}

atomic_number_map = {
    'H':  1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B':  5,
    'C':  6,
    'N':  7,
    'O':  8,
    'F':  9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P':  15,
    'S':  16,
    'Cl': 17,
    'Ar': 18,
    'K':  19,
    'Ca': 20,
    'Sc': 21,
    'Ti': 22,
    'V':  23,
    'Cr': 34,
    'Mn': 25,
    'Fe': 26,
    'Co': 27,
    'Ni': 28,
    'Cu': 29,
    'Zn': 30,
    'Ga': 31,
    'Ge': 32,
    'As': 33,
    'Se': 34,
    'Br': 35,
    'Kr': 36,
    'Rb': 37,
    'Sr': 38,
    'Y':  39,
    'Zr': 40,
    'Nb': 41,
    'Mo': 42,
    'Tc': 43,
    'Ru': 44,
    'Rh': 45,
    'Pd': 46,
    'Ag': 47,
    'Cd': 48,
    'In': 49,
    'Sn': 50,
    'Sb': 51,
    'Te': 52,
    'I':  53,
    'Xe': 54,
    'Cs': 55,
    'Ba': 56,
    'La': 57,
    'Ce': 58,
    'Pr': 59,
    'Nd': 60,
    'Pm': 61,
    'Sm': 62,
    'Eu': 63,
    'Gd': 64,
    'Tb': 65,
    'Dy': 66,
    'Ho': 67,
    'Er': 68,
    'Tm': 69,
    'Yb': 70,
    'Lu': 71,
    'Hf': 72,
    'Ta': 73,
    'W':  74,
    'Re': 75,
    'Os': 76,
    'Ir': 77,
    'Pt': 78,
    'Au': 79,
    'Hg': 80,
    'Tl': 81,
    'Pb': 82,
    'Bi': 83,
    'Po': 84,
    'At': 85,
    'Rn': 86,
    'Fr': 87,
    'Ra': 88,
    'Ac': 89,
    'Th': 90,
    'Pa': 91,
    'U':  92,
    'Np': 93,
    'Pu': 94,
    'Am': 95,
    'Cm': 96,
    'Bk': 97,
    'Cf': 98,
    'Es': 99,
    'Fm': 100,
    'Md': 101,
    'No': 102,
    'Lr': 103,
    'Rf': 104,
    'Db': 105,
    'Sg': 106,
    'Bh': 107,
    'Hs': 108,
    'Mt': 109
}
#
