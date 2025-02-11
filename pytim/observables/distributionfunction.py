# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: DistributionFunction
    ============================
"""
from __future__ import print_function
import numpy as np
from . import Position, RelativePosition


class DistributionFunction(object):
    r"""Calculates a 3d distribution function of some observable from two
    groups.

    The two functions must return an array (of scalars or of vectors)
    having the same size of the group. The scalar product between the
    two functions is used to weight the distriution function.

    .. math::

          sdf(x,y,z) = \frac{1}{N}\left\langle \sum_{i\neq j} \delta(x-x_i,y-y_i,z-z_i)\
            f_1(r_i,v_i)\cdot f_2(r_j,v_j) \right\rangle


    :param double max_distance:     compute the sdf up to this distance along 
                                    the axes. If a list or an array is supplied,
                                    each element will determine the maximum distance
                                    along each axis.
                                    If 'full' is supplied (default) computes
                                    it up to half of the smallest box side.
    :param int nbins:               number of bins used for the sampling in all directions. 
                                    Supply a list or an array to use a different number of
                                    bins along each of the directions

    :param Observable observable:   observable for the first group
    :param Observable observable2:  observable for the second group

    Example:

    >>> import MDAnalysis as mda
    >>> import numpy as np
    >>> import pytim
    >>> from pytim import observables
    >>> from pytim.datafiles import *
    >>>
    >>> u = mda.Universe(WATER_GRO,WATER_XTC)
    >>> oxygens = u.select_atoms("name OW")
    >>>
    >>> nres = observables.NumberOfResidues()
    >>>
    >>> rdf = observables.RDF(u,nbins=120,observable=nres,observable2=nres)
    >>>
    >>> interface = pytim.ITIM(u,alpha=2.,group=oxygens,cluster_cut=3.5,molecular=False)
    >>>
    >>> for ts in u.trajectory[::50]:
    ...     layer=interface.layers[0,0]
    ...     rdf.sample(layer,layer)
    >>> rdf.count[0]=0
    >>> np.savetxt('RDF3D.dat', np.column_stack((rdf.bins,rdf.rdf)))


    Note that one needs to specify neither both groups, not both observables.
    If only the first group (observable) is specified, the second is assumed
    to be the same as the first, as in the following example:

    >>> rdf1 = observables.RDF(u,observable=nres)
    >>> rdf2 = observables.RDF(u,observable=nres)
    >>> rdf3 = observables.RDF(u,observable=nres,observable2=nres)
    >>>
    >>> rdf1.sample(layer)
    >>> rdf2.sample(layer,layer)
    >>> rdf3.sample(layer,layer)
    >>> print (np.all(rdf1.rdf[:]==rdf2.rdf[:]))
    True
    >>> print (np.all(rdf1.rdf[:]==rdf3.rdf[:]))
    True

    """
    cartesian_coords = ['x', 'y', 'z']
    spherical_coords = ['r', 'phi', 'theta']

    def __init__(self,
                 universe,
                 order,
                 nbins=75,
                 start=None,
                 stop=None,
                 step=None,
                 generalized_coordinate=None,
                 generalized_coordinate2=None,
                 observable=None,
                 observable2=None,
                 max_distance=None,
                 coords_in=['x', 'y', 'z'],
                 coords_out=['x', 'y', 'z'],
                 kargs1=None,
                 kargs2=None):

        kargs1 = kargs1 or {}
        kargs2 = kargs2 or {}

        self.order = order
        # let's start setting some default values
        self.universe = universe
        self.coords_in = coords_in
        self.coords_out = coords_out
        self._set_default_values(generalized_coordinate, max_distance, nbins)
        self.settings = {'bins': self.nbins, 'range': self.range}
        self.observable = observable
        self.kargs1 = kargs1
        self.kargs2 = kargs2

        if observable2 is None:
            self.observable2 = observable
        else:
            self.observable2 = observable2

        self.n_frames = 0
        self.volume = 0.0
        self.n_normalize = 0
        count, edges = np.histogramdd(
            np.array([[0.] * self.dimensions_out]), bins=self.nbins, range=self.range)
        self.count = count * 0.0
        self.edges = edges
        self.g1 = self.universe.atoms
        self.g2 = None
        self._rdf = self.count

    def _set_default_values(self, generalized_coordinate, max_distance, nbins):

        self.dimensions = np.asarray(self.coords_in).shape[0]
        self.dimensions_out = np.asarray(self.coords_out).shape[0]
        self.generalized_coordinate = generalized_coordinate

        self.cartesian = np.any(
            [coord in self.cartesian_coords for coord in self.coords_out])
        self.spherical = np.any(
            [coord in self.spherical_coords for coord in self.coords_out])

        if (self.cartesian and self.spherical) or (not self.cartesian and not self.spherical):
            raise ValueError("Can pass either any of Cartesian ['x','y','z'] or of spherical ['r','phi','theta'] coordinates ")
        if self.spherical:
            all_coords = self.spherical_coords
        if self.cartesian:
            all_coords = self.cartesian_coords

        self.dirmask_out = np.array(
            [np.where(np.asarray(all_coords) == c)[0][0] for c in self.coords_out])

        if self.generalized_coordinate is None:
            if self.order == 1:
                self.generalized_coordinate = Position(self.coords_in, cartesian=self.cartesian,
                                                       spherical=self.spherical)
            if self.order == 2:
                self.generalized_coordinate = RelativePosition(self.coords_in, cartesian=self.cartesian,
                                                               spherical=self.spherical)

        if max_distance is None:
            if (isinstance(self.generalized_coordinate, Position) or
                    isinstance(self.generalized_coordinate, RelativePosition)):
                if self.cartesian:
                    self.max_distance = self.universe.dimensions[self.dirmask_out] / 2.
                if self.spherical:
                    _md = [np.min(
                        self.universe.dimensions[self.generalized_coordinate.dirmask]) / 2., 2. * np.pi, np.pi]
                    self.max_distance = np.array(_md)[self.dirmask_out]
            else:
                # There can be two signatures (so far): an observable that takes one or two groups
                # we assume the generalized_coordinate returns an ndarray, e.g. [ [x1,y1,z1], ...]
                # and we take half of the maximum extension in each of the directions
                # This should fit most cases, as a first approximation.
                try:
                    self.max_distance = np.max(
                        self.generalized_coordinate.compute(universe.atoms), axis=0) / 2.
                    self.max_distance -= np.min(
                        self.generalized_coordinate.compute(universe.atoms), axis=0) / 2.
                except:
                    self.max_distance = np.max(self.generalized_coordinate.compute(
                        universe.atoms, universe.atoms), axis=0) / 2.
                    self.max_distance -= np.min(self.generalized_coordinate.compute(
                        universe.atoms, universe.atoms), axis=0) / 2.
        else:
            if np.asarray(max_distance).shape == ():  # a scalar
                self.max_distance = np.array(
                    [max_distance * 1.0] * self.dirmask_out.shape[0])
            else:  # an array
                try:  # check if it has the right dimension
                    self.max_distance = np.array(
                        [0.] * self.dirmask_out.shape[0]) + max_distance
                except ValueError:
                    print('PUT HERE A PROPER ERROR')

        if self.cartesian:
            self.range = np.append(-self.max_distance,
                                   self.max_distance).reshape(2, self.dimensions_out).T
        if self.spherical:
            self.range = np.append(np.asarray([0., 0., 0.])[
                                   self.dirmask_out], self.max_distance).reshape(2, self.dimensions_out).T
        # process nbins

        if np.asarray(nbins).shape == ():  # a scalar
            self.nbins = np.array([nbins] * self.dimensions_out, dtype=int)
        else:
            try:
                self.nbins = np.array([0.] * self.dimensions_out) + nbins
            except ValueError:
                print('PUT HERE ANOTHER PROPER ERROR')

    def _compute_observable(self, ka1, ka2):
        try:
            fg1 = self.observable.compute(self.g1, ka1)
        except:
            fg1 = self.observable.compute(self.g1)

        if (self.g1 == self.g2 and self.observable == self.observable2):
            fg2 = fg1
        else:
            try:
                fg2 = self.observable2.compute(self.g2, ka2)
            except:
                fg2 = self.observable2.compute(self.g2)

        try:
            error = (fg1.shape[0] != self.g1.n_atoms
                     or fg2.shape[0] != self.g2.n_atoms)
        except:
            error = True
        return fg1, fg2, error

    def _determine_weights(self, fg1, fg2):
        # both are (arrays of) scalars
        if len(fg1.shape) == 1 and len(fg2.shape) == 1:
            weights = np.outer(fg1, fg2)
        # both are (arrays of) vectors
        elif len(fg1.shape) == 2 and len(fg2.shape) == 2:
            # TODO: tests on the second dimension...
            weights = np.dot(fg1, fg2.T)
        else:
            raise Exception("Error, shape of the observable output not handled"
                            "in RDF")
        return weights.ravel()

    def sample(self, g1=None, g2=None, kargs1=None, kargs2=None):
        kargs1 = kargs1 or {}
        kargs2 = kargs2 or {}
        self.n_frames += 1
        self.g2 = g2
        if g1 is not None:
            self.g1 = g1
        if g2 is not None:
            self.g2 = g2
        ka1 = self.kargs1.copy()
        ka1.update(kargs1)
        ka2 = self.kargs2.copy()
        ka2.update(kargs2)

        if self.observable is not None:
            # determine weights, otherwise assumes number of atoms (default)
            fg1, fg2, error = self._compute_observable(ka1, ka2)

            if error is True:
                raise Exception(
                    "Error, the observable passed to RDF should output "
                    "an array (of scalar or vectors) the same size of "
                    "the group")

            # numpy.histogram accepts negative weights
            self.settings['weights'] = self._determine_weights(fg1, fg2)

        # This still uses MDA's distance_array. Pro: works also in triclinic
        # boxes. Con: could be faster (?)
        if self.g2 is None or len(self.g2) == 0:
            _distances = [
                np.zeros(len(self.g1), dtype=np.float64)] * self.dimensions
        else:
            _distances = [
                np.zeros((len(self.g1), len(self.g2)), dtype=np.float64)] * self.dimensions

        if self.g2 is None:
            try:
                _distances = self.generalized_coordinate.compute(self.g1)
            except AttributeError:  # not an observable, we assume the user know
                                   # what to pass here ...
                _distances = self.generalized_coordinate(self.g1, ka1)

        else:
            try:  # this kind of observable (depending on two groups) are not yet
                  # implemented. This is just an implementation draft for the general
                  # distribution function. TODO update this when needed
                # TODO: handle kwargs...
                _distances = self.generalized_coordinate.compute(
                    self.g1, self.g2)
            except AttributeError:  # not an observable, we assume the user knows
                                   # what to pass here ...
                _distances = self.generalized_coordinate.compute(
                    self.g1, self.g2)

        if len(_distances.shape) == 1:
            _distances = _distances.reshape(_distances.shape[0], 1)

        count = np.histogramdd(
            _distances[:, self.dirmask_out], **self.settings)[0]

        self.count += count

        box = self.universe.dimensions
        self.volume += np.prod(box[:3])
        if self.g2 is None or len(self.g2) == 0:
            self.n_normalize += len(self.g1)
        else:
            self.n_normalize += len(self.g1) * len(self.g2)

    @property
    def distribution(self, density=False):
        # no metric factors involved at this level.
        # these cases are handled by the derived classes
        # as, e.g. the RDF
        if density is False:
            self._distribution = self.count / self.n_frames
        else:
            # Volume of each voxel
            vol = np.prod([e[1] - e[0] for e in self.edges])
            # normalization
            density = self.n_normalize / self.volume

            self._distribution = self.count / (density * vol * self.n_frames)

        return self._distribution
