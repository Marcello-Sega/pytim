# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: ContactAngle
    ====================
"""

import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import warnings

class ContactAngle(object):
    """ Base class for contact angle calculation
    """

    # todo pass surface normal and adapt algorithms.
    def __init__(self, droplet, substrate, zcut=5.0, debug=False):
        """
            Computes radial profiles and contact angles using different approaches

            :param AtomGroup fluid     : Atom group representative of the droplet
            :param AtomGroup substrate : Atom group representative of the substrate
            :param float     zcut      : don't consider contributions from atoms nearer than this to the substrate

            Example:

            >>> import MDAnalysis as mda
            >>> import numpy as np
            >>> import pytim
            >>> from pytim import observables
            >>> from pytim.datafiles import *
            >>>
            >>> u = mda.Universe(WATER_DROPLET_CYLINDRICAL_GRO,WATER_DROPLET_CYLINDRICAL_XTC)
            >>> droplet = u.select_atoms("name OW")
            >>> substrate = u.select_atoms("name C")
            >>>
            >>> CA = observables.ContactAngleGitim(droplet, substrate)
            >>>
            >>> inter = pytim.GITIM(universe=u,group=droplet, molecular=False,alpha=2.5,cluster_cut=3.4, biggest_cluster_only=True)
            >>>
            >>> for ts in u.trajectory[::]:
            ...     CA.sample(inter)
            >>>
            >>> np.round(CA.contact_angle,2)
            79.52

        """

        self.droplet = droplet
        self.substrate = substrate
        self.universe = self.droplet.universe
        self.trajectory = self.droplet.universe.trajectory
        self.zcut = zcut
        self.nframes = 0
        self.debug = debug
        masses = self.universe.atoms.masses.copy()
        masses[self.universe.atoms.masses == 0.0] = 40.0
        self.universe.atoms.masses = masses.copy()

    def _offset_positions(self, group):
        if id(self.droplet.universe) != id(group.universe):
            raise RuntimeError('Universes are not the same')
        com = self.droplet.center_of_mass()
        zoff = np.max(self.substrate.atoms.positions[:, 2])
        com[2] = zoff
        pos = group.positions - com
        pos = pos[pos[:, 2] > self.zcut]
        return pos

    @staticmethod
    def arc(r, center, rad):
        return center+np.sqrt(rad**2-r**2)

    @staticmethod
    def fit_arc_(z, r, rmin=None, rmax=None, use_points=False, p0=None):
        """ fit an arc through the profile z(r) sampled by the class

            :param float rmin       : minimum radius used for the fit (default: 0)
            :param float rmax       : maximum radius used for the fit (default: maximum from dataset)
            :param bool  use_points : False: use the profile estimate, True: use the surface atoms positions (only)
        """
        def arc(r, center, R):
            return center+np.sqrt(R**2-r**2)

        z = np.asarray(z)
        r = np.asarray(r)

        con = np.logical_and(np.isfinite(z), np.isfinite(r))
        z, r = z[con], r[con]

        if rmax is None:
            rmax = np.nanmax(r)
            rmin = np.nanmin(r)
        cond = np.logical_and(r > rmin, r < rmax)
        if p0 is None:
            p0 = (-20., 110.)
        popt, pcov = curve_fit(arc, r[cond], z[cond], p0=p0)
        center, rad = popt
        rad = np.abs(rad)
        base_radius = np.sqrt(rad**2-center**2)
        costheta = -center/rad
        return (rad, base_radius, costheta, center)

    @staticmethod
    def fit_ellipse_(r, z, yy=None):
        ZZ = np.asarray(z)
        RR = np.asarray(r)
        idx1 = np.isnan(RR)
        idx2 = np.isnan(ZZ)
        idx = np.logical_and(~idx1, ~idx2)  # nan removed
        R_, Z_ = RR[idx], ZZ[idx]
        x1, y1 = R_, Z_
        X = x1[:, np.newaxis]
        Y = y1[:, np.newaxis]
        # Formulate and solve the least squares problem ||Ax - b ||^2
        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = np.ones_like(X)
        a = np.linalg.lstsq(A, b)[0].squeeze()
        # Print the equation of the ellipse in standard form
        print('Ellipse equ: {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 0'.format(
            a[0], a[1], a[2], a[3], a[4]))
        if yy is None:
            yy = 0  # positon of substrate
        bc = a[1]*yy+a[3]
        cc = a[2]*yy**2+a[4]*yy-1
        x1 = (-bc+np.sqrt(bc**2-4*a[0]*cc))/(2*a[0])
        x2 = (-bc-np.sqrt(bc**2-4*a[0]*cc))/(2*a[0])
        m1 = -(2*a[0]*x1+a[3]+a[1]*yy)/(a[1]*x1+a[4]+2*a[2]*yy)
        m2 = -(2*a[0]*x2+a[3]+a[1]*yy)/(a[1]*x2+a[4]+2*a[2]*yy)
        theta1 = np.arctan(m1)*180/np.pi
        theta2 = np.arctan(m2)*180/np.pi
        return a, theta1, theta2

    @staticmethod
    def arc_ellipse(x, rmin=None, rmax=None, bins=None):
        if rmax is None:
            rmax, rmin, bins = 80, -80, 3200
        val = np.linspace(rmin, rmax, bins)
        bb = x[1]*val+x[4]
        cc = x[0]*val*val+x[3]*val-1
        yyP = (-bb+np.sqrt(bb*bb - 4*x[2]*cc))/(2*x[2])
        yyN = (-bb-np.sqrt(bb*bb - 4*x[2]*cc))/(2*x[2])
        yy = list(yyP)+list(yyN)
        xx = list(val)+list(val)
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        idy = ~np.isnan(yy)
        return xx[idy], yy[idy]

    def fit_arc(self, rmin=0, rmax=None, use_points=None, p0=None):
        """ fit an arc through the profile z(r) sampled by the class

            :param float rmin       : minimum radius used for the fit (default: 0)
            :param float rmax       : maximum radius used for the fit (default: maximum from dataset)

        """
        try:
            z, r = self.z, self.r

        except AttributeError:
            raise RuntimeError(
                'something is wrong: no surface atoms or not gibbs surface present')

        return self.fit_arc_(z, r, rmin=rmin, rmax=rmax, p0=p0)

    def fit_arc_ellipse(self):
        try:
            z, r = self.z, self.r
        except AttributeError:
            raise RuntimeError(
                'something is wrong: no surface atoms or not gibbs surface present')

        return self.fit_ellipse_(r, z, yy=None)

    def arc_function(self, r, rmin=0, rmax=None, use_points=False, base_cut=None):
        rad, br, ct, center = self.fit_arc(
            rmin=rmin, rmax=rmax, use_points=use_points)
        return self.arc(r, center, rad)

    @property
    def contact_angle(self):
        rad, br, ct, center = self.fit_arc()
        return np.arccos(ct)*180./np.pi

    @property
    def base_radius(self):
        rad, br, ct, center = self.fit_arc()
        return br

    @property
    def radius(self):
        rad, br, ct, center = self.fit_arc()
        return rad

    @property
    def center(self):
        rad, br, ct, center = self.fit_arc()
        return center

    def droplet_size(self, direction):
        delta = np.max(self.inter.atoms.positions[:, direction])
        delta = delta - np.min(self.inter.atoms.positions[:, direction])
        return np.abs(delta)

    def remove_COM(self, droplet, rel_tol=0.1):
        if self.removeCOM is None: return
        if type(self.removeCOM) == type(1):
            self.removeCOM = [self.removeCOM]
        box = droplet.universe.dimensions[:3]
        for self.removeCOM in self.removeCOM:
            pos = droplet.universe.atoms.positions.copy()
            while self.droplet_size(self.removeCOM) > box[self.removeCOM]*(1. - rel_tol) :
                pos[:, self.removeCOM] +=  box[self.removeCOM] * rel_tol
                droplet.universe.atoms.positions = pos.copy()
                pos = droplet.universe.atoms.pack_into_box()
                droplet.universe.atoms.positions = pos.copy()
            try: 
                com = self.inter.atoms.center_of_mass()[self.removeCOM]
            except:    
                com = droplet.atoms.center_of_mass()[self.removeCOM]
            pos[:, self.removeCOM] -= com
            droplet.universe.atoms.positions = pos.copy()

            pos[:, self.removeCOM] += box[self.removeCOM]/2
            droplet.universe.atoms.positions = pos.copy()
            pos = droplet.universe.atoms.pack_into_box()
            droplet.universe.atoms.positions = pos.copy()

class ContactAngleGitim(ContactAngle):
    """ eontactAngle class implementation with GITIM
    """

    def sample(self, inter, bins=100, pdbOut=None, binning='theta', periodic=None, removeCOM=None, base_cut=0.0):
        """ compute the height profile z(r) of a droplet

            :param int       bins     : number of bins used to collect statistics on points
            :param str       pdbOut   : optional pdb file where to store trajectory with tagged surface atoms
            :param str       binning  : either 'theta' or 'z': select anglular or Cartesian binning (along z)
            :param int       periodic : either None (assume a spherical cap droplet) or one of 0,1,2 (assume a 
                                        cylindrical droplet and selects the direction of its axis)
            :param int       removeCOM: either None (does not remove the COM motion) or one of 0,1,2 to remove
                                        the COM motion along that direction
            :param float     base_cut : elevation of the substrate, used to perform the fit that extracts the
                                        contact angle

        """
        droplet, substrate = self.droplet, self.substrate
        box = droplet.universe.dimensions[:3]
        try:
            _ = self.r_
        except:
            self.r_, self.z_, self.theta_, self.R_ = [], [], [], []

        self.nframes += 1
        self.base_cut = base_cut
        self.inter = inter
        self.removeCOM=removeCOM
        self.remove_COM(droplet)
        self.maxr = np.nanmax(droplet.universe.dimensions[:2])
        self.maxz = droplet.universe.dimensions[2]

        if pdbOut is not None:
            self.inter.writepdb(pdbOut, multiframe=True)

        if periodic is None:
            r, theta, z = self.spherical_coordinates()
        else:
            r, theta, z = self.cylindrical_coordinates(periodic)

        if binning == 'theta':
            self.sample_theta_R(bins, theta, r, z, base_cut)
        elif binning == 'z':
            self.sample_z_r(bins, z, r, base_cut)
        else:
            raise (ValueError("Wrong binning type"))

    def _compute_coords(self, z_, r_):
        # r_,distance from center;R_, projection on xy plane can be considered as x in 2d

        zmin = np.nanmin(z_)
        cut = zmin+self.base_cut
        cond = z_>cut

        R_ = np.sqrt(r_*r_ + z_*z_)[cond]
        theta_ = np.arccos(r_[cond]/R_)
        self.theta_ += list(theta_)
        self.r_ += list(r_[cond])
        self.z_ += list(z_[cond])

        r, z, theta = np.asarray(self.r_), np.asarray(
            self.z_), np.asarray(self.theta_)

        return r, theta, z

    def cylindrical_coordinates(self, periodic):
        pos = self._offset_positions(self.inter.atoms)
        dirs = np.array([0, 1])
        dd = dirs[dirs != periodic][0]
        r_ = pos[:, dd]
        z_ = pos[:, 2]
        return self._compute_coords(z_, r_)

    def spherical_coordinates(self):
        pos = self._offset_positions(self.inter.atoms)
        # this r_ is the distance from the center
        r_ = np.linalg.norm(pos[:, 0:2], axis=1)
        z_ = pos[:, 2]
        return self._compute_coords(z_, r_)

    def sample_theta_R(self, bins, theta, r, z, base_cut):
        R = np.sqrt(r*r+z*z)
        n_h, t_edges = np.histogram(theta, bins=bins, range=(
            0, np.pi), weights=None, density=False)
        R_h, t_edges = np.histogram(theta, bins=bins, range=(
            0, np.pi), weights=R,    density=False)
        cond = np.where(n_h>0)
        zz,rr = np.zeros(n_h.shape)+np.nan,np.zeros(n_h.shape)+np.nan
        zz[cond], rr[cond] = R_h[cond] * np.sin(t_edges[:-1][cond]) / \
            (1.*n_h[cond]), R_h[cond] * np.cos(t_edges[:-1][cond])/(1.*n_h[cond])
        zzmin = np.nanmin(zz)
        ercut = zzmin+base_cut
        self.z, self.r = zz[zz > ercut], rr[zz > ercut]

    def sample_z_r(self, bins, z, r, base_cut):
        print('binning z,r on',bins, 'bins up to r=',self.maxr)
        n_h, r_edges = np.histogram(r, bins=bins, range=(
            0, self.maxr), weights=None, density=False)
        z_h, r_edges = np.histogram(r, bins=bins, range=(
            0, self.maxr), weights=z,    density=False)
        cond = np.where(n_h>0)
        zz, rr = z_h[cond]/(1.*n_h[cond]), r_edges[:-1][cond]
        zzmin = np.nanmin(zz)
        ercut = zzmin+base_cut
#        self.z, self.r = zz[zz > ercut], rr[zz > ercut]
        self.z, self.r = zz,rr

    def fit_arc(self, rmin=0, rmax=None, use_points=False):
        """ fit an arc through the profile z(r) sampled by the class

            :param float rmin       : minimum radius used for the fit (default: 0)
            :param float rmax       : maximum radius used for the fit (default: maximum from dataset)
            :param bool  use_points : False: use the mean surface atoms positions
                                      True : use the surface atoms positions
        """

        try:
            if use_points:
                z, r = self.z_, self.r_
            else:
                z, r = self.z, self.r

        except AttributeError:
            raise RuntimeError(
                'something is wrong: no surface atoms or not gibbs surface present')

        return self.fit_arc_(z, r, rmin=rmin, rmax=rmax)

    def fit_points_ellipse(self, rmax=None, rmin=None, bins=None, yy=None, use_points=False):
        if rmax is None:
            rmax, rmin, bins = 80, -80, 3200
            yy = 0
        try:
            if use_points:
                z, r = self.z_, self.r_
            else:
                z, r = self.z, self.r
        except AttributeError:
            raise RuntimeError(
                'something is wrong: no surface atoms or not gibbs surface present')
        cofX, th1, th2 = self.fit_ellipse_(r, z, yy)

        return self.arc_ellipse(cofX, rmax=rmax, rmin=rmin, bins=bins)

        def sample_theta_R(self, bins, th_left, th_right, RR_left, RR_right):
            n_h_left, t_edges_left = np.histogram(
                th_left, bins=bins, range=(0, np.pi), weights=None, density=False)
            R_h_left, t_edges_left = np.histogram(th_left, bins=bins, range=(
                0, np.pi), weights=RR_left,    density=False)
            n_h_right, t_edges_right = np.histogram(
                th_right, bins=bins, range=(0, np.pi), weights=None, density=False)
            R_h_right, t_edges_right = np.histogram(th_right, bins=bins, range=(
                0, np.pi), weights=RR_right, density=False)
            self.left_z, self.left_r = R_h_left * np.sin(t_edges_left[:-1])/(1.*n_h_left),\
                R_h_left * np.cos(t_edges_left[:-1])/(1.*n_h_left)
            self.right_z, self.right_r = R_h_right * np.sin(t_edges_right[:-1])/(1.*n_h_right),\
                R_h_right * np.cos(t_edges_right[:-1])/(1.*n_h_right)


class ContactAngleGibbs(ContactAngle):
    """ ContactAngle class implementation using an estimate of the 
        Gibbs dividing surface
    """

    def sigmoid(self, r, r0, A, w):
        return A*(1.+np.tanh((r0-r)/w))/2.

    def sample(self, bins=100, params=None, binning='theta', periodic=None, removeCOM=None,base_cut=0.0):
        """ compute the height profile z(r) of a droplet

            :param int       bins     : number of bins used to collect statistics on points
            :param str       binning  : either 'theta' or 'z': select anglular or Cartesian binning (along z)
            :param int       periodic : either None (assume a spherical cap droplet) or one of 0,1,2 (assume a 
                                        cylindrical droplet and selects the direction of its axis)
            :param int       removeCOM: either None (does not remove the COM motion) or one of 0,1,2 to remove
                                        the COM motion along that direction
            :param float     base_cut : elevation of the substrate, used to perform the fit that extracts the
                                        contact angle

        """

        droplet, substrate = self.droplet, self.substrate
        self.nframes += 1
        self.removeCOM = removeCOM
        self.remove_COM(droplet)
        pos = droplet.universe.atoms.positions 
        if periodic is None:
            r = np.linalg.norm(pos[:, 0:2], axis=1)
            z = pos[:, 2]
            R = np.sqrt(r*r + z*z)
        else:
            dirs = np.array([0, 1])
            dd = dirs[dirs != periodic][0]
            r = np.abs(pos[:, dd])
            z = pos[:, 2]
            R = np.sqrt(r*r+z*z)
        # Those rare cases of molecules close to the origin will be assigned theta=0
        # This should not affect the statistics 
        cond = np.abs(R)<1e-5
        R[cond]=1e-5
        r[cond]=1e-5
        theta = np.arccos(r/R)
        maxr = np.nanmax(droplet.universe.dimensions[:2])
        maxR = maxr
        maxz = droplet.universe.dimensions[2]
        if binning == 'z':
            # histogram for this frame
            H_, zedge, redge = np.histogram2d(
                z, r, bins, [[0, maxz], [0, maxr]])
            zedge = (zedge[:-1]+zedge[1:])/2.
            redge = (redge[:-1]+redge[1:])/2.
        elif binning == 'theta':
            H_, thetaedge, Redge = np.histogram2d(
                theta, R, bins, [[0, np.pi/2.], [30, maxR]])
            thetaedge = (thetaedge[:-1]+thetaedge[1:])/2.
            Redge = (Redge[:-1]+Redge[1:])/2.
        else:
            raise ValueError("binning can be only 'z' or 'theta'")

        # cumulative histogram
        try:
            self.H
            self.H += H_

        except:
            self.H = H_

        # normalize by number of frames and metric factor

        # fit the histogram with a sigmoidal function, determine a Gibbs dividing distance for each z-slice
        # we redo this every frame, it does not take much time, and we always have the up-to-date location
        # of the Gibbs dividing surface with maximum statistics

        # note the change of geometric meaning of the variables redge, thetaedge, Redge depending on binning/periodic
        if binning == 'z':
            if periodic is None:
                H_ = self.H / (self.nframes*redge)
            else:
                H_ = self.H / self.nframes

        elif binning == 'theta':
            # theta is the angle between the position vector and the xy plane, hence cos(theta) in the 
            # normalization
            if periodic is None:
                H_ = self.H / (self.nframes*Redge**2 * np.cos(thetaedge))
            else:
                H_ = self.H / (self.nframes*Redge)

        parms, zvals, rvals = [], [], []
        # we go through each z or theta slab and fit the
        # sigmoid vs R/r
        for i, h in enumerate(H_):
            cond = h > 0
            if np.sum(cond) < 5 : continue # bad statistics, typically on top of the drop
            if binning == 'z': _edge = redge
            elif binning == 'theta': _edge = Redge
            if params is None:
                # some heuristics
                maxh =np.max(h)
                r0 = np.sum(h[h>maxh/2]) / len(h)
                r0 = r0 * (_edge[-1]-_edge[0])
                rho0 = np.mean(h[h>maxh/2])
                w0 = np.sum(np.logical_and(h>maxh/4,h<3.*maxh/4)) * (_edge[1]-_edge[0])
                p0 = (r0, rho0, w0)
            else:
                p0 = params
            try:
                popt, pcov = curve_fit(self.sigmoid, _edge[cond], h[cond], p0=p0)
                if np.isfinite(pcov[0][0]) and popt[0] != p0[0] and popt[0] > 0 and popt[0] < maxR:
                   parms.append(list(popt))
                   if binning == 'z':
                       zvals.append(zedge[i])
                       rvals.append(popt[0])

                   else:
                       rad = popt[0]
                       zvals.append(rad * np.sin(thetaedge[i]))
                       rvals.append(rad * np.cos(thetaedge[i]))
            except TypeError as e:
                warnings.warn(str(e))
                pass  # not enough points in slice i
            except ValueError as e:
                warnings.warn(str(e))
                # not enough points (another version of numpy throws this error? check)
                pass
            except RuntimeError as e:
                warnings.warn(str(e))

        # check here that we have a reasonable number of points. Handle this
        zvals = np.array(zvals)
        rvals = np.array(rvals)
        parms = np.array(parms)
        self.z, self.r = zvals[zvals > base_cut], rvals[zvals > base_cut ]

