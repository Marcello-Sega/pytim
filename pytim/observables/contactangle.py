# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: ContactAngle
    ====================
"""

import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree


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

    def remove_com(self, group):
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

        from scipy.optimize import curve_fit

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
        #print("center=",center, "radius=",rad)
        rad = np.abs(rad)
        base_radius = np.sqrt(rad**2-center**2)
        # old from marcello costheta = -np.sin(center/rad)
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
        #x_coord = np.linspace(rmin,rmax,bins)
        #y_coord = np.linspace(zmin,zmax,bins)
        #X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        #Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
        # rmin,rmax,bins=int(rmin),int(rmax),int(bins)
        val = np.linspace(rmin, rmax, bins)
        bb = x[1]*val+x[4]
        # aa=x[2]
        cc = x[0]*val*val+x[3]*val-1
        yyP = (-bb+np.sqrt(bb*bb - 4*x[2]*cc))/(2*x[2])
        yyN = (-bb-np.sqrt(bb*bb - 4*x[2]*cc))/(2*x[2])
        yy = list(yyP)+list(yyN)
        xx = list(val)+list(val)
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        idy = ~np.isnan(yy)
        # return X_coord,Y_coord,Z_coord,xx[idy],yy[idy]
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
        #print('fitted radius, center, base_radius, cos(theta):',rad,center,br,ct)
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

    def remove_COM(self, removeCOM, droplet, inter, alpha, box):
        if removeCOM is not None:
            while(self.droplet_size(removeCOM) > box[removeCOM] - 4 * alpha):
                pos = droplet.universe.atoms.positions.copy()
                pos[:, removeCOM] += alpha
                droplet.universe.atoms.positions = pos.copy()
                pos = droplet.universe.atoms.pack_into_box()
                droplet.universe.atoms.positions = pos.copy()
            com = self.inter.atoms.center_of_mass()[removeCOM]
            pos = droplet.universe.atoms.positions.copy()
            pos[:, removeCOM] -= com
            droplet.universe.atoms.positions = pos.copy()

            pos[:, removeCOM] += box[removeCOM]/2
            droplet.universe.atoms.positions = pos.copy()

            droplet.universe.atoms.positions = pos.copy()
            pos = droplet.universe.atoms.pack_into_box()
            droplet.universe.atoms.positions = pos.copy()
        return droplet.universe.atoms.positions


class ContactAngleGitim(ContactAngle):
    """ ContactAngle class implementation with GITIM
    """

    def sample(self, inter, bins=100, cut=3.5, alpha=2.5, pdbOut=None, binning='theta', periodic=None, removeCOM=None, base_cut=0.0):
        """ compute the height profile z(r) of a droplet

            :param int       bins     : number of slices along the z axis (across the whole box z-edge)
            :param float     cut      : cut-off for the clustering algorithm that separates liquid from vapor
            :param float     alpha    : probe sphere radius for GITIM
            :param str       pdbOut   : optional pdb file where to store trajectory with tagged surface atoms

            NOTES: 
            1) not tested with the molecular option of GITIM
            2) bins in x,y and z directions are the same, this might be a problem for boxes with large aspect 
               ratios
            3) make removeCOM=0 to remove movement of droplet along x axis

        """
        droplet, substrate = self.droplet, self.substrate
        box = droplet.universe.dimensions[:3]
        try:
            _ = self.r_
        except:
            self.r_, self.z_, self.theta_, self.R_ = [], [], [], []

        self.nframes += 1

        self.inter = inter

        self.remove_COM(removeCOM, droplet, inter, alpha, box)
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
        R_ = np.sqrt(r_*r_ + z_*z_)
        theta_ = np.arccos(r_/R_)
        self.theta_ += list(theta_)
        self.r_ += list(r_)
        self.z_ += list(z_)
        #self.R_ += list(R_)

        r, z, theta = np.asarray(self.r_), np.asarray(
            self.z_), np.asarray(self.theta_)

        return r, theta, z

    def cylindrical_coordinates(self, periodic):
        pos = self.remove_com(self.inter.atoms)
        dirs = np.array([0, 1])
        dd = dirs[dirs != periodic][0]
        r_ = pos[:, dd]  # change to this if you face error "np.abs(pos[:,dd])"
        z_ = pos[:, 2]
        # R_=np.sqrt(R_*R_+z_*z_)
        # print(z_,r_)
        # print(r_.shape,z_.shape)
        return self._compute_coords(z_, r_)

    def spherical_coordinates(self):
        pos = self.remove_com(self.inter.atoms)
        #r_ = np.linalg.norm(pos[:,0:2],axis=1)
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
        #cond = np.where(n_h>0)
        zz, rr = R_h * np.sin(t_edges[:-1]) / \
            (1.*n_h), R_h * np.cos(t_edges[:-1])/(1.*n_h)
        zzmin = np.nanmin(zz)
        ercut = zzmin+base_cut
        self.z, self.r = zz[zz > ercut], rr[zz > ercut]
        # print("sample",z,r)

    def sample_z_r(self, bins, z, r, base_cut):
        #print("I love maxr",self.maxr)
        n_h, r_edges = np.histogram(r, bins=bins, range=(
            0, self.maxr), weights=None, density=False)
        z_h, r_edges = np.histogram(r, bins=bins, range=(
            0, self.maxr), weights=z,    density=False)
        zz, rr = z_h/(1.*n_h), r_edges[:-1]
        zzmin = np.nanmin(zz)
        ercut = zzmin+base_cut
        self.z, self.r = zz[zz > ercut], rr[zz > ercut]
        #self.z , self.r =   z_h/(1.*n_h), r_edges[:-1]

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

        print(cofX, th1, th2)
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

    def sample(self, bins=100, params=None, binning='theta', periodic=None, base_cut=0.0):
        """ compute the height profile z(r) of a droplet
            :param int       bins     : number of slices along the z axis (across the whole box z-edge)

        """
        from scipy.optimize import curve_fit

        droplet, substrate = self.droplet, self.substrate
        self.nframes += 1

        pos = self.remove_com(droplet)
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
            # print(H_[H_>0],zedge,redge)
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
        if binning == 'z':
            if periodic is None:
                H_ = self.H / (self.nframes*redge)
            else:
                H_ = self.H / (self.nframes)  # need to check

        elif binning == 'theta':
            if periodic is None:
                H_ = self.H / (self.nframes*Redge**2 * np.cos(thetaedge))

            else:
                H_ = self.H / (self.nframes*Redge * np.cos(thetaedge))  # check
            #print(self.H, H_, self.nframes)

        parms, zvals, rvals = [], [], []
        for i, h in enumerate(H_):
            cond = h > 1e-6
            if params is None:
                p0 = (40., 10.0, 0.8)
            else:
                p0 = params
            try:
                if binning == 'z':
                    popt, pcov = curve_fit(
                        self.sigmoid, redge[cond], h[cond], p0=p0)
                    #popt, pcov = curve_fit(self.exp_sigmoid, redge[cond], h[cond], p0=p0)

                if binning == 'theta':
                    popt, pcov = curve_fit(
                        self.sigmoid, Redge[cond], h[cond], p0=p0)

            except TypeError:
                pass  # not enough points in slice i
            except ValueError:
                # not enough points (another version of numpy throws this error? check)
                pass
            except RuntimeError:
                pass  # fit failed
            try:
                if np.isfinite(pcov[0][0]) and popt[0] != p0[0] and popt[0] > 0 and popt[0] < maxR:
                    #print(pcov[0][0], popt[0])
                    # if np.isfinite(pcov[0][0]) and popt[0]>0 and popt[0]<maxr:
                    #print("I am here")
                    if True or 2*popt[2] < popt[0]:
                        parms.append(list(popt))
                        if binning == 'z':
                            zvals.append(zedge[i])
                            rvals.append(popt[0])

                        else:
                            rad = popt[0]
                            zvals.append(rad * np.sin(thetaedge[i]))
                            rvals.append(rad * np.cos(thetaedge[i]))

            except UnboundLocalError:
                pass

        # check here that we have a reasnoable number of points. Handle this
        zvals = np.array(zvals)
        rvals = np.array(rvals)
        # print(zvals,rvals)
        parms = np.array(parms)
        flcut = np.nanmin(zvals)+base_cut
        #print (zvals.shape,parms.shape)
        self.z, self.r = zvals[zvals > flcut], rvals[zvals > flcut]
