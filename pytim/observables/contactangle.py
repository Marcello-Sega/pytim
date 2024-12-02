# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: ContactAngle
    ====================
"""
import pytim
import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree
import scipy.linalg
import warnings
from numpy.linalg import solve
from scipy.optimize import curve_fit
from scipy.optimize import minimize


class ContactAngle(object):
    """ ContactAngle class implementation that uses interfacial atoms to compute\
        droplet profiles and contact angles using different approaches.

        :param PYTIM             interface:   Compute the contact angle for this interface
        :param AtomGroup         droplet:     Atom group representative of the droplet
        :param AtomGroup         substrate:   Atom group representative of the substrate
        :param int               periodic:    direction along which the system is periodic. Default: None, not periodic\
                                              If None, the code performs best fit to ellipsoids, otherwise, to ellipses\
                                              If not None, selects the direction of the axis of macroscopic translational\
                                              invariance: 0:x, 1:y, 2:z
        :param float             hcut:        don't consider contributions from atoms closer than this to the substrate\
                                              (used to disregard the microscopic contact angle)
        :param float             hcut_upper:  don't consider contributions from atoms above this distance from the substrate\
                                              default: None
        :param int               bins:        bins used for sampling the profile
        :param int or array_like removeCOM:   remove the COM motion along this direction(s). Default: None, does not remove COM motion

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
        >>> inter = pytim.GITIM(universe=u,group=droplet, molecular=False,alpha=2.5,cluster_cut=3.4, biggest_cluster_only=True)
        >>> 
        >>> # Contact angle calculation using interfacial atoms, angular bining for a cylindrical droplet
        >>> # with periodicity along the y (1) axis
        >>> 
        >>> CA = observables.ContactAngle(inter, substrate, periodic=1,bins=397,removeCOM=0,hcut=5)
        >>> for ts in u.trajectory[::]:
        ...     CA.sample()
        >>> # Instantaneous contact angle (last frame) by fitting a circle...
        >>> print(np.round(CA.contact_angle,2))
        90.58

        >>> 
        >>> # ... and using an elliptical fit:
        >>> left, right = CA.contact_angles
        >>> # left angle
        >>> print(np.round(np.abs(left),2))
        79.95

        >>> # right angle
        >>> print(np.round(right,2))
        83.84

        >>> # Contact angles from the averaged binned statistics of
        >>> # surface atoms' radial distance as a function of the azimuthal angle
        >>> np.around(CA.mean_contact_angles,1).tolist()
        [96.2, 100.7]

   """

    class Histogram(object):
        def __init__(self):
            self.x, self.y, self.h = np.array([]), np.array([]), np.array([])

    histo = Histogram()

    def __init__(self, inter, substrate, periodic=None, hcut=0.0, hcut_upper=None,
                 contact_cut=0.0, bins=100, removeCOM=None, store=False):
        self._sampled = False
        self.inter, self.substrate = inter, substrate
        self.universe, self.trajectory = self.inter.universe, self.inter.universe.trajectory
        self.hcut, self.hcut_upper = hcut, hcut_upper
        self.nframes, self.periodic = 0, periodic
        self.bins, self.removeCOM = bins, removeCOM
        self.contact_cut, self.store = contact_cut, store
        self.x, self.y, self.h = np.array([]), np.array([]), np.array(
            [])  # in 2d, x and y are radius and elevation
        if self.contact_cut == 'auto':
            tree = cKDTree(substrate.positions,
                           boxsize=substrate.universe.dimensions[:3])
            d, _ = tree.query(inter.atoms.positions, k=1)
            hist, bins = np.histogram(d, int(np.max(d)/0.5))
            # check the location of the maximum with resolution 0.5 Angstrom
            vmax, imax = np.max(hist), np.argmax(hist)
            # find where the distribution drops below 5% of the maximum; this could happen before and after the maximum
            idx = np.where(hist <= 0.05*vmax)[0]
            # select the first bin that satisfies the above condition and is after the maximum
            self.contact_cut = bins[idx[idx > imax][0]]

    @property
    def contact_angle(self):
        """ 
            The contact angle from the best-fititng circumference or sphere
            computed using the current frame
        """
        if self.periodic is None:
            raise NotImplementedError(
                'fit_sphere not implemented, use contact_angles instead of contact_angle')
        else:
            c, cp, theta, _, _ = self.fit_circle()
        self._polynomial_coefficients = c
        self._canonical_form = cp
        return theta

    @property
    def mean_contact_angle(self):
        """ 
            The contact angle from the best-fititng circumference 
            computed using the location of the interface averaged 
            over the sampled frames
        """
        if self.periodic is None:
            raise NotImplementedError(
                'fit_sphere not implemented, use contact_angles instead of contact_angle')
        else:
            c, cp, theta, _, _ = self.fit_circle(use='histogram')
        self._polynomial_coefficients = c
        self._canonical_form = cp
        return theta

    @property
    def contact_angles(self):
        """ 
            The contact angles from the best-fititng ellipse or 
            ellipsoid computed using the current frame
        """
        if self.periodic is None:
            c, cp, theta, _ = self.fit_ellipsoid()
        else:
            c, cp, theta, _ = self.fit_ellipse()
        self._polynomial_coefficients = c
        self._canonical_form = cp
        return theta

    @property
    def mean_contact_angles(self):
        """ 
            The contact angles from the best-fititng ellipse or 
            ellipsoid computed using the location of the interface 
            averaged over the sampled frames
        """
        if self.periodic is None:
            c, cp, theta, _ = self.fit_ellipsoid(use='histogram')
        else:
            c, cp, theta, _ = self.fit_ellipse(use='histogram')
        self._polynomial_coefficients = c
        self._canonical_form = cp
        return theta

    @property
    def polynomial_coefficients(self):
        try:
            return self._polynomial_coefficients
        except:
            raise RuntimeError('No fit has been performed yet')

    @property
    def canonical_form(self):
        try:
            return self._canonical_form
        except:
            raise RuntimeError('No fit has been performed yet')

    def _shifted_pos(self, group):
        com = group.center_of_mass()
        hoff = np.max(self.substrate.atoms.positions[:, 2])
        com[2] = hoff
        # removes the com along x,y and uses the topmost substrate atom to
        # define the location of the origin along the vertical axis
        pos = group.positions - com
        # further, we won't be using any coordinate that is below a certain
        # distance from the substrate surface
        pos = pos[pos[:, 2] > self.hcut]
        if self.hcut_upper is not None:
            pos = pos[pos[:, 2] < self.hcut_upper]
        return pos

    def _remove_COM(self, group, inter, alpha, box):
        def group_size(g, direction):
            p = g.atoms.positions[:, direction]
            return np.abs(np.max(p) - np.min(p))

        if self.removeCOM is not None:
            RC = self.removeCOM
            if type(RC) == int:
                RC = [RC]
            for axis in RC:
                while (group_size(group, axis) > box[axis] - 4 * alpha):
                    pos = group.universe.atoms.positions.copy()
                    pos[:, axis] += alpha
                    group.universe.atoms.positions = pos.copy()
                    pos = group.universe.atoms.pack_into_box()
                    group.universe.atoms.positions = pos.copy()
                com = self.inter.atoms.center_of_mass()[axis]
                pos = group.universe.atoms.positions.copy()
                pos[:, axis] -= com
                group.universe.atoms.positions = pos.copy()

                pos[:, axis] += box[axis]/2
                group.universe.atoms.positions = pos.copy()

                pos = group.universe.atoms.pack_into_box()
                group.universe.atoms.positions = pos.copy()
        return group.universe.atoms.positions

    def sample(self):
        """ samples the profile of a droplet, stores
            the current atomic coordinates of the liquid surface
            not in contact with the substrate in the reference
            frame of the droplet, and optionally the coordinates
            along the whole trajectory.
        """
        if self.contact_cut > 0:
            tree_substrate = cKDTree(
                self.substrate.atoms.positions, boxsize=self.substrate.universe.dimensions[:3])
            dists, _ = tree_substrate.query(self.inter.atoms.positions, k=1)
            self.liquid_vapor = self.inter.atoms[dists > self.contact_cut]
        else:
            self.liquid_vapor = self.inter.atoms

        box = self.substrate.universe.dimensions[:3]
        if self.store:
            try:
                _ = self._x
            except:
                self._x, self._y, self._h = [], [], []

        self.nframes += 1

        self._remove_COM(self.inter.atoms, self.liquid_vapor,
                         self.inter.alpha, box)
        self.maxr = np.max(self.substrate.universe.dimensions[:2])
        self.maxh = self.substrate.universe.dimensions[2]

        self.x, self.y, self.h = self.cartesian_coordinates(
            self.liquid_vapor.atoms)
        if self.store:
            self._x += list(self.x)
            self._h += list(self.h)
            if self.periodic is None:
                self._y += list(self.y)

        if self.periodic is None:
            r, theta, h, phi = self.spherical_coordinates(
                self.liquid_vapor.atoms)
            self.phi = phi.copy()
        else:
            r, theta, h, z = self.cylindrical_coordinates(
                self.liquid_vapor.atoms)

        self.r = r.copy()
        self.theta = theta.copy()
        self.h = h.copy()

        if self.periodic is None:
            pass  # TODO implement the 2d histogram
        else:
            self.sample_theta_R(theta, r, h)

    @staticmethod
    def rmsd_ellipse(p, x, N, check_coeffs=True):
        cp = ContactAngle._ellipse_general_to_canonical(p, check_coeffs)
        xx, yy = ContactAngle.ellipse(cp, N)
        pos = np.vstack([xx, yy]).T
        cond = np.logical_and(yy >= np.min(x[:, 1]), yy <= np.max(x[:, 1]))
        pos = pos[cond]
        return np.sqrt(np.mean(cKDTree(pos).query(x)[0]**2))

    @staticmethod
    def rmsd_ellipse_penalty(p, x, N, check_coeffs=True):
        rmsd = ContactAngle.rmsd_ellipse(p, x, N, check_coeffs)
        penalty = (4*p[0]*p[2]-p[1]**2-1)**2
        return rmsd + penalty

    @staticmethod
    def rmsd_ellipsoid(p, x, N, check_coeffs=True):
        """ RMSD between the points x and the ellipsoid defined by the general\
            parameters p of the associated polynomial.

            :param list    p:            general coefficients [a,b,c,f,g,h,p,q,r,d]
            :param ndarray x:            points coordinates as a (x,3)-ndarray
            :param int     N:            number of points on the ellipsoid that are\
                                         generated and used to compute the rmsd
            :param bool    check_coeffs: if true, perform additional checks

        """
        cp = ContactAngle._ellipsoid_general_to_affine(p, check_coeffs)
        xx, yy, zz = ContactAngle.ellipsoid(cp, N)
        pos = np.vstack([xx, yy, zz]).T
        cond = np.logical_and(zz >= np.min(x[:, 2]), zz <= np.max(x[:, 2]))
        pos = pos[cond]
        return np.sqrt(np.mean(cKDTree(pos).query(x)[0]**2))

    @staticmethod
    def rmsd_ellipsoid_penalty(p, x, N, check_coeffs=True):
        rmsd = ContactAngle.rmsd_ellipsoid(p, x, N, check_coeffs)
        violation = (4*p[0]*p[2]-p[1]**2)**2
        penalty = 0 if 4*p[0]*p[2]-p[1]**2 > 0 else violation
        return rmsd + penalty

    @staticmethod
    def rmsd_circle(p, x):
        R, x0, y0 = p
        d = np.linalg.norm(np.array([x0, y0])-x, axis=1) - R
        return np.sqrt(np.mean(d**2))

    @staticmethod
    def ellipse(parmsc, npts=100, tmin=0., tmax=2.*np.pi):
        """ Return npts points on the ellipse described by the canonical parameters\
            x0, y0, ap, bp, e, phi for values of the paramter between tmin and tmax.

            :param dict  parmsc: dictionary with keys: x0,y0,a,b,phi
            :param float tmin:   minimum value of the parameter
            :param float tmax:   maximum value of the parameter
            :param int   npts:   number of points to use

            :return:

            :tuple:  (x,y):       coordinates as numpy arrays

        """
        t = np.linspace(tmin, tmax, npts)
        x = parmsc['x0'] + parmsc['a'] * np.cos(t) * np.cos(
            parmsc['phi']) - parmsc['b'] * np.sin(t) * np.sin(parmsc['phi'])
        y = parmsc['y0'] + parmsc['a'] * np.cos(t) * np.sin(
            parmsc['phi']) + parmsc['b'] * np.sin(t) * np.cos(parmsc['phi'])
        return x, y

    @staticmethod
    def circle(parmsc, npts=100, tmin=0., tmax=2.*np.pi):
        """ Return npts points on the circle described by the canonical parameters
            R, x0, y0 for values of the paramter between tmin and tmax.

            :param dict  parmsc: dictionary with keys: R, x0, y0
            :param float tmin:   minimum value of the parameter
            :param float tmax:   maximum value of the parameter
            :param int   npts:   number of points to use

            :return:

            :tuple:      (x,y): coordinates as numpy arrays

        """
        t = np.linspace(tmin, tmax, npts)
        x = parmsc['x0'] + parmsc['R'] * np.cos(t)
        y = parmsc['y0'] + parmsc['R'] * np.sin(t)
        return x, y

    @staticmethod
    def ellipsoid(parmsc, npts=1000):
        """ Compute npts points on the ellipsoid described by the affine parameters T, v

            :param dict  parmsc: dictionary with keys: T (3x3 matrix), v (3x1 vector)
            :param int   npts:   number of points to use

            :return:
            :tuple: (x,y,z)     : coordinates of points on the ellipsoid as ndarrays

        """
        phi = np.arccos(2 * np.random.rand(npts) - 1)
        theta = 2 * np.pi * np.random.rand(npts)
        s = np.array([np.sin(phi) * np.cos(theta), np.sin(phi)
                     * np.sin(theta), np.cos(phi)])
        r = (np.array(parmsc['T'])@s).T + np.array(parmsc['v']).T
        return r[:, 0], r[:, 1], r[:, 2]

    @staticmethod
    def _fit_circle(hr, hh, nonlinear=True):
        """ fit an arc through the profile h(r) sampled by the class

            :param list hr:        list of arrays with the radial coordinates
            :param list hh:        list of arrays with the elevations
            :param bool nonlinear: use the more accurate minimization of the rmsd instead of the algebraic distance

            :return:
            :list:             : a list with the tuple (radius, base radius, cos(theta), center, rmsd)
                                for each bin. If only one bin is present, return just the tuple. 
        """
        parms = []
        for i in np.arange(len(hr)):
            r = hr[i]
            h = hh[i]
            if len(r) == 0:
                parms.append(None)
                break
            M = np.vstack((r, h, np.ones(r.shape))).T
            b = r**2 + h**2
            sol = np.linalg.lstsq(M, b, rcond=None)[0]
            rc, hc = sol[:2]/2.
            rad = np.sqrt(sol[2]+rc**2+hc**2)
            pos = np.vstack([r, h]).T
            if nonlinear:
                res = minimize(ContactAngle.rmsd_circle, x0=[rad, rc, hc], args=(pos),
                               method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
                rad, rc, hc = res.x
                base_radius = np.sqrt(rad**2-hc**2)
                rmsdval = res.fun
            else:
                rmsdval = ContactAngle.rmsd_circle([rad, rc, hc], pos)

            base_radius = np.sqrt(rad**2-hc**2)
            costheta = -hc/rad
            theta = np.arccos(costheta) * 180./np.pi
            if theta < 0:
                theta += 180.

            parms.append((rad, base_radius, theta, [rc, hc], rmsdval))
        if len(parms) == 1:
            return parms[0]
        else:
            return parms

    @staticmethod
    def _contact_angles_from_ellipse(p, off=0.0):
        """  compute the contact angle from the parameters of the polynomial representation

             :parms array_like  p: a sequence (list, tuple, ndarray) of parameters of the ellipse equation\
                                      in general form:\
                                      p[0] x^2 + p[1] x y + p[2] y^2 + p[3] x + p[4] y + p[5] = 0
             :param float off:     elevation from the substrate surface, where the\
                                      contact angle should be evaluated. Default; 0.0, corresponding\
                                      to the atomic center of the highest atom of the substrate

             :return:
             :tuple                 : a tuple (theta1,theta2) with the left and right (internal) contact angles

             We require the solution of the line passing through the point to have two coincident
             solutions for the system ax2 + bxy + cy2 + dx + ey +f = 0 and y-y0 = m (x-x0)
             This yields m = 4*a*c*x0*y0 + 2*a*e*x0 - b**2*x0*y0 - b*d*x0 - b*e*y0 - 2*b*f + 2*c*d*y0 + d*e
             The point x0,y0 is the one that solves the ellipse equation when requesting y0=off, so
             x0 =[-b*y_0 - d +/- sqrt(-4*a*c*y_0**2 - 4*a*e*y_0 - 4*a*f + b**2*y_0**2 + 2*b*d*y_0 + d**2)]/(2*a)

        """

        a, b, c, d, e, f = p
        if np.isclose(4*a*c, b**2, atol=1e-3):
            rad = np.sqrt((d**2+e**2)/(4*a**2)-f/a)
            hc = -e/(2*a) - off
            base_radius = np.sqrt(rad**2-hc**2)
            costheta = -hc/rad
            theta = np.arccos(costheta) * 180./np.pi
            if theta < 0:
                theta += 180.
            return (theta, theta)
        y_0 = off
        x_01 = (-b*y_0 - d + np.sqrt(-4*a*c*y_0**2 - 4*a*e*y_0 -
                4*a*f + b**2*y_0**2 + 2*b*d*y_0 + d**2))/(2*a)
        x_02 = (-b*y_0 - d - np.sqrt(-4*a*c*y_0**2 - 4*a*e*y_0 -
                4*a*f + b**2*y_0**2 + 2*b*d*y_0 + d**2))/(2*a)
        m1 = (4*a*c*x_01*y_0 + 2*a*e*x_01 - b**2*x_01*y_0 -
              b*d*x_01 - b*e*y_0 - 2*b*f + 2*c*d*y_0 + d*e)
        m1 = m1/(4*a*c*x_01**2 - b**2*x_01**2 - 2 *
                 b*e*x_01 + 4*c*d*x_01 + 4*c*f - e**2)
        m2 = (4*a*c*x_02*y_0 + 2*a*e*x_02 - b**2*x_02*y_0 -
              b*d*x_02 - b*e*y_0 - 2*b*f + 2*c*d*y_0 + d*e)
        m2 = m2/(4*a*c*x_02**2 - b**2*x_02**2 - 2 *
                 b*e*x_02 + 4*c*d*x_02 + 4*c*f - e**2)
        # depending on the sign of a at the denominator of x_01 and x_02, they can have a different order
        # along the axis: let's keep them ordered, so that the first is the left one and the second the right one.
        if x_02 < x_01:
            x_01, x_02 = x_02, x_01
            m1, m2 = m2, m1

        theta1 = np.arctan(m1)*180/np.pi
        theta2 = np.arctan(m2)*180/np.pi
        # we compute the internal angle (assuming the droplet is in the upper half space), and need to take care of this
        # theta1 is at the left edge of the droplet, theta2 at the right
        if theta1 < 0.0:
            theta1 += 180.
        if theta2 < 0.0:
            theta2 = -theta2
        else:
            theta2 = 180.-theta2

        return (theta1, theta2)

    @staticmethod
    def _fit_ellipse_fitzgibbon(x, y):
        D = np.vstack([x**2, x*y, y**2, x, y, np.ones(len(x))]).T
        S = D.T @ D
        C = np.zeros((6, 6), dtype=float)
        C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
        eigval, eigvec = np.linalg.eig(np.linalg.solve(S, C))
        sort = np.argsort(eigval,kind='stable')
        eigval, eigvec = eigval[sort], eigvec[sort]
        lam = np.nonzero(np.logical_and(
            eigval > 0, np.isfinite(eigval)))[0][-1]
        return eigvec[lam]

    @staticmethod
    def _fit_ellipse_flusser(x, y):
        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T

        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.solve(C, M)
        eigval, eigvec = np.linalg.eig(M)
        sort = np.argsort(eigval,kind='stable')[::-1]
        eigval, eigvec = eigval[sort], eigvec[sort]
        con = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2 > 0
        ak = eigvec[:, np.nonzero(con)[0]]
        return np.concatenate((ak, T @ ak)).ravel()

    @staticmethod
    def _fit_ellipsoid_lstsq_cox(x, y, z):
        """
            Return the polynomial coefficients that perform a least square\
            fit to a set of points, using a modified version of:\
            Turner, D. A., I. J. Anderson, J. C. Mason, and M. G. Cox.\
            "An algorithm for fitting an ellipsoid to data."\
            National Physical Laboratory, UK (1999).

            :param array_like  x: list or ndarray array with coordinate x
            :param array_like  y: list or ndarray array with coordinate y
            :param array_like  z: list or ndarray array with coordinate z

            :return:

            :ndarray: a (10,)-ndarray with the polynomial coefficients    
        """
        D = np.vstack([x**2+y**2-2*z**2, x**2 - 2*y**2 + z**2,
                      4*x*y, 2*x*z, 2*y*z, x, y, z, np.ones(len(x))]).T
        E = np.vstack([x**2+y**2+z**2]).T
        DTD, DTE = D.T@D, D.T@E
        p2 = scipy.linalg.solve(DTD, DTE).flatten()
        u, v, m, n, p, q, r, s, t = p2
        a, b = (1-u-v), (1-u+2*v)
        c = (3.-a-b)
        # a-c = a- 3 + a + b = 2a + b - 3 = 2-2u-2v +1 -u+2v -3 = -3 u
        # a-b = 1-u-v -1 +u -2 v = -3 v
        d, e, f = -2*m, -n, -p
        g, h, k = -q, -r, -s
        l = -t
        p = a, b, c, f, e, d, g, h, k, l
        if any(np.array([4*a*b-d*d, 4*a*c-e*e, 4*b*c-f*f]) < 0):
            print('non positive definite')
        return np.array(p)

    @staticmethod
    def _fit_ellipse(x, y, nonlinear=True, off=0.0, points_density=25, verbose=False):
        """  fit an ellipse through the points (x,y)

             :param list  x        : list of arrays with coordinates x
             :param list  y        : list of arrays with coordinates y
             :param bool  nonlinear : use the more accurate minimization of the rmsd instead of the algebraic distance
             :param float off       : elevation from the substrate surface, where the
                                     contact angle should be evaluated. Default; 0.0, corresponding
                                     to the atomic center of the highest atom of the substrate
             :param int   points_density: number of points per Angstrom on the ellipse that are used to compute the rmsd

             :return:

             :list      : a list with a tuple (parms,parmsc,theta, rmsd) for each of the bins
                          If only one bin is present, return just the tuple
                 parms  : array of parameters of the ellipse equation in general form:
                             ellipse:   a[0] x^2 + a[1] x y + a[2] y^2 + a[3] x + a[4] y = 0
                 parmsc : dictionary with parameters of the ellipse canoncial form: (x0,y0,a,b,phi,e)
                                  with a,b the major and minor semiaxes, x0,y0 the center and theta the angle
                                  between x axis and major axis
                 theta  : [left angle, right angle]
                 rmsd   : the rmsd to the best fit (linear or nonlinear) ellipse or ellipsoid

             Uses a slighly modified version of Fitzgibbon's least square algorithm from 
             Stable implementation from Halır, Radim, and Jan Flusser,
             "Numerically stable direct least squares fitting of ellipses."
             Proc. 6th International Conference in Central Europe on Computer
             Graphics and Visualization. WSCG. Vol. 98. Citeseer, 1998.

             python code for ellipse fitting from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
        """
        retvals = []

        for i in np.arange(len(x)):
            XX = x[i]
            YY = y[i]
            if len(XX) == 0:
                retvals.append(None)
                break
            idx1 = np.isnan(XX)
            idx2 = np.isnan(YY)
            idx = np.logical_and(~idx1, ~idx2)  # nan removed
            X, Y = XX[idx], YY[idx]
            p = ContactAngle._fit_ellipse_flusser(X, Y)
            pos = np.vstack([X, Y]).T
            # we aim at accuracy of about 0.04 Angstrom with points_density = 25
            N = int(points_density * np.max(np.sqrt(X**2+Y**2)))
            if len(p) == 0:
                # no positive eigenval, we start over from the (slightly perturbed) circle
                rad, rb, costheta, [x0, y0], rmsd = ContactAngle._fit_circle(
                    [X], [Y], nonlinear=nonlinear)
                # set a=c=1 in Eqs 27-30 https://mathworld.wolfram.com/Circle.html
                p = [1.0001, 0.0, 0.9999, -2*x0, -2*y0, x0**2+y0**2-rad**2]
            if nonlinear:
                rmsdf = ContactAngle.rmsd_ellipse_penalty
                res = minimize(rmsdf, x0=p, args=(pos, N, False),
                               method='nelder-mead', options={'xatol': 1e-4, 'disp': verbose})
                p, rmsdval = res.x, res.fun
            else:
                rmsdval = ContactAngle.rmsd_ellipse(p, pos, N)
            theta1, theta2 = ContactAngle._contact_angles_from_ellipse(
                p, off=off)
            # canonical form (careful: we overwrite internal variables a,b)
            cp = ContactAngle._ellipse_general_to_canonical(
                p, check_coeffs=False)
            retvals.append((p, cp, [theta1, theta2], rmsdval))
        if len(retvals) == 1:
            return retvals[0]
        else:
            return retvals

    @staticmethod
    def _fit_ellipsoid(x, y, z, nonlinear=True, off=0.0, points_density=25):
        """  fit an ellipsoid through the points (x,y,z)

             :param list  x        : list of arrays with coordinates x
             :param list  y        : list of arrays with coordinates y
             :param list  z        : list of arrays with coordinates z, 
             :param bool  nonlinear : use the more accurate minimization of the rmsd instead of the algebraic distance
             :param float off       : elevation from the substrate surface, where the
                                     contact angle should be evaluated. Default; 0.0, corresponding
                                     to the atomic center of the highest atom of the substrate
             :param int   points_density: number of points per Angstrom on the ellipse that are used to compute the rmsd


             :return:

             :list      : a list with a tuple (parms,parmsc,theta, rmsd) for each of the bins
                          If only one bin is present, return just the tuple
                 parms  : array of parameters of the ellipsoid equation in general form:
                             a[0] x^2 + a[1] y^2 + a[2] z^2 + a[3] yz + a[4] xz + 
                             a[5] xy  + a[6] x + a[7] y + a[8] z + a[9] = 0, otherwise.
                 parmsc : dictionary with parameters of the ellipsoid affine form (T,v),
                          such that the ellipsoid points are
                                         r  = T s + v, 
                          if s are points from the unit sphere centered in the origin.
                 theta  : the contact angle as a function of the azimuthal angle phi from phi=0, aligned 
                          with (-1,0,0) to 2 pi.
                 rmsd   : the rmsd to the best fit (linear or nonlinear) ellipsoid

             For the least square approach see _fit_ellipsoid_lstsq_cox

        """
        retvals = []
        for i in np.arange(len(x)):
            XX, YY, ZZ = x[i], y[i], z[i]
            if len(XX) == 0:
                retvals.append(None)
                break
            idx1, idx2, idx3 = np.isnan(XX), np.isnan(YY), np.isnan(ZZ)
            idx = np.logical_and(~idx1, ~idx2)
            idx = np.logical_and(idx, ~idx3)    # nan removed
            X, Y, Z = XX[idx], YY[idx], ZZ[idx]
            p = ContactAngle._fit_ellipsoid_lstsq_cox(X, Y, Z)
            pos = np.vstack([X, Y, Z]).T
            # we aim at accuracy of about 0.04 Angstrom with points_density = 25
            N = int(points_density * np.max(np.sqrt(X**2+Y**2)))
            if nonlinear:
                rmsdf = ContactAngle.rmsd_ellipsoid_penalty
                start = rmsdf(p, pos, N, False)
                # we aim at improving by 10% the least square fit
                res = minimize(rmsdf, x0=p, args=(pos, N, False),
                               method='nelder-mead', options={'xatol': start/10, 'disp': False})
                p, rmsdval = res.x, res.fun
            else:
                rmsdval = ContactAngle.rmsd_ellipsoid(p, pos, N)

            theta = np.array([])
            cp = ContactAngle._ellipsoid_general_to_affine(
                p, check_coeffs=False)
            retvals.append((p, cp, theta, rmsdval))

        if len(retvals) == 1:
            return retvals[0]
        else:
            return retvals

    @staticmethod
    def _ellipse_canonical_to_general(coeffs):
        """ Convert canonical coefficients (x0,y0,A,B,phi) to general ones (a,b,c,d,e,f)
        """
        x0, y0, A, B, phi = coeffs
        a = A*A * np.sin(phi)**2 + B*B*np.cos(phi)**2
        b = 2*(A*A-B*B)*np.sin(phi)*np.cos(phi)
        c = A*A * np.cos(phi)**2 + B*B*np.sin(phi)**2
        d = -2*a*x0 - b*y0
        e = -b*x0 - 2*c*y0
        f = a*x0**2 + b * x0*y0 + c*y0**2 - a*a*b*b
        return [a, b, c, d, e, f]

    @staticmethod
    def _ellipse_general_to_canonical(coeffs, check_coeffs=True):
        """ Convert general coefficients (a,b,c,d,e,f) to canonical ones.

            :param list coeffs       : general coefficients
            :param bool check_coeffs : raise an error if the coefficients do not represent
                                       an ellipse. In some cases this checks needs to be
                                       turned off, e.g. for a COBYLA minimization, as
                                       during the minimization the constraints might
                                       be violated.

            :return:
            : dict                   : a dictionary with the canonical coefficients x0,y0,a,b,phi,e

            from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

            Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
            ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
            The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
            ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
            respectively; e is the eccentricity; and phi is the rotation of the semi-
            major axis from the x-axis.

        """
        # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
        # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
        # Therefore, rename and scale b, d and f appropriately.
        a, b, c, d, f, g = coeffs
        b /= 2
        d /= 2
        f /= 2
        den = b**2 - a*c
        if check_coeffs and den > 0:
            raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                             ' be negative!'+str(den))
        # The location of the ellipse centre.
        x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den
        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        if np.isclose(fac, 0.0):  # a=c and b=0
            ap, bp = a, a
        else:
            argument = num / den / (fac - a - c), num / den / (-fac - a - c)
            if den > 0:  # it is not an ellipse, so not all semi axes are defined.
                # for the purpose of the minimizer, we still need to provide
                # one solution
                ap, bp = np.sqrt([np.max(argument)]*2)
            else:
                ap, bp = np.sqrt(argument)
        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp:
            width_gt_height, ap, bp = False, bp, ap
        # The eccentricity.
        r = (bp/ap)**2
        if r > 1:
            r = 1./r
        e = np.sqrt(1. - r)
        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if np.isclose(b, 0) or np.isclose(ap, bp):  # handle also the circle
            phi = 0.0 if a <= c else np.pi/2.
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2.
            if a > c:
                phi += np.pi/2.
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi/2.
        # NOTE: don't change the units of phi to deg., other parts of the code
        #       depend on it being in rad.
        phi = phi % np.pi
        return {'x0': x0, 'y0': y0, 'a': ap, 'b': bp, 'phi': phi, 'e': e}

    @staticmethod
    def _ellipsoid_base(coeffs, off, npts=1000):
        # sets z=off and expresses the polynomial coefficients
        # of an ellipse in terms of the intersection with the ellipsoid
        # then calculates the points in the Cartesian plane.
        a, b, c, f, g, h, p, q, r, d = coeffs
        coeffs_ellipse = a, 2*h, b, 2*g*off+p, 2*f*off+q, r*off+d
        pc = ContactAngle._ellipse_general_to_canonical(coeffs_ellipse)
        x, y = ContactAngle.ellipse(pc, npts=npts, tmin=0, tmax=2.*np.pi)
        return x, y

    @staticmethod
    def _ellipsoid_normal(coeffs, x, y, z):
        # first we need to recover the values of (theta,phi) corresponding to the
        # points x,y,z belonging to the ellipsoid defined by the polynomial with
        # coefficients coeffs
        T, v = ContactAngle._ellipsoid_general_to_affine(coeffs).values()
        pos = np.vstack([x, y, z]).T
        # points on the unit sphere
        s = np.dot(np.linalg.inv(T), (pos - v).T).T
        s = s.T / np.linalg.norm(s, axis=1)  # tidy up roundoff errors
        sx, sy, sz = s[0], s[1], s[2]
        theta = np.arccos(sz)  # radius is one
        phi = np.sign(sy) * sx / np.sqrt(1-sz**2)
        dsdtheta = np.vstack(
            [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]).T
        dsdphi = np.vstack([-np.sin(theta)*np.sin(phi),
                           np.sin(theta)*np.cos(phi), 0*phi]).T
        normal = np.cross(dsdtheta, dsdphi)
        # we need now to check if it's pointing outwards or inwards.
        # It's enough to inspect one point: since v is the center of
        # the affine transformation, x-v is the displacement from it,
        # and we request that (x-v).n > 0 for n to be pointing out.
        if (pos[0]-v).T@normal[0] < 0:
            normal = -normal
        # tidy up roundoff errors
        return (normal.T/np.linalg.norm(normal, axis=1)).T

    @staticmethod
    def _ellipsoid_contact_line_and_angles(coeffs, off, npts=1000):
        """ Computes the contact line and the contact angles along the
            line. 

            :param array_like coeffs : the coefficients of the polynomial
                                       defining the ellipsoid, see, e.g.,
                                       _ellipsoid_general_to_affine
            :param float      off    : elevation above the topmost atoms
                                       of the substrate, used to compute
                                       the elliptical intersection that is
                                       the contact line
            :param int        npts   : number of points to sample

            :return:
            :tuple                   : a tuple containing the contact line
                                       on the (x,y) plane as a (npts,2)-ndarray 
                                       and the contact angles as a (npts,)-ndarray.
        """
        bx, by = ContactAngle._ellipsoid_base(coeffs, off, npts=npts)
        bz = np.zeros(bx.shape)+off
        n = ContactAngle._ellipsoid_normal(coeffs, bx, by, bz)
        theta = np.arccos(n[:, 2])
        return (np.vstack([bx, by]).T, theta)

    @staticmethod
    def _ellipsoid_general_to_affine(coeffs, check_coeffs=True):
        """ Convert general coefficients (a,b,c,f,g,h,p,q,r,d) of the polynomial
            a xx + b yy +c zz + 2f yz + 2g xz + 2h xy + px + qy +rz +d to the affine 
            transformation ones that map the unitary sphere centered in the origin 
            to a point on the ellipsoid.

            :param list coeffs       : general coefficients
            :param bool check_coeffs : raise an error if the coefficients do not represent
                                       an ellipse. In some cases this checks needs to be
                                       turned off, e.g. for a COBYLA minimization, as
                                       during the minimization the constraints might
                                       be violated.

            :return:
            : dict                   : a dictionary with the affine matrix and vector, T and v
                                       If a point on the unit sphere centered in the origin is s, 
                                       then the correspoinding point r=(x,y,z) on the ellipsoid is 
                                       defined by the affine transofmation r = T s + v = A^(-1/2) s + v

            Here, A is the matrix of the quadratic form defining the ellipsoid, 

                               r^t A r + beta^t r + gamma = 0. 

            The connection to the general coefficients is given by

                          |  a   h    g  |
                       A= |  h   b    f  | ,   beta^t = (p, q, r) , gamma = d 
                          |  g   f    c  |

            In addition, the quadratic form can be recast to (r-v)^t A' (r-v) = k.
            This form is not unique, but this is helpful to find the correspondence
            to the polynmomial coefficients, which are also defined up to an immaterial
            multiplicative constant. Note that setting k=1 imposes a constaint on gamma,
            but gamma=d is given, so we need to keep the general form and determine k.
            After expansion and comparison, one finds

                A' = A ,   v = -1/2 A^-1 beta ,   k = (1/4) beta^t A^-1 beta - gamma,

            Now that k is known, we can recast the equation in the standard form
            (r-v)^t M (r-v) = 1 by setting M = A'/k = A/k. Note that v has to be determined
            using A and not A/k. The standard form can be used to generate points r on the
            ellipse starting from the unit sphere s centered in zero as the affine transformation

                                   r = v + T s, with T = A^{-1/2}  

            This function eventually returns the elements of the affine transofmation T and v.

            >>> # This tests that we can recover the original parameters from a fit
            >>> p1 = np.array([2, 3, 4., 0.4, 0.5, -0.6, 0.3, 4.2, 0.1, -10])
            >>> cp = ContactAngle._ellipsoid_general_to_affine(p1,check_coeffs=True)
            >>> x,y,z = ContactAngle.ellipsoid(cp)
            >>> p2 = ContactAngle._fit_ellipsoid_lstsq_cox(x,y,z) 
            >>> # NOTE: for Cox, a+b+c = 3, so, to compare the two sets we need to normalize
            >>> p2 *= np.sum(p1[:3])/3.
            >>> print(np.all(np.isclose(p1,p2)))
            True
        """

        a, b, c, f, g, h, p, q, r, d = coeffs
        if check_coeffs:
            conds = [a*b - h ** 2 > 0,  a*c - g**2 > 0, b*c - f**2 > 0]
            if not all(conds):
                raise ValueError('coeffs do not represent an ellipsoid!')
        A = np.array([[a, h, g], [h, b, f], [g, f, c]])
        beta = np.array([p, q, r])
        norm = -d + beta.T@np.linalg.inv(A)@beta/4
        A /= norm
        U, Sigma, VT = np.linalg.svd(A)
        invsqrt_A = np.linalg.inv(U@np.diag(np.sqrt(Sigma))@VT)
        v = -0.5 * np.linalg.inv(A*norm)@beta
        return {'T': invsqrt_A.tolist(), 'v': v.tolist()}

    def _compute_coords(self, r, h, symm):
        # r is the distance of the point projected on the
        # substrate from the axis/center of the
        # cylindrical/sperical droplet
        #
        # h is the elevation from the substrate surface
        #
        # symm is the remaining coordinate (the angle phi in
        # spherical coordinates, the position along the cylinder
        # axis in cylindrical ones.

        R = np.sqrt(r**2 + h**2)
        theta = np.arccos(r/R)
        return np.asarray(r), np.asarray(theta), np.asarray(h), np.asarray(symm)

    def cartesian_coordinates(self, group):
        pos = self._shifted_pos(group)
        if self.periodic is None:
            return pos[:, 0], pos[:, 1], pos[:, 2]
        else:
            dirs = np.array([0, 1])
            dd = dirs[dirs != self.periodic][0]
            od = list(set((1, 2, 3))-set((2, dd)))[0]
            return pos[:, dd], pos[:, od], pos[:, 2]

    def cylindrical_coordinates(self, group):
        pos = self._shifted_pos(group)
        dirs = np.array([0, 1])
        dd = dirs[dirs != self.periodic][0]
        r = pos[:, dd]
        h = pos[:, 2]
        od = list(set((1, 2, 3))-set((2, dd)))[0]
        z = pos[:, od]
        return self._compute_coords(r, h, z)

    def spherical_coordinates(self, group):
        pos = self._shifted_pos(group)
        r = np.linalg.norm(pos[:, 0:2], axis=1)
        h = pos[:, 2]
        symm = np.arctan2(pos[:, 1], pos[:, 0])
        phi = np.arctan2(h, r)*180/np.pi
        phi[phi < 0] += 360.
        return self._compute_coords(r, h, phi)

    def sample_theta_R(self, theta, r, h):
        """
            given a set of angles (theta), radii (r) and elevations (h), compute the mean profile h(r)
            by taking the binned average values of h and r. 
        """
        R = np.sqrt(r*r+h*h)
        n_h, t_edges = np.histogram(
            theta, bins=self.bins, density=False, range=(0, np.pi), weights=None)
        R_h, t_edges = np.histogram(
            theta, bins=self.bins, density=False, range=(0, np.pi), weights=R)
        cond = n_h > 0.0
        R_h = R_h[cond]
        t_edges = t_edges[:-1][cond]
        n_h = n_h[cond]
        hh, rr = R_h * np.sin(t_edges)/n_h, R_h * np.cos(t_edges)/n_h
        hhmin = np.min(hh)
        ercut = hhmin+self.hcut
        cond = hh > ercut
        self.histo.x, self.histo.h = rr[cond], hh[cond]

    def _select_coords(self, use, bins):
        # return lists of array with coordinates falling into the bins that partition the symmetric coordinate.
        # The main idea is to use the instantaneous or stored cartesian coordinates (renamed so that z is always
        # the surface normal and, in case of cylyndrical symmetry, y points along the cylinder axis.
        # The cae of the histogram is different, because the relation, say (x,z) is not necessarily a function.
        # In this case, we store the spherical/cylindrical coordinate values in the histogram, and we recover the
        # averaged location of the surface a posteriori.
        if self._sampled == False:
            self._sampled, _ = True, self.sample()
        try:
            if self.store and use == 'stored':
                x, h = self._x, self._h
                if self.periodic is None:
                    y = self._y
            elif not self.store and use == 'stored':
                raise (ValueError(
                    "The choice use='stored' can only be used if store=True was passed at initialization "))
            elif use == 'frame':
                x, h = self.x, self.h
                if self.periodic is None:
                    y = self.y
            elif use == 'histogram':
                x, h = self.histo.x, self.histo.h
                if self.periodic is None:
                    y = self.histo.y
            else:
                raise (ValueError(
                    "The parameter use can only take the values 'frame', 'histogram', or 'stored' "))

        except AttributeError:
            raise RuntimeError('No surface atoms or Gibbs surface present')

        # we always comply with the request of cutting all molecules below hcut from the minimum
        if self.hcut > 0:
            hmin = np.min(h)
            cond = h > hmin+self.hcut
            x, h = x[cond], h[cond]
            if self.periodic is None:
                y = y[cond]
        if bins > 1:
            # TODO: extend for histogram and stored
            if use != 'frame':
                raise (ValueError("bins>1 can be used only with use='frame'"))
            hx, hy, hh = [], [], []
            symm = self.phi[cond] if self.periodic is None else self.y[cond]
            limit = 2 * \
                np.pi if self.periodic is None else self.substrate.dimensions[self.periodic]
            # set the right edges of the bins. This assumes that shifted_pos() has been called on the
            # coordinates (hence, this function should not be public)
            binvals = np.linspace(-limit/2., limit/2., bins+1)[1:-1]
            # this will put in bin index 0 eveything close to or below 0.0 and in bin index
            # nbins-1 everyhing close to or above limit
            inds = np.digitize(symm, binvals)
            for i in range(bins):
                hx.append(x[inds == i])
                hh.append(h[inds == i])
                if self.periodic is None:
                    hy.append(y[inds == i])
        else:
            hx, hh = [x], [h]
            if self.periodic is None:
                hy = [y]
        if self.periodic is None:
            return hx, hy, hh
        else:
            return hx, hh

    def fit_circle(self, use='frame', nonlinear=True, bins=1):
        """ fit an arc through the profile h(r) sampled by the class

            :param str   use: 'frame'    : use the positions of the current frame only (default)\
                              'histogram': use the binned values sampled so far\
                              'stored'   : use the stored surface atoms positions, if the option store=True was passed at initialization
            :param int   bins: the number of bins to use along the symmetry direction (cylinder axis, azimuthal angle)

            :return:
                a list including, for each of the bins:

                :tuple:  radius, base radius, cos(theta), center
        """
        r, h = self._select_coords(use, bins=bins)

        return self._fit_circle(r, h, nonlinear=nonlinear)

    def fit_ellipse(self, use='frame', nonlinear=True, bins=1):
        """ fit an ellipse through the points sampled by the class. See implementation details in _fit_ellipsoid()

            :param str   use: 'frame'    : use the positions of the current frame only (default)\
                              'histogram': use the binned values sampled so far\
                              'stored'   : use the stored surface atoms positions, if the option store=True\
                                           was passed at initialization
            :param int   bins: the number of bins to use along the symmetry direction (cylinder axis, azimuthal angle)

            :return:
                a list including, for each of the bins, a tuple with elements:

                :list: parms:  parameters of the ellipse polynomial in general form:\
                              a[0] x^2 + a[1] x y + a[2] y^2 + a[3] x + a[4] y = 0
                :dict: parmsc: dictionary of parameters in canoncial form: (a,b, x0,y0,phi, e)\
                              with a,b the major and minor semiaxes, x0,y0 the center, phi  the angle\
                              (in rad) between x axis and major axis, and e the eccentricity.
                :list: theta:  [left contact angle, right contact angle]
        """

        r, h = self._select_coords(use, bins=bins)
        p, cp, theta, rmsd = self._fit_ellipse(
            r, h, nonlinear=nonlinear, off=0.0)
        self._polynomial_coefficients, self._canonical_form = p, cp
        return p, cp, theta, rmsd

    def fit_ellipsoid(self, use='frame', nonlinear=True, bins=1):
        """  fit an ellipsoid through the points sampled by the class. See implementation details in _fit_ellipsoid()

            :param str   use        : 'frame'    : use the positions of the current frame only (default)\
                                      'histogram': use the binned values sampled so far\
                                      'stored'   : use the stored surface atoms positions, if the option store=True\
                                                   was passed at initialization
            :param int   bins       : the number of bins to use along the symmetry direction (cylinder axis, azimuthal angle)

            :return:

                 a list with a tuple (parms,parmsc,theta, rmsd) for each of the bins;\
                          If only one bin is present, return just the tuple.

                 :array: parms  : array of parameters of the ellipsoid equation in general form:\
                             a[0] x^2 + a[1] y^2 + a[2] z^2 + a[3] yz + a[4] xz + \
                             a[5] xy  + a[6] x + a[7] y + a[8] z + a[9] = 0, otherwise.
                 :dict: parmsc : dictionary with parameters of the ellipsoid affine form (T,v),\
                          such that the ellipsoid points are\
                                         r  = T s + v,\
                          if s are points from the unit sphere centered in the origin.
                 :float: theta  : the contact angle as a function of the azimuthal angle phi from phi=0, aligned\
                          with (-1,0,0) to 2 pi.
                 :float: rmsd   : the rmsd to the best fit (linear or nonlinear) ellipsoid
        """

        x, y, z = self._select_coords(use, bins=bins)  # TODO FIXME

        p, cp, theta, rmsd = self._fit_ellipsoid(
            x, y, z, nonlinear=nonlinear, off=0.0)
        self._polynomial_coefficients, self._canonical_form = p, cp
        return p, cp, theta, rmsd
