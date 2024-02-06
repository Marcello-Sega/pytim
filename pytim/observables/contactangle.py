# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: ContactAngle
    ====================
"""
import pytim
import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree
import warnings
from numpy.linalg import solve
from scipy.optimize import curve_fit
from scipy.optimize import minimize

class ContactAngle(object):
    """ ContactAngle class implementation that uses interfacial atoms
        Computes radial profiles and contact angles using different approaches

        :param AtomGroup droplet   : Atom group representative of the droplet
        :param PYTIM interface     : Compute the contact angle for this interface
        :param AtomGroup droplet   : Atom group representative of the droplet
        :param AtomGroup substrate : Atom group representative of the substrate
        :param str       binning   : 'theta' for angular binning, 'h' for elevation binning
        :param float     hcut      : don't consider contributions from atoms closer than this to the substrate
                                     (used to disregard the microscopic contact angle)
        :param float     hcut_upper: don't consider contributions from atoms above this distance from the substrate
                                     default: None
        :param int       bins      : bins used for sampling the profile
        :param int       periodic  : direction along which the system is periodic. Default: None, not periodic
        :param int       removeCOM : remove the COM motion along this direction. Default: None, does not remove COM motion

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
        >>> CA = observables.ContactAngle(inter, substrate, binning='theta', periodic=1,bins=397,removeCOM=0,hcut=5)
        >>> for ts in u.trajectory[::]:
        ...     CA.sample()
        >>> # Instantaneous contact angle (last frame) by fitting a circle...
        >>> np.round(CA.contact_angle,2)
        90.12

        >>> 
        >>> # ... and using an elliptical fit:
        >>> left, right = CA.contact_angles
        >>> # left angle
        >>> np.round(np.abs(left),2)
        90.25

        >>> # right angle
        >>> np.round(right,2)
        92.55

        >>> # Contact angles from the averaged binned statistics of
        >>> # surface atoms' radial distance as a function of the azimuthal angle
        >>> list(np.round(CA.mean_contact_angles,2))
        [83.78, 87.01]

   """

    class Histogram(object):
        def __init__(self):
            self.r,self.h = np.array([]), np.array([])

    histo = Histogram()

    def __init__(self, inter, substrate, binning='theta',hcut=0.0,hcut_upper=None,
                 contact_cut=0.0,bins=100,periodic=None,removeCOM=None,store=False):
        self.inter, self.substrate = inter, substrate
        self.universe, self.trajectory = self.inter.universe, self.inter.universe.trajectory
        self.hcut, self.hcut_upper = hcut, hcut_upper
        self.nframes, self.binning, self.periodic = 0, binning, periodic
        self.bins, self.removeCOM = bins, removeCOM
        self.contact_cut, self.store = contact_cut, store
        self.r, self.h = np.array([]),np.array([])
        if self.contact_cut == 'auto':
            tree = cKDTree(substrate.positions,boxsize = substrate.universe.dimensions[:3])
            d,_ =tree.query(inter.atoms.positions, k=1)
            hist,bins =np.histogram(d,int(np.max(d)/0.5))
            # check the location of the maximum with resolution 0.5 Angstrom
            vmax,imax = np.max(hist),np.argmax(hist)
            # find where the distribution drops below 5% of the maximum; this could happen before and after the maximum
            idx = np.where(hist<=0.05*vmax)[0]
            # select the first bin that satisfies the above condition and is after the maximum
            self.contact_cut = bins[idx[idx>imax][0]]

    @property
    def contact_angle(self):
        """ 
            The contact angle from the best-fititng circumference 
            computed using the current frame
        """
        _ , _, theta, _ ,_ = self.fit_arc()
        return theta
    @property

    def mean_contact_angle(self):
        """ 
            The contact angle from the best-fititng circumference 
            computed using the location of the interface averaged 
            over the sampled frames
        """
        _ , _, theta, _ ,_ = self.fit_arc(use='histogram')
        return theta

    @property
    def contact_angles(self):
        """ 
            The left and rigth contact angles from the best-fititng ellipse
            computed using the current frame
        """
        _, _, left, right, _  = self.fit_arcellipse()
        return left, right

    @property
    def mean_contact_angles(self):
        """ 
            The left and rigth contact angles from the best-fititng ellipse
            computed using the location of the interface averaged 
            over the sampled frames
        """
        _, _, left, right, _  = self.fit_arcellipse(use='histogram')
        return left, right

    def shifted_pos(self, group):
        com = group.center_of_mass()
        hoff = np.max(self.substrate.atoms.positions[:, 2])
        com[2] = hoff
        # removes the com along x,y and uses the topmost substrate atom to
        # define the location of the origin along the vertical axis
        pos = group.positions - com
        # further, we won't be using any coordinate that is below a certain
        # distance from the substrate surface
        pos = pos[pos[:, 2] > self.hcut]
        if self.hcut_upper is not None: pos = pos[pos[:, 2] < self.hcut_upper]
        return pos

    def _remove_COM(self, group, inter, alpha, box):
        def group_size(g,direction):
            p = g.atoms.positions[:, direction]
            return np.abs(np.max(p) - np.min(p))

        if self.removeCOM is not None:
            while(group_size(group,self.removeCOM) > box[self.removeCOM] - 4 * alpha):
                pos = group.universe.atoms.positions.copy()
                pos[:, self.removeCOM] += alpha
                group.universe.atoms.positions = pos.copy()
                pos = group.universe.atoms.pack_into_box()
                group.universe.atoms.positions = pos.copy()
            com = self.inter.atoms.center_of_mass()[self.removeCOM]
            pos = group.universe.atoms.positions.copy()
            pos[:, self.removeCOM] -= com
            group.universe.atoms.positions = pos.copy()

            pos[:, self.removeCOM] += box[self.removeCOM]/2
            group.universe.atoms.positions = pos.copy()

            group.universe.atoms.positions = pos.copy()
            pos = group.universe.atoms.pack_into_box()
            group.universe.atoms.positions = pos.copy()
        return group.universe.atoms.positions


    def sample(self):
        """ samples the height profile h(r) of a droplet, stores
            the current atomic coordinates of the liquid surface
            not in contact with the substrate in the reference
            frame of the droplet, and optionally the coordinates
            along the whole trajectory.
            TODOs:
            1) test with the molecular option of GITIM
            2) bins in x,y and h directions are the same, this might be a problem for boxes with large aspect
               ratios
            3) removeCOM=0 should remove the movement of droplet along the x axis
        """
        if self.contact_cut > 0:
            tree_substrate = cKDTree(self.substrate.atoms.positions,boxsize=self.substrate.universe.dimensions[:3])
            dists , _ = tree_substrate.query(self.inter.atoms.positions,k=1)
            self.liquid_vapor = self.inter.atoms[dists > self.contact_cut]
        else:
            self.liquid_vapor = self.inter.atoms

        box = self.substrate.universe.dimensions[:3]
        if self.store:
            try: _ = self._r
            except: self._r, self._h, self._theta, self._R = [], [], [], []

        self.nframes += 1

        self._remove_COM(self.inter.atoms, self.liquid_vapor, self.inter.alpha, box)
        self.maxr = np.max(self.substrate.universe.dimensions[:2])
        self.maxh = self.substrate.universe.dimensions[2]

        if self.periodic is None:
            r, theta, h , phi  = self.spherical_coordinates()
            self.phi= phi.copy()
        else:
            r, theta, h , z = self.cylindrical_coordinates()
            self.z = z.copy()

        self.r = r.copy()
        self.theta = theta.copy()
        self.h = h.copy()

        if self.binning == 'theta':
            self.sample_theta_R(theta, r, h)
        elif self.binning == 'h':
            self.sample_r_h(r,h)
        else:
            raise (ValueError("Wrong binning type"))

    @staticmethod
    def rmsd_ellipse(p,x,N,check_coeffs=True):
        cp = ContactAngle._ellipse_general_to_canonical(p,check_coeffs)
        cp = {'x0':cp[0],'y0':cp[1],'a':cp[2],'b':cp[3],'phi':cp[4]}
        xx,yy = ContactAngle.arc_ellipse(cp,N)
        pos = np.vstack([xx,yy]).T
        cond = np.logical_and(yy>=np.min(x[:,1]),yy<=np.max(x[:,1]))
        pos = pos[cond]
        return np.sqrt(np.mean(cKDTree(pos).query(x)[0]**2))

    @staticmethod
    def rmsd_ellipse_penalty(p,x,N,check_coeffs=True):
        rmsd = ContactAngle.rmsd_ellipse(p,x,N,check_coeffs)
        penalty = 0 if 4*p[0]*p[2]-p[1]**2 > 0  else 10*rmsd
        return rmsd + penalty

    @staticmethod
    def rmsd_circle(p,x):
        R,x0,y0  = p
        d = np.linalg.norm(np.array([x0,y0])-x,axis=1) - R
        return np.sqrt(np.mean(d**2))

    @staticmethod
    def arc_ellipse(parmsc, npts=100, tmin=0.,tmax=2.*np.pi):
        """ Return npts points on the ellipse described by the canonical parameters
            x0, y0, ap, bp, e, phi for values of the paramter between tmin and tmax.

            :param dict  parmsc : dictionary with keys: x0,y0,a,b,phi
            :param float tmin   : minimum value of the parameter
            :param float tmax   : maximum value of the parameter
            :param int   npts   : number of points to use

            return:
            :tuple              : (x,y) of coordinates as numpy arrays
        """
        t = np.linspace(tmin, tmax, npts)
        x = parmsc['x0'] + parmsc['a'] * np.cos(t) * np.cos(parmsc['phi']) - parmsc['b'] * np.sin(t) * np.sin(parmsc['phi'])
        y = parmsc['y0'] + parmsc['a'] * np.cos(t) * np.sin(parmsc['phi']) + parmsc['b'] * np.sin(t) * np.cos(parmsc['phi'])
        return x, y

    @staticmethod
    def arc_circle(parmsc, npts=100, tmin=0.,tmax=2.*np.pi):
        """ Return npts points on the circle described by the canonical parameters
            R, x0, y0 for values of the paramter between tmin and tmax.

            :param dict  parmsc : dictionary with keys: R, x0, y0
            :param float tmin   : minimum value of the parameter
            :param float tmax   : maximum value of the parameter
            :param int   npts   : number of points to use

            return:
            :tuple              : (x,y) of coordinates as numpy arrays
        """
        t = np.linspace(tmin, tmax, npts)
        x = parmsc['x0'] + parmsc['R'] * np.cos(t)
        y = parmsc['y0'] + parmsc['R'] * np.sin(t)
        return x, y

    @staticmethod
    def _fit_arc(hr,hh,nonlinear=True):
        """ fit an arc through the profile h(r) sampled by the class
            :param list hr        : list of arrays with the radial coordinates
            :param list hh        : list of arrays with the elevations
            :param bool nonlinear : use the more accurate minimization of the rmsd instead of the algebraic distance

            return:
            :list             : a list with the tuple (radius, base radius, cos(theta), center, rmsd)
                                for each bin. If only one bin is present, return just the tuple. 
        """
        parms = []
        for i in np.arange(len(hr)):
            r = hr[i]
            h = hh[i]
            if len(r) == 0: 
                parms.append(None)
                break
            M = np.vstack((r,h,np.ones(r.shape))).T
            b = r**2 + h**2
            sol = np.linalg.lstsq(M,b,rcond=None)[0]
            rc,hc = sol[:2]/2.
            rad = np.sqrt(sol[2]+rc**2+hc**2)
            pos = np.vstack([r,h]).T
            if nonlinear:
                res = minimize(ContactAngle.rmsd_circle, x0=[rad,rc,hc], args=(pos), 
                    method='nelder-mead',options={'xatol': 1e-8, 'disp': False})
                rad, rc, hc = res.x
                base_radius = np.sqrt(rad**2-hc**2)
                rmsdval = res.fun
            else:
                rmsdval = ContactAngle.rmsd_circle([rad, rc,hc], pos)

            base_radius = np.sqrt(rad**2-hc**2)
            costheta = -hc/rad
            theta = np.arccos(costheta) * 180./np.pi
            if theta<0: theta+=180.

            parms.append((rad, base_radius, theta, [rc,hc], rmsdval))
        if len(parms) == 1 : return parms[0]
        else : return parms


    @staticmethod
    def _fit_ellipse(hr, hh, nonlinear=True, off=0.0, points_density=25):
        """  fit an ellipse through the values h(r)

             :param list  hr        : list of arrays with distances from the center along the contact surface
             :param list  hh        : list of arrays with elevations from the contact surface
             :param bool  nonlinear : use the more accurate minimization of the rmsd instead of the algebraic distance
             :param float off       : elevation from the substrate surface, where the
                                     contact angle should be evaluated. Default; 0.0, corresponding
                                     to the atomic center of the highest atom of the substrate
             :param int   points_densioty: number of points per Angstrom on the ellipse that are used to compute the rmsd

             return:
             :list      : a list with a tuple (parms,parmsc,theta1,theta2) for each of the bins
                          If only one bin is present, return just the tuple
                 parms  : array of parameters of the ellipse equation in general form:
                          a[0] x^2 + a[1] x y + a[2] y^2 + a[3] x + a[4] y = 0
                 parmsc : dictionary with parameters of the ellipse in canoncial form: (x0,y0,a,b,phi,e)
                          with a,b the major and minor semiaxes, x0,y0 the center and theta the angle
                          between x axis and major axis
                 theta1 : right angle
                 theta2 : left angle
                 rmsd   : the rmsd to the best fit (linear or nonlinear) ellipse

             Stable implementation from Halır, Radim, and Jan Flusser,
             "Numerically stable direct least squares fitting of ellipses."
             Proc. 6th International Conference in Central Europe on Computer
             Graphics and Visualization. WSCG. Vol. 98. Citeseer, 1998.

             python code for ellipse fitting from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
        """
        parms = []

        def constr(x):
            # can be used if switching to COBYLA
            return 4*x[0]*x[2]-x[1]**2

        for i in np.arange(len(hr)):
            RR = hr[i]
            HH = hh[i]
            if len(RR) == 0:
                parms.append(None)
                break
            idx1 = np.isnan(RR)
            idx2 = np.isnan(HH)
            idx = np.logical_and(~idx1, ~idx2)  # nan removed
            x, y = RR[idx], HH[idx]

            D1 = np.vstack([x**2, x*y, y**2]).T
            D2 = np.vstack([x, y, np.ones(len(x))]).T
            S1 = D1.T @ D1
            S2 = D1.T @ D2
            S3 = D2.T @ D2
            T = -np.linalg.inv(S3) @ S2.T
            M = S1 + S2 @ T
            C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
            M = np.linalg.solve(C,M)
            eigval, eigvec = np.linalg.eig(M)
            con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
            ak = eigvec[:, np.nonzero(con > 0)[0]]
            p = np.concatenate((ak, T @ ak)).ravel()
            pos = np.vstack([x,y]).T
            # we aim at accuracy of about 0.04 Angstrom with points_density = 25
            N = int(points_density * np.max(np.sqrt(x**2+y**2)))
            if nonlinear:
                res = minimize(ContactAngle.rmsd_ellipse_penalty,x0=p, args=(pos,N,False),
                      method='nelder-mead',options={'xatol': 1e-2, 'disp': False})
                #  res = minimize(ContactAngle.rmsd_ellipse,x0=p, args=(pos,N,False),
                #     method='COBYLA',options={'rhobeg': 1e-1, 'disp': True,'tol':1e-9},
                #                  constraints=[{'type':'ineq','fun':constr,'catol':0.0}])
                p,rmsdval = res.x,res.fun
            else:
                rmsdval = ContactAngle.rmsd_ellipse(p,pos,N)

            # contact angles from derivative of the curve when it crosses the substrate:
            # We require the solution of the line passing through the point to have two coincident
            # solutions for the system ax2 + bxy + cy2 + dx + ey +f = 0 and y-y0 = m (x-x0)
            # This yields m = 4*a*c*x0*y0 + 2*a*e*x0 - b**2*x0*y0 - b*d*x0 - b*e*y0 - 2*b*f + 2*c*d*y0 + d*e
            # The point x0,y0 is the one that solves the ellipse equation when requesting y0=off, so
            # x0 =[-b*y_0 - d +/- sqrt(-4*a*c*y_0**2 - 4*a*e*y_0 - 4*a*f + b**2*y_0**2 + 2*b*d*y_0 + d**2)]/(2*a)
            a,b,c,d,e,f = p
            y_0 = off
            x_01 = (-b*y_0 - d + np.sqrt(-4*a*c*y_0**2 - 4*a*e*y_0 - 4*a*f + b**2*y_0**2 + 2*b*d*y_0 + d**2))/(2*a)
            x_02 = (-b*y_0 - d - np.sqrt(-4*a*c*y_0**2 - 4*a*e*y_0 - 4*a*f + b**2*y_0**2 + 2*b*d*y_0 + d**2))/(2*a)
            m1 = (4*a*c*x_01*y_0 + 2*a*e*x_01 - b**2*x_01*y_0 - b*d*x_01 - b*e*y_0 - 2*b*f + 2*c*d*y_0 + d*e )
            m1 = m1/(4*a*c*x_01**2 - b**2*x_01**2 - 2*b*e*x_01 + 4*c*d*x_01 + 4*c*f - e**2)
            m2 = (4*a*c*x_02*y_0 + 2*a*e*x_02 - b**2*x_02*y_0 - b*d*x_02 - b*e*y_0 - 2*b*f + 2*c*d*y_0 + d*e )
            m2 = m2/(4*a*c*x_02**2 - b**2*x_02**2 - 2*b*e*x_02 + 4*c*d*x_02 + 4*c*f - e**2)
            # depending on the sign of a at the denominator of x_01 and x_02, they can have a different order
            # along the axis: let's keep them ordered, so that the first is the left one and the second the right one.
            if x_02<x_01 :
                x_01,x_02 = x_02, x_01
                m1,m2 = m2,m1

            theta1 = np.arctan(m1)*180/np.pi
            theta2 = np.arctan(m2)*180/np.pi
            # we compute the internal angle (assuming the droplet is in the upper half space), and need to take care of this
            # theta1 is at the left edge of the droplet, theta2 at the right
            if theta1<0.0: theta1+=180.
            if theta2<0.0: theta2=-theta2
            else: theta2=180.-theta2

            # canonical form (careful: we overwrite internal variables a,b)
            x0,y0,a,b,phi,e = ContactAngle._ellipse_general_to_canonical(p,check_coeffs=False)
            parms.append((p, {'x0':x0,'y0':y0,'a':a,'b':b,'phi':phi,'e':e}, theta1, theta2, rmsdval))
        if len(parms)==1: return parms[0]
        else : return parms

    @staticmethod
    def _ellipse_canonical_to_general(coeffs):
        """ Convert canonical coefficients (x0,y0,A,B,phi) to general ones (a,b,c,d,e,f)
        """
        x0,y0,A,B,phi = coeffs
        a = A*A *np.sin(phi)**2 + B*B*np.cos(phi)**2
        b = 2*(A*A-B*B)*np.sin(phi)*np.cos(phi)
        c =  A*A *np.cos(phi)**2 + B*B*np.sin(phi)**2
        d = -2*a*x0 -b*y0
        e = -b*x0 -2*c*y0
        f = a*x0**2 + b * x0*y0 + c*y0**2 - a*a*b*b
        return [a,b,c,d,e,f]

    @staticmethod
    def _ellipse_general_to_canonical(coeffs,check_coeffs=True):
        """ Convert general coefficients (a,b,c,d,e,f) to canonical ones.

            :param list coeffs       : general coefficients
            :param bool check_coeffs : raise an error if the coefficients do not represent
                                       an ellipse. In some cases this checks needs to be
                                       turned off, e.g. for a COBYLA minimization, as
                                       during the minimization the constraints might
                                       be violated.

            return:
            : tuple     : canonical coefficients x0,y0,a,b,phi,e

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
        a,b,c,d,f,g = coeffs
        b/=2; d/=2 ; f/=2
        den = b**2 - a*c
        if check_coeffs and den > 0:
            raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                             ' be negative!'+str(den))
        # The location of the ellipse centre.
        x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den
        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        if np.isclose(fac,0.0): # a=c and b=0
            ap,bp=a
        else:
            argument = num / den / (fac - a - c), num / den / (-fac - a - c)
            if den>0: # it is not an ellipse, so not all semi axes are defined.
                      # for the purpose of the minimizer, we still need to provide
                      # one solution
                ap,bp = np.sqrt([np.max(argument)]*2)
            else:
                ap,bp = np.sqrt(argument)
        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp: width_gt_height , ap,bp = False, bp, ap
        # The eccentricity.
        r = (bp/ap)**2
        if r > 1: r = 1./r
        e = np.sqrt(1. - r)
        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if np.isclose(b,0):
            phi = 0.0 if a < c else np.pi/2.
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2.
            if a > c: phi += np.pi/2.
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi/2.
        # NOTE: don't change the units of phi to deg., other parts of the code
        #       depend on it being in rad.
        phi = phi % np.pi
        return x0, y0, ap, bp, phi, e

    def _compute_coords(self, r, h, symm):
        # r is the distance from the center of the droplet
        # projected on the substrate surface.
        # h is the elevation from the substrate surface
        # symm is the remaining coordinate (the angle phi in
        # spherical coordinates, the position along the cylinder
        # axis in cylindrical ones.
        R = np.sqrt(r**2 + h**2)
        theta = np.arccos(r/R)
        if self.store:
            self._theta += list(theta)
            self._r += list(r)
            self._h += list(h)

        return  np.asarray(r), np.asarray(theta), np.asarray(h), np.asarray(symm)

    def cylindrical_coordinates(self):
        pos = self.shifted_pos(self.liquid_vapor.atoms)
        dirs = np.array([0, 1])
        dd = dirs[dirs != self.periodic][0]
        r = pos[:, dd]
        h = pos[:, 2]
        od = list(set((1,2,3))-set((2,dd)))[0]
        z = pos[:,od]
        return self._compute_coords(r, h, z)

    def spherical_coordinates(self):
        pos = self.shifted_pos(self.liquid_vapor.atoms)
        r = np.linalg.norm(pos[:, 0:2], axis=1)
        h = pos[:, 2]
        symm = np.arctan2(pos[:,1],pos[:,0])
        phi  = np.arctan2(y,x)*180/np.pi
        phi[phi<0]+=360.
        return self._compute_coords(r, h, phi)

    def sample_theta_R(self, theta, r, h):
        """
            given a set of angles (theta), base radii (r) and elevations (s), compute the mean profile h(r)
            by taking the binned average values of h and r.
        """
        R = np.sqrt(r*r+h*h)
        n_h, t_edges = np.histogram(theta, bins=self.bins, density=False, range=(0, np.pi), weights=None)
        R_h, t_edges = np.histogram(theta, bins=self.bins, density=False, range=(0, np.pi), weights=R)
        cond = n_h > 0.0
        R_h = R_h [cond]
        t_edges = t_edges[:-1][cond]
        n_h = n_h [cond]
        hh, rr = R_h * np.sin(t_edges)/n_h, R_h * np.cos(t_edges)/n_h
        hhmin = np.min(hh)
        ercut = hhmin+self.hcut
        self.histo.r, self.histo.h = rr[hh > ercut], hh[hh > ercut]

    def sample_r_h(self, r, h):
        """
            gven a set of elevations (h) and base radii (r), compute the mean profile h(r)
            by taking the binned average values of h and r.
        """
        n_h, r_edges = np.histogram(r, bins=self.bins, density=False, range=(0, self.maxr), weights=None)
        h_h, r_edges = np.histogram(r, bins=self.bins, density=False, range=(0, self.maxr), weights=h)
        cond = n_h > 0.0
        h_h = h_h [cond]
        r_edges = r_edges[:-1][cond]
        n_h = n_h [cond]

        hh, rr = h_h/n_h, r_edges
        hhmin = np.min(hh)
        ercut = hhmin+self.hcut
        self.histo.h, self.histo.r = hh[hh > ercut], rr[hh > ercut]

    def _select_coords(self, use, bins):
        # return lists of array with coordinates falling into the bins that partition the symmetric coordinate.
        try:
            if self.store and use == 'stored': r, h = self._r, self._h
            elif not self.store and use == 'stored': raise(ValueError("The choice use='stored' can only be used if store=True was passed at initialization "))
            elif use == 'frame':  r, h = self.r, self.h
            elif use == 'histogram': r, h= self.histo.r, self.histo.h
            else: raise(ValueError("The parameter use can only take the values 'frame', 'histogram', or 'stored' "))

        except AttributeError:
            raise RuntimeError('No surface atoms or Gibbs surface present')

        # we always comply with the request of cutting all molecules below hcut from the minimum
        if self.hcut > 0:
            hmin = np.min(h)
            cond = h > hmin+self.hcut
            r,h=r[cond],h[cond]
        if bins > 1:
            # TODO: extend for histogram and stored
            if use != 'frame' : raise(ValueError("bins>1 can be used only with use='frame'"))
            hr,hh = [],[]
            symm = self.phi[cond] if self.periodic is None  else self.z[cond]
            limit = 2*np.pi if self.periodic is None else self.substrate.dimensions[self.periodic]
            # set the right edges of the bins. This assumes that shifted_pos() has been called on the
            # coordinates (hence, this function should not be public)
            binvals = np.linspace(-limit/2.,limit/2., bins+1)[1:-1]
            # this will put in bin index 0 eveything close to or below 0.0 and in bin index nbins-1 everyhin
            # close to or above limit
            inds = np.digitize(symm, binvals)
            for i in range(bins):
                hr.append(r[inds==i])
                hh.append(h[inds==i])
        else:
            hr,hh=[r],[h]
        return hr,hh

    def fit_arc(self, use='frame', nonlinear=True, bins=1):
        """ fit an arc through the profile h(r) sampled by the class

            :param str   use        : 'frame'    : use the positions of the current frame only (default)
                                      'histogram': use the binned values sampled so far
                                      'stored'   : use the stored surface atoms positions, if the option store=True was passed at initialization
            :param int   bins       : the number of bins to use along the symmetry direction (cylinder axis, azimuthal angle)

            return:
                a list including, for each of the bins:
                : tuple                 : radius, base radius, cos(theta), center
        """
        r, h = self._select_coords(use,bins=bins)

        return self._fit_arc(r, h, nonlinear=nonlinear)

    def fit_arcellipse(self,use='frame',nonlinear=True,bins=1):

        """  fit an ellipse through the points sampled by the class. See implementation details in _fit_ellipse()
            :param str   use        : 'frame'    : use the positions of the current frame only (default)
                                      'histogram': use the binned values sampled so far
                                      'stored'   : use the stored surface atoms positions, if the option store=True
                                                   was passed at initialization
            :return:
                a list including, for each of the bins:
                parms  : parameters of the ellipse equation in general form:
                         a[0] x^2 + a[1] x y + a[2] y^2 + a[3] x + a[4] y = 0
                parmsc : dictionary of parameters in canoncial form: (a,b, x0,y0,phi, e)
                         with a,b the major and minor semiaxes, x0,y0 the center, phi  the angle
                         (in rad) between x axis and major axis, and e the eccentricity.
                theta1 : right contact angle
                theta2 : left contact angle
        """

        r, h = self._select_coords(use,bins=bins)
        return self._fit_ellipse(r, h, nonlinear=nonlinear, off=0.0)

