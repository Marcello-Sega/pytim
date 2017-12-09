# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
"""
Location of data files for Pytim examples and tests
====================================================

Real MD simulation data are stored in the ``data/`` subdirectory.


    Example: load an example trajectory

    >>> import MDAnalysis as mda
    >>> import pytim
    >>> from pytim.datafiles import *
    >>> u         = mda.Universe(WATER_GRO,WATER_XTC)
    >>> print u
    <Universe with 12000 atoms>

    Example: list all configurations

    >>> for config in sorted(pytim_data.config):
    ...     print("{:20s} {:s}".format(config,pytim_data.description[config]))
    CCL4_WATER_GRO       Carbon tetrachloride/TIP4p water interface
    FULLERENE_PDB        fullerene
    GLUCOSE_PDB          solvated beta-d-glucose
    MICELLE_PDB          DPC micelle
    WATERSMALL_GRO       small SPC water/vapour interface
    WATER_520K_GRO       SPC/E water/vapour interface, 520K
    WATER_550K_GRO       SPC/E water/vapour interface, 550K
    WATER_GRO            SPC water/vapour interface
    WATER_PDB            SPC water/vapour interface
    WATER_XYZ            SPC water/vapour interface




    Example: list all topologies

    >>> print pytim_data.topol
    ['WATER_LMP_DATA', 'AMBER03_TOP', 'G43A1_TOP', 'CHARMM27_TOP']



    Example: list all trajectories

    >>> print pytim_data.traj
    ['WATER_LMP_XTC', 'WATER_XTC']



    Example: list all files, file type, file format and description

    >>> for label in  pytim_data.label:
    ...      type        = pytim_data.type[label]
    ...      format      = pytim_data.format[label]
    ...      description = pytim_data.description[label]


"""

__all__ = [
    "CCL4_WATER_GRO",        # GROMACS single frame, carbon tetrachloride / water interface
    "WATER_GRO",             # GROMACS single frame, water/vapour interface
    "WATER_LMP_DATA",        # LAMMPS topology for WATER_LAMMPS, water/vapour interface
    "WATER_LMP_XTC",         # LAMMPS trajectory, water/vapour interface
    "WATER_PDB",             # PDB single frame, water/vapour interface, same as WATER_GRO
    "WATER_XYZ",             # XYZ single frame, water/vapour interface, same as WATER_GRO
    "WATERSMALL_GRO",        # GROMACS single frame, SPC water/vapour interface
    "WATER_520K_GRO",        # GROMACS single frame, SPC/E water/vapour interface,520 K
    "WATER_550K_GRO",        # GROMACS single frame, SPC/E water/vapour interface,550 K
    "METHANOL_GRO",          # methanol/vapour interface with molecules in the  vapour phase
    "ILBENZENE_GRO",         # Ionic liquid/benzene, partial miscibility
    "MICELLE_PDB",           # PDB of dodecylphosphocholine micelle in water
    "FULLERENE_PDB",         # PDB of C60
    "GLUCOSE_PDB",           # PDB of solvated beta-d-glucose
    "WATER_XTC",             # GROMACS trajectory, 100 frames, water/vapour interface
    "G43A1_TOP",             # GROMOS 43a1 nonbonded parameters, from the gromacs distribution
    "AMBER03_TOP",           # AMBER03 nonbonded parameters, from the gromacs distribution
    "CHARMM27_TOP",          # CHARM27 nonbonded parameters, from the gromacs distribution
    "pytim_data",            # class to access the data
    "_TEST_ORIENTATION_GRO",  # test file
    "_TEST_PROFILE_GRO",  # test file
]


from pkg_resources import resource_filename
import tempfile
import os.path
import re as re
import urllib
import hashlib
import tempfile


class Data(object):
    """" a class for storing/accessing configurations, trajectories, topologies
    """

    def fetch(self, name, tmpdir=None):
        """ Fetch a sample trajectory from the github repository.

            Have a look at https://github.com/Marcello-Sega/pytim/raw/extended_datafiles/files/
            for the available files

            Example:

            >>> import MDAnalysis as mda
            >>> import pytim
            >>> from pytim.datafiles import WATERSMALL_GRO

            >>> # tmpdir here is specified only for travis
            >>> import os
            >>> WATERSMALL_TRR = pytim.datafiles.pytim_data.fetch('WATERSMALL_TRR',tmpdir='./')
            checking presence of a cached copy... not found. Fetching remote file... done.

            >>> u = mda.Universe(WATERSMALL_GRO,WATERSMALL_TRR)
            >>> os.unlink('./'+WATERSMALL_TRR)
            >>> print u
            <Universe with 648 atoms>

        """

        filename = name.replace("_", ".")
        if tmpdir is None:
            dirname = tempfile.gettempdir()
        else:
            dirname = tmpdir
        urlbase_md5 = 'https://raw.githubusercontent.com/Marcello-Sega/pytim/extended_datafiles/files/'
        urlbase = 'https://github.com/Marcello-Sega/pytim/raw/extended_datafiles/files/'
        try:
            md5 = urllib.urlopen(urlbase_md5 + filename + '.MD5').readline()
            print "checking presence of a cached copy...",
            md5_local = hashlib.md5(
                open(dirname + filename, 'rb').read()).hexdigest()
            if md5_local in md5:
                print "found"
                return dirname + filename
        except:
            pass
        print "not found. Fetching remote file...",
        newfile = urllib.urlopen(urlbase + filename + '?raw=true')
        with open(dirname + filename, 'wb') as output:
            output.write(newfile.read())
        print "done."
        return dirname + filename

    def _generate_data_property(self, name):
        labels = [label for label, val in self.type.iteritems() if val == name]
        return list(set(labels) & set(self.label))

    @property
    def config(self):
        return self._generate_data_property('config')

    @property
    def topol(self):
        return self._generate_data_property('topol')

    @property
    def traj(self):
        return self._generate_data_property('traj')

    def __init__(self):
        self._label = list()
        self.label = list()
        self.file = dict()
        self.type = dict()
        self.format = dict()
        self.description = dict()

    def add(self, label, filetype, fileformat, desc):
        self._label.append(label)
        if label[0] is not '_':
            self.label.append(label)
        self.file[label] = globals()[label]
        file = self.file[label]
        self.type[file] = filetype
        self.type[label] = filetype
        self.format[file] = fileformat
        self.format[label] = fileformat
        self.description[file] = desc
        self.description[label] = desc

    nm2angs = 10.0

    def vdwradii(self, filename):
        input_type = 'sigeps'
        if self.type[filename] == 'topol' and self.format[filename] == 'GMX':
            with open(filename) as f:
                scan = False
                radii = dict()
                for line in f:
                    if (scan and re.match('^ *\[', line)):
                        return radii
                    if (scan):
                        try:
                            if re.match(';.*name.*c6 *c12',line):
                                input_type = 'c6c12'
                            data = (line.split(";")[0]).split()
                            atom , sigma, epsilon = str(data[0]) ,float(data[5]) ,float(data[6])
                            if input_type == 'c6c12':
                                c6 , c12 = sigma, epsilon 
                                if (c6 == 0.0 or c12 == 0.0):
                                    sigma = 0.0
                                else:
                                    sigma = (c12 / c6)**(1. / 6.) * self.nm2angs
                            else:
                                sigma *= self.nm2angs

                            radii[atom] = sigma / 2.
                        except Exception:
                            pass
                    if (re.match('^ *\[ *atomtypes *\]', line)):
                        scan = True
            return radii


pytim_data = Data()

# NOTE: to add a new datafile, make sure it is listed in setup.py (in the root directory)
# in the package_data option (a glob like 'data/*' is usually enough)
CCL4_WATER_GRO = resource_filename('pytim', 'data/CCL4.H2O.GRO')
pytim_data.add('CCL4_WATER_GRO',  'config', 'GRO',
               'Carbon tetrachloride/TIP4p water interface')

WATER_GRO = resource_filename('pytim', 'data/water.gro')
pytim_data.add('WATER_GRO',  'config', 'GRO', 'SPC water/vapour interface')

WATER_LMP_XTC = resource_filename('pytim', 'data/water_lmp.xtc')
pytim_data.add('WATER_LMP_XTC',  'traj', 'LAMMPS',
               'SPC water/vapour interface')

WATER_PDB = resource_filename('pytim', 'data/water.pdb')
pytim_data.add('WATER_PDB',  'config', 'PDB', 'SPC water/vapour interface')

WATER_XYZ = resource_filename('pytim', 'data/water.xyz')
pytim_data.add('WATER_XYZ',  'config', 'XYZ', 'SPC water/vapour interface')

MICELLE_PDB = resource_filename('pytim', 'data/micelle.pdb')
pytim_data.add('MICELLE_PDB',  'config', 'GRO', 'DPC micelle')

FULLERENE_PDB = resource_filename('pytim', 'data/fullerene.pdb')
pytim_data.add('FULLERENE_PDB',  'config', 'PDB', 'fullerene')

GLUCOSE_PDB   = resource_filename('pytim', 'data/glucose.pdb')
pytim_data.add('GLUCOSE_PDB',  'config', 'PDB', 'solvated beta-d-glucose')

WATERSMALL_GRO = resource_filename('pytim', 'data/water-small.gro')
pytim_data.add('WATERSMALL_GRO',  'config', 'GRO',
               'small SPC water/vapour interface')

WATER_520K_GRO = resource_filename('pytim', 'data/water_520K.gro')
pytim_data.add('WATER_520K_GRO',  'config', 'GRO',
               'SPC/E water/vapour interface, 520K')

WATER_550K_GRO = resource_filename('pytim', 'data/water_550K.gro')
pytim_data.add('WATER_550K_GRO',  'config', 'GRO',
               'SPC/E water/vapour interface, 550K')

METHANOL_GRO = resource_filename('pytim', 'data/methanol.gro')
pytim_data.add('METHANOL_GRO',  'conf', 'GRO', 'methanol/vapour interface')

ILBENZENE_GRO = resource_filename('pytim', 'data/ilbenzene.gro')
pytim_data.add('ILBENZENE_GRO',  'conf', 'GRO', 'BMIM PF4 / benzene interface')

WATER_XTC = resource_filename('pytim', 'data/water.xtc')
pytim_data.add('WATER_XTC',  'traj', 'XTC',
               'SPC water/vapour interface trajectory')

_TEST_ORIENTATION_GRO = resource_filename(
    'pytim', 'data/_test_orientation.gro')
pytim_data.add('_TEST_ORIENTATION_GRO',  'config', 'GRO', 'test file')

_TEST_PROFILE_GRO = resource_filename('pytim', 'data/_test_profile.gro')
pytim_data.add('_TEST_PROFILE_GRO',  'config', 'GRO', 'test file')

WATER_LMP_DATA = resource_filename('pytim', 'data/water_lmp.data')
pytim_data.add('WATER_LMP_DATA', 'topol', 'DATA',
               'LAMMPS topology for WATER_LAMMPS')

G43A1_TOP = resource_filename('pytim', 'data/ffg43a1.nonbonded.itp')
pytim_data.add('G43A1_TOP', 'topol', 'GMX', 'GROMOS 43A1 topology for GROMACS')

AMBER03_TOP = resource_filename('pytim', 'data/ffamber03.nonbonded.itp')
pytim_data.add('AMBER03_TOP', 'topol', 'GMX', 'AMBER 03 topology for GROMACS')

CHARMM27_TOP = resource_filename('pytim', 'data/ffcharmm27.nonbonded.itp')
pytim_data.add('CHARMM27_TOP', 'topol', 'GMX', 'CHARMM 27 topology for GROMACS')


# This should be the last line: clean up namespace
del resource_filename
