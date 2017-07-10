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
    MICELLE_PDB          DPC micelle
    WATERSMALL_GRO       small SPC water/vapour interface
    WATER_520K_GRO       SPC/E water/vapour interface, 520K
    WATER_550K_GRO       SPC/E water/vapour interface, 550K
    WATER_GRO            SPC water/vapour interface
    WATER_PDB            SPC water/vapour interface
    WATER_XYZ            SPC water/vapour interface




    Example: list all topologies

    >>> print pytim_data.topol
    ['G43A1_TOP']

    Example: list all trajectories

    >>> print pytim_data.traj
    ['WATER_XTC']


    Example: list all files, file type, file format and description

    >>> for label in  pytim_data.label:
    ...      type        = pytim_data.type[label]
    ...      format      = pytim_data.format[label]
    ...      description = pytim_data.description[label]


"""

__all__ = [
    "CCL4_WATER_GRO",        # GROMACS single frame, carbon tetrachloride / water interface
    "WATER_GRO",             # GROMACS single frame, water/vapour interface
    "WATER_PDB",             # PDB single frame, water/vapour interface, same as WATER_GRO
    "WATER_XYZ",             # XYZ single frame, water/vapour interface, same as WATER_GRO
    "WATERSMALL_GRO",        # GROMACS single frame, SPC water/vapour interface
    "WATER_520K_GRO",        # GROMACS single frame, SPC/E water/vapour interface,520 K
    "WATER_550K_GRO",        # GROMACS single frame, SPC/E water/vapour interface,550 K
    "METHANOL_GRO",          # methanol/vapour interface with molecules in the  vapour phase
    "ILBENZENE_GRO",         # Ionic liquid/benzene, partial miscibility
    "MICELLE_PDB",           # PDB of dodecylphosphocholine micelle in water
    "FULLERENE_PDB",         # PDB of C60
    "WATER_XTC",             # GROMACS trajectory, 100 frames, water/vapour interface
    "G43A1_TOP",             # GROMOS 43a1 nonbonded parameters, from gromacs distribution
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

    def fetch(self,name):
        filename =  name.replace("_", ".")
        dirname = tempfile.gettempdir()
        urlbase_md5 = 'https://raw.githubusercontent.com/Marcello-Sega/pytim/extended_datafiles/files/'
        urlbase = 'https://github.com/Marcello-Sega/pytim/raw/extended_datafiles/files/'
        try:
            md5 = urllib.urlopen(urlbase_md5+filename+'.MD5').readline()
            print "checking presence of a cached copy...",
            md5_local =hashlib.md5(open(dirname+filename, 'rb').read()).hexdigest()
            if md5_local in md5:
                print "found"
                return dirname+filename
        except:
            pass
        print "not found. Fetching remote file...",
        newfile = urllib.urlopen(urlbase+filename+'?raw=true')
        with open(dirname+filename,'wb') as output:
            output.write(newfile.read())
        print "done."
        return dirname+filename

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
        if self.type[filename] == 'topol' and self.format[filename] == 'GMX':
            with open(filename) as _f:
                scan = False
                _radii = dict()
                for _line in _f:
                    if (scan and re.match('^ *\[', _line)):
                        return _radii
                    if (scan):
                        try:
                            _data = (_line.split(";")[0]).split()
                            _atom = str(_data[0])
                            _c6 = float(_data[5])
                            _c12 = float(_data[6])
                            if (_c6 == 0.0 or _c12 == 0.0):
                                _sigma = 0.0
                            else:
                                _sigma = (_c12 / _c6)**(1. / 6.) * self.nm2angs
                            _radii[_atom] = _sigma / 2.
                        except Exception:
                            pass
                    if (re.match('^ *\[ *atomtypes *\]', _line)):
                        scan = True
            return _radii


pytim_data = Data()

# NOTE: to add a new datafile, make sure it is listed in setup.py (in the root directory)
# in the package_data option (a glob like 'data/*' is usually enough)
CCL4_WATER_GRO = resource_filename('pytim', 'data/CCL4.H2O.GRO')
pytim_data.add('CCL4_WATER_GRO',  'config', 'GRO', 'Carbon tetrachloride/TIP4p water interface')

WATER_GRO = resource_filename('pytim', 'data/water.gro')
pytim_data.add('WATER_GRO',  'config', 'GRO', 'SPC water/vapour interface')

WATER_PDB = resource_filename('pytim', 'data/water.pdb')
pytim_data.add('WATER_PDB',  'config', 'PDB', 'SPC water/vapour interface')

WATER_XYZ = resource_filename('pytim', 'data/water.xyz')
pytim_data.add('WATER_XYZ',  'config', 'XYZ', 'SPC water/vapour interface')

MICELLE_PDB = resource_filename('pytim', 'data/micelle.pdb')
pytim_data.add('MICELLE_PDB',  'config', 'GRO', 'DPC micelle')

FULLERENE_PDB = resource_filename('pytim', 'data/fullerene.pdb')
pytim_data.add('FULLERENE_PDB',  'config', 'PDB', 'fullerene')

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

# This should be the last line: clean up namespace
G43A1_TOP = resource_filename('pytim', 'data/ffg43a1.nonbonded.itp')
pytim_data.add('G43A1_TOP', 'topol', 'GMX', 'GROMOS 43A1 topology for GROMACS')


del resource_filename
