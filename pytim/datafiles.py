# vim: set expandtab:
# vim: set tabstop=4:
"""
Location of data files for Pytim examples and tests 
====================================================

Real MD simulation data are stored in the ``data/`` sub
directory. Use as ::

  from pytim.datafiles import *

"""

__all__ = [
    "WATER_GRO", # GROMACS single frame, water/vapour interface
    "G43A1_TOP", # GROMOS 43a1 nonbonded parameters, from gromacs distribution
    "pytim_data",# class to access the data
]

from pkg_resources import resource_filename
import re as re


class Data():

    def __init__(self):
        self.type=dict()
        self.format=dict()

    def add(self,filename,type,format):
        self.type[filename]=type
        self.format[filename]=format

    nm2angs=10.0
    
    def vdwradii(self,filename):
        if self.type[filename] == 'topol':
            if self.format[filename] == 'GMX':
                with open(filename) as _f:
                    scan=False
                    _radii=dict()
                    for _line in _f:
                        if (scan and re.match('^ *\[',_line)):
                            return _radii
                        if (scan):
                            try:
                                _data=(_line.split(";")[0]).split()
                                _atom = str(_data[0])
                                _c6  = float(_data[5])
                                _c12 = float(_data[6])
                                if (_c6 == 0.0 or _c12 == 0.0) :
                                    _sigma=0.0
                                else:
                                    _sigma = (_c12/_c6)**(1./6.) * self.nm2angs
                                _radii[_atom]=_sigma
                            except:
                                pass
                        if (re.match('^ *\[ *atomtypes *\]',_line)):
                            scan=True
                return _radii
                
     
pytim_data=Data()

## NOTE: to add a new datafile, make sure it is listed in setup.py (in the root directory)
##       in the package_data option (a glob like 'data/*' is usually enough)
WATER_GRO = resource_filename('pytim', 'data/water.gro')           
pytim_data.add(WATER_GRO  ,  'config', 'GRO') 

G43A1_TOP = resource_filename('pytim', 'data/ffg43a1.nonbonded.itp') # This should be the last line: clean up namespace
pytim_data.add(G43A1_TOP  , 'topol' , 'GMX')



del resource_filename
