"""
Location of data files for Pytim examples and tests 
====================================================

Real MD simulation data are stored in the ``data/`` sub
directory. Use as ::

  from pytim.datafiles import *

"""

__all__ = [
    "WATER_GRO" # GROMACS single frame, water/vapour interface
]

from pkg_resources import resource_filename

WATER_GRO = resource_filename(__name__, 'data/water.gro')

# This should be the last line: clean up namespace
del resource_filename
