# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#from pytim.patches import patchTrajectory, patchOpenMM, patchMDTRAJ

from .simple_interface import SimpleInterface
from .itim import ITIM
from .gitim import GITIM
from .sasa import SASA
from .willard_chandler import WillardChandler
from . import observables, utilities, datafiles
from .version import __version__
import warnings
try: # waiting for numpy2 support in mdtraj>=1.10.2
    from .patches import patchMDTRAJ_ReplacementTables
    patchMDTRAJ_ReplacementTables()
except:
    pass

warnings.filterwarnings(
    "ignore",
    'Failed to guess the mass for the following*')  # To ignore warnings in MDA
