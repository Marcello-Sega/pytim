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
from .patches import patchNumpy, patchMDTRAJ_ReplacementTables
patchNumpy()
patchMDTRAJ_ReplacementTables()

warnings.filterwarnings(
    "ignore",
    'Failed to guess the mass for the following*')  # To ignore warnings in MDA
