# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#from pytim.patches import patchTrajectory, patchOpenMM, patchMDTRAJ

from .patches import patchNumpy
patchNumpy()
import warnings
from .version import __version__
from . import observables, utilities, datafiles
from .chacon_tarazona import ChaconTarazona
from .willard_chandler import WillardChandler
from .sasa import SASA
from .gitim import GITIM
from .itim import ITIM
from .simple_interface import SimpleInterface

warnings.filterwarnings(
    "ignore",
    'Failed to guess the mass for the following*')  # To ignore warnings in MDA
