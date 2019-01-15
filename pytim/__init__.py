# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#from pytim.patches import PatchTrajectory, PatchOpenMM, PatchMDTRAJ

import numpy

try:
    numpy.lib.arraypad._validate_lengths
except:
    def patch_validate_lengths(ar,crop_width):
        return numpy.lib.arraypad._as_pairs(crop_width, ar.ndim, as_index=True)
    numpy.lib.arraypad._validate_lengths = patch_validate_lengths


from .simple_interface import SimpleInterface
from .itim import ITIM
from .gitim import GITIM
from .sasa import SASA
from .willard_chandler import WillardChandler
from .chacon_tarazona import ChaconTarazona
from . import observables, utilities, datafiles
from .version import __version__

import warnings
warnings.filterwarnings(
    "ignore",
    'Failed to guess the mass for the following*')  # To ignore warnings in MDA
