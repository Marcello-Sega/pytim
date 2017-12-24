# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function
import numpy as np
from distutils.version import LooseVersion
import MDAnalysis


def _writepdb(interface, filename='layers.pdb', centered='no', group='all', multiframe=True):
    """ Write the frame to a pdb file, marking the atoms belonging
        to the layers with different beta factors.

        :param str       filename   : the output file name
        :param str       centered   : 'origin', 'middle', or 'no'
        :param AtomGroup group      : if 'all' is passed, the universe is used
        :param bool      multiframe : append to pdb file if True

        Example: save the positions (centering the interface in the cell)
                 without appending

        >>> import pytim
        >>> import MDAnalysis as mda
        >>> from pytim.datafiles import WATER_GRO
        >>> u = mda.Universe(WATER_GRO)
        >>> interface = pytim.ITIM(u)
        >>> interface.writepdb('layers.pdb',multiframe=False)

        Example: save the positions without centering the interface. This will
                 not shift the atoms from the original position (still, they
                 will be put into the basic cell).
                 The :multiframe: option set to :False: will overwrite the file

        >>> interface.writepdb('layers.pdb',centered='no')

    """
    if isinstance(group, interface.universe.atoms.__class__):
        interface.group = group
    else:
        interface.group = interface.universe.atoms

    temp_pos = np.copy(interface.universe.atoms.positions)
    options = {'no': False, False: False, 'middle': True, True: True}

    if options[centered] != interface.do_center:
        # i.e. we don't have already what we want ...
        if interface.do_center == False:  # we need to center
            interface.center(planar_to_origin=True)
        else:  # we need to put back the original positions
            try:
                # original_positions are (must) always be defined
                interface.universe.atoms.positions = interface.original_positions
            except:
                raise AttributeError
    try:
        # it exists already, let's add information about the box, as
        # MDAnalysis forgets to do so for successive frames. A bugfix
        # should be on the way for the next version...
        interface.PDB[filename].CRYST1(
            interface.PDB[filename].convert_dimensions_to_unitcell(interface.universe.trajectory.ts))
    except:
        if LooseVersion(interface._MDAversion) >= LooseVersion('0.16'):
            bondvalue = None
        else:
            bondvalue = False
        interface.PDB[filename] = MDAnalysis.Writer(
            filename, multiframe=multiframe,
            n_atoms=interface.group.atoms.n_atoms,
            bonds=bondvalue
        )
    interface.PDB[filename].write(interface.group.atoms)
    interface.PDB[filename].pdbfile.flush()
    interface.universe.atoms.positions = np.copy(temp_pos)
