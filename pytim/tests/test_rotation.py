# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: test_rotation
    =====================
"""


class TestRotation():

    """
    This is a collection of test to check
    that the algorithms are behaving properly if
    the interface is rotated in space.

    >>> # TEST:1, ITIM+GITIM, flat interface
    >>> import MDAnalysis as mda
    >>> import pytim
    >>> import numpy as np
    >>> from pytim.datafiles import WATER_GRO
    >>>
    >>> for method in [pytim.ITIM , pytim.GITIM] :
    ...     u = mda.Universe(WATER_GRO)
    ...     positions = np.copy(u.atoms.positions)
    ...     oxygens = u.select_atoms('name OW')
    ...     interface = method(u,itim_group=oxygens,molecular=False)
    ...     #interface.writepdb(method.__name__+'.ref.pdb') ; # debug
    ...     ref_box = np.copy(u.dimensions)
    ...     ref_ind   = np.sort(np.copy(interface.atoms.indices))
    ...     ref_pos   = np.copy(interface.atoms.positions)
    ...
    ...     u.atoms.positions = np.roll(positions,1,axis=1)
    ...     box = np.roll(ref_box[:3],1)
    ...     ref_box[:3] =  box
    ...     u.dimensions = ref_box
    ...     interface = method(u,itim_group=oxygens,molecular=False)
    ...     ind = np.sort(interface.atoms.indices)
    ...     #interface.writepdb(method.__name__+'.pdb') ; # debug
    ...     cond = (ref_ind == ind )
    ...     if np.all(cond) ==  False:
    ...         miss1 = (np.in1d(ref_ind,ind)==False).sum()
    ...         miss2 = (np.in1d(ind,ref_ind)==False).sum()
    ...         percent = (miss1 + miss2)*1.0/len(ref_ind) * 100.
    ...         if percent > 6: # this should be 0 for ITIM, and ~5
    ...                         # for GITIM, with this config+alpha
    ...             print miss1+miss2
    ...             print  " differences in indices in method",
    ...             print  method.__name__, " == ",percent," %"

    >>> del interface
    >>> del u

    >>> # TEST:2, GITIM, micelle
    >>> import MDAnalysis as mda
    >>> import pytim
    >>> import numpy as np
    >>> from pytim.datafiles import MICELLE_PDB
    >>>
    >>> for method in [pytim.GITIM] :
    ...     u = mda.Universe(MICELLE_PDB)
    ...     positions = np.copy(u.atoms.positions)
    ...     DPC = u.select_atoms('resname DPC')
    ...     interface = method(u,itim_group=DPC,molecular=False)
    ...     #interface.writepdb(method.__name__+'.ref.pdb') ; # debug
    ...     ref_box = np.copy(u.dimensions)
    ...     ref_ind   = np.sort(np.copy(interface.atoms.indices))
    ...     ref_pos   = np.copy(interface.atoms.positions)
    ...
    ...     u.atoms.positions = np.roll(positions,1,axis=1)
    ...     box = np.roll(ref_box[:3],1)
    ...     ref_box[:3] =  box
    ...     u.dimensions = ref_box
    ...     interface = method(u,itim_group=DPC,molecular=False)
    ...     ind = np.sort(interface.atoms.indices)
    ...     #interface.writepdb(method.__name__+'.pdb') ; # debug
    ...     cond = (ref_ind == ind )
    ...     if np.all(cond) ==  False:
    ...         miss1 = (np.in1d(ref_ind,ind)==False).sum()
    ...         miss2 = (np.in1d(ind,ref_ind)==False).sum()
    ...         percent = (miss1 + miss2)*1.0/len(ref_ind) * 100.
    ...         if percent > 6: # should be ~ 4 % for this system
    ...             print miss1+miss2
    ...             print  " differences in indices in method",
    ...             print  method.__name__, " == ",percent," %"

    >>> del interface
    >>> del u



    """

    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
