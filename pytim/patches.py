# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function


def patchVirial():
    import libmdaxdr
    import MDAnalysis
    def get_virials(self):
        """Virials on each :class:`Atom` in the :class:`AtomGroup`.
    
        A :class:`numpy.ndarray` with
        :attr:`~numpy.ndarray.shape`\ ``=(``\ :attr:`~AtomGroup.n_atoms`\ ``, 3)``
        and :attr:`~numpy.ndarray.dtype`\ ``=numpy.float32``.
    
        The virials can be changed by assigning an array of the appropriate
        shape, i.e. either ``(``\ :attr:`~AtomGroup.n_atoms`\ ``, 3)`` to assign
        individual forces or ``(3,)`` to assign the *same* force to all
        :class:`Atoms<Atom>` (e.g. ``ag.virials = array([0,0,0])`` will give all
        :class:`Atoms<Atom>` a zero :attr:`~Atom.virial`).
    
        Raises
        ------
        ~MDAnalysis.exceptions.NoDataError
            If the :class:`~MDAnalysis.coordinates.base.Timestep` does not
            contain :attr:`~MDAnalysis.coordinates.base.Timestep.forces`.
        """
        ts = self.universe.trajectory.ts
        try:
            return ts.virials[self.ix]
        except (AttributeError, NoDataError):
            raise NoDataError("Timestep does not contain virials")

    def set_virials(self):
        return

    def get_pressures(self):
        """Pressures on each :class:`Atom` in the :class:`AtomGroup`.
    
        A :class:`numpy.ndarray` with
        :attr:`~numpy.ndarray.shape`\ ``=(``\ :attr:`~AtomGroup.n_atoms`\ ``, 3)``
        and :attr:`~numpy.ndarray.dtype`\ ``=numpy.float32``.
    
        The pressures can be changed by assigning an array of the appropriate
        shape, i.e. either ``(``\ :attr:`~AtomGroup.n_atoms`\ ``, 3)`` to assign
        individual forces or ``(3,)`` to assign the *same* force to all
        :class:`Atoms<Atom>` (e.g. ``ag.pressures = array([0,0,0])`` will give all
        :class:`Atoms<Atom>` a zero :attr:`~Atom.pressure`).
    
        Raises
        ------
        ~MDAnalysis.exceptions.NoDataError
            If the :class:`~MDAnalysis.coordinates.base.Timestep` does not
            contain :attr:`~MDAnalysis.coordinates.base.Timestep.forces`.
        """
        from numpy import expand_dims
        ts = self.universe.trajectory.ts
        #try:
        press =  ts.virials[self.ix] 
        press = press + 16.6054 * 0.01 * ts.velocities[self.ix]**2  * expand_dims(self.universe.atoms.masses[self.ix] ,axis=1)
        return press
        #except (AttributeError, NoDataError):
        #    raise NoDataError("Timestep does not contain forces")

    def set_pressures(self):
        return

    MDAnalysis.core.groups.AtomGroup.virials = property(get_virials,set_virials)
    MDAnalysis.core.groups.AtomGroup.pressures = property(get_pressures,set_pressures)
    
    class TRRReaderVIR(MDAnalysis.coordinates.XDR.XDRBaseReader):
        from libmdaxdr import TRRFile as TRRFileVIR
        from MDAnalysis.coordinates.TRR import TRRWriter
    
        def convert_virials_from_native(self, virial, inplace=True):
            """Conversion of forces array *virial* from native to base units
    
            Parameters
            ----------
            virial : array_like
              Forces to transform
            inplace : bool (optional)
              Whether to modify the array inplace, overwriting previous data
    
            Note
            ----
            By default, the input *virial* is modified in place and also returned.
            In-place operations improve performance because allocating new arrays
            is avoided.
    
            .. versionadded:: 0.7.7
            """
    
            f = units.get_conversion_factor(
                'force', self.units['force'], flags['force_unit'])
            f = f *  units.get_conversion_factor(
                'length', self.units['length'], flags['length_unit'])
            if f == 1.:
                return virial
            if not inplace:
                return f * virial
            virial *= f
            return virial
    
    
        """Reader for the Gromacs TRR format.
    
        The Gromacs TRR trajectory format is a lossless format. The TRR format can
        store *velocoties* and *forces* in addition to the coordinates. It is also
        used by other Gromacs tools to store and process other data such as modes
        from a principal component analysis.
    
        The lambda value is written in the data dictionary of the returned
        :class:`Timestep`
    
        Notes
        -----
        See :ref:`Notes on offsets <offsets-label>` for more information about
        offsets.
    
        """
        format = 'TRR'
        units = {'time': 'ps', 'length': 'nm', 'velocity': 'nm/ps',
                 'force': 'kJ/(mol*nm)'}
        _writer = TRRWriter
        _file = TRRFileVIR
    
        def _frame_to_ts(self, frame, ts):
            from MDAnalysis.lib.mdamath import triclinic_box as triclinic_box
    
            """convert a trr-frame to a mda TimeStep"""
            ts.time = frame.time
            ts.frame = self._frame
            ts.data['step'] = frame.step
    
            ts.has_positions = frame.hasx
            ts.has_velocities = frame.hasv
            ts.has_forces = frame.hasf
            ts.has_virials = frame.hasvir
    
            ts.dimensions = triclinic_box(*frame.box)
    
            if self.convert_units:
                self.convert_pos_from_native(ts.dimensions[:3])
    
            if ts.has_positions:
                if self._sub is not None:
                    ts.positions = frame.x[self._sub]
                else:
                    ts.positions = frame.x
                if self.convert_units:
                    self.convert_pos_from_native(ts.positions)
    
            if ts.has_velocities:
                if self._sub is not None:
                    ts.velocities = frame.v[self._sub]
                else:
                    ts.velocities = frame.v
                if self.convert_units:
                    self.convert_velocities_from_native(ts.velocities)
    
            if ts.has_forces:
                if self._sub is not None:
                    ts.forces = frame.f[self._sub]
                else:
                    ts.forces = frame.f
                if self.convert_units:
                    self.convert_forces_from_native(ts.forces)
    
            if ts.has_virials:
                if self._sub is not None:
                    ts.virials = frame.f[self._sub]
                else:
                    ts.virials = frame.vir
                if self.convert_units:
                    self.convert_forces_from_native(ts.virials)
    
    
            ts.data['lambda'] = frame.lmbda
    
            return ts

    MDAnalysis.coordinates.TRR.TRRReader = TRRReaderVIR
    MDAnalysis.coordinates.TRR.TRRFile = libmdaxdr.TRRFile
    MDAnalysis.lib.formats.libmdaxdr.TRRFile = libmdaxdr.TRRFile
    MDAnalysis.coordinates.TRR.TRRReader._file = libmdaxdr.TRRFile



def patchNumpy():
    # this try/except block patches numpy and provides _validate_lengths
    # to skimage<=1.14.1
    import numpy
    try:
        numpy.lib.arraypad._validate_lengths
    except AttributeError:
        def patch_validate_lengths(ar, crop_width):
            return numpy.lib.arraypad._as_pairs(crop_width, ar.ndim, as_index=True)
        numpy.lib.arraypad._validate_lengths = patch_validate_lengths


def patchTrajectory(trajectory, interface):
    """ Patch the MDAnalysis trajectory class

        this patch makes the layer assignement being automatically
        called whenever a new frame is loaded.
    """
    try:
        trajectory.interface
        trajectory.interface = interface

    except AttributeError:
        trajectory.interface = interface
        trajectory.interface._frame = trajectory.frame
        trajectory.original_read_next_timestep = trajectory._read_next_timestep

        trajectory.original_read_frame_with_aux = trajectory._read_frame_with_aux

        class PatchedTrajectory(trajectory.__class__):
            def _read_next_timestep(self, ts=None):
                tmp = self.original_read_next_timestep(ts=ts)
                if self.interface.autoassign is True:
                    self.interface._assign_layers()
                    self.interface._frame = self.frame
                return tmp

            def _read_frame_with_aux(self, frame):
                if frame != self.frame:
                    tmp = self.original_read_frame_with_aux(frame)
                    if self.interface.autoassign is True:
                        if self.interface._frame != self.frame:
                            self.interface._assign_layers()
                            self.interface._frame = self.frame
                    return tmp
                return self.ts

        oldname = trajectory.__class__.__name__
        oldmodule = trajectory.__class__.__module__

        PatchedTrajectory.__name__ = oldname
        PatchedTrajectory.__module__ = oldmodule
        trajectory.__class__ = PatchedTrajectory


def patchOpenMM(simulation, interface):
    #       return _openmm.CompoundIntegrator_step(self, steps)
    """ Patch the Openmm Simulation class

    """
    from simtk.unit import angstrom as openmm_AA
    try:
        simulation.interface
    except AttributeError:
        simulation.interface = interface
        simulation.original_step = simulation.step

        class PatchedOpenMMSimulation(simulation.__class__):
            def step(self, steps):
                tmp = self.original_step(steps)
                pos = self.context.getState(getPositions=True).getPositions(
                    asNumpy=True).value_in_unit(openmm_AA)
                self.interface.universe.atoms.positions = pos
                if self.interface.autoassign is True:
                    self.interface._assign_layers()
                return tmp

        oldname = simulation.__class__.__name__
        oldmodule = simulation.__class__.__module__

        PatchedOpenMMSimulation.__name__ = oldname
        PatchedOpenMMSimulation.__module__ = oldmodule
        simulation.__class__ = PatchedOpenMMSimulation


def patchMDTRAJ_ReplacementTables():
    try:
        import mdtraj
        if not (mdtraj.formats.pdb.PDBTrajectoryFile._atomNameReplacements == {}):
            print('Warning: mdtraj has not been patched correctly. The trajectory must be loaded *after* importing pytim: some atom names might have been replaced')

        @staticmethod
        def _NoReplacementTables():
            mdtraj.formats.pdb.PDBTrajectoryFile._atomNameReplacements = {}

        mdtraj.formats.pdb.PDBTrajectoryFile._loadNameReplacementTables = _NoReplacementTables
    except ImportError:
        pass


def patchMDTRAJ(trajectory, universe):
    """ Patch the mdtraj Trajectory class

        automates the data exchange between MDAnalysis and mdtraj classes

        Example:

        >>> import mdtraj
        >>> import pytim
        >>> from pytim.datafiles import WATER_GRO, WATER_XTC
        >>> t = mdtraj.load_xtc(WATER_XTC,top=WATER_GRO)
        >>> inter = pytim.ITIM(t)


    """

    try:
        trajectory.universe
    except AttributeError:
        trajectory.universe = universe

        class PatchedMdtrajTrajectory(trajectory.__class__):
            def __getitem__(self, key):
                slice_ = self.slice(key)
                patchMDTRAJ(slice_, universe)

                if isinstance(key, int):
                    # mdtraj uses nm as distance unit, we need to convert to
                    # Angstrom for MDAnalysis
                    slice_.universe.atoms.positions = slice_.xyz[0] * 10.0
                    dimensions = slice_.universe.dimensions[:]
                    dimensions[0:3] = slice_.unitcell_lengths[0:3] * 10.0
                    slice_.universe.dimensions = dimensions
                    if slice_.universe.trajectory.interface.autoassign is True:
                        slice_.universe.trajectory.interface._assign_layers()
                return slice_

        oldname = trajectory.__class__.__name__
        oldmodule = trajectory.__class__.__module__

        PatchedMdtrajTrajectory.__name__ = oldname
        PatchedMdtrajTrajectory.__module__ = oldmodule
        trajectory.__class__ = PatchedMdtrajTrajectory
