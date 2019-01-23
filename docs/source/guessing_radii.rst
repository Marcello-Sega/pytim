Choosing the atomic radii
*************************

Summary
-------
The result of interfacial atoms identification can depend markedly on the choice of atomic radii, especially with methods like :class:`~pytim.gitim.GITIM`.
We review how :doc:`Pytim <quick>` chooses which values to use, and how this can be overridden by the user.

Example
-------

**The problem**

Configuration file formats such as the gromos one do not bear
information neither about the atom types nor about the atomic radii.
In this case :doc:`Pytim <quick>` (starting from v 0.6.1) chooses the corresponding radius using the
closest lexycographic match (with higher weight on the first letter)
to the Gromos 43A1 forcefield types. Alternatively, it is possible
to choose parameters from other forcefields provided with :doc:`Pytim <quick>`
(Amber 03 and Charmm 27, as of v 0.6.1), or to provide a custom one.
We make here several examples using a methanol box:


>>> import MDAnalysis as mda
>>> import numpy as np
>>> import pytim
>>> from pytim.datafiles import *
>>> u = mda.Universe(METHANOL_GRO)
...

You can safely ignore any warning about missing atomtypes / masses in
MDAnalysis_, and deal with atomtypes only through the :doc:`Pytim <quick>` interface.

.. _MDAnalysis: http://www.mdanalysis.org/

**Case 1: built-in forcefields**

As there is no specific information in the gromos input file regarding atomic types,  these are
assigned by MDAnalysis_ using the first letter of the atom name.

>>> print (u.residues[0].atoms.types)
['M' 'O' 'H']

>>> print (u.residues[0].atoms.names)
['Me1' 'O2' 'H3']

When the interface is initialized, :doc:`Pytim <quick>` searches by default for the best
match from the gromos 43A1 forcefield. It is possible to see explicitly the
choices by passing the option :py:obj:`warnings=True`

>>> interface = pytim.ITIM(u, warnings=True)
guessed radii:  {'H3': 0.0, 'Me1': 1.8230510193786977, 'O2': 1.312927017714457} You can override this by using, e.g.: pytim.ITIM (u,radii_dict={ 'H3':1.2 , ... } )

The Gromos 43A1, Amber 03 and Charmm 27 forcefields are accessible through labels provided in  the :py:mod:`~pytim.datafiles` module. The radii can be extracted
using the :py:func:`pytim_data.vdwradii()`  function

>>> gromos  = pytim_data.vdwradii(G43A1_TOP)

>>> amber   = pytim_data.vdwradii(AMBER03_TOP)

>>> charmmm = pytim_data.vdwradii(CHARMM27_TOP)

and can be passed to :doc:`Pytim <quick>` through the :py:obj:`radii_dict` option, for example:

>>> interface = pytim.ITIM(u, radii_dict = amber )


**Case 2: custom dictionary of radii**

The custom dictionary should provide the atomtypes as keys
and the radii as values. This database will override the standard one (new behavior in 0.6.1):

>>> u = mda.Universe(METHANOL_GRO)
>>> mydict    = {'Me':1.6, 'O':1.5, 'H':0.0}
>>> interface = pytim.ITIM(u,radii_dict=mydict)
>>> print (u.residues[0].atoms.radii)
[1.6 1.5 0. ]

**Case 3: overriding the default match**

The values of the van der Waals radii present in a topology can be read
using the :py:func:`pytim_data.vdwradii()`  function:

>>> u = mda.Universe(METHANOL_GRO)
>>> # Load the radii database from the G43A1 forcefield
>>> gromos = pytim_data.vdwradii(G43A1_TOP)
>>> # check which types are available
>>> print (sorted(list(gromos.keys())))
['AR', 'BR', 'C', 'CA2+', 'CChl', 'CCl4', 'CDmso', 'CH1', 'CH2', 'CH3', 'CH4', 'CL', 'CL-', 'CLChl', 'CLCl4', 'CMet', 'CR1', 'CU1+', 'CU2+', 'DUM', 'F', 'FE', 'H', 'HC', 'HChl', 'IW', 'MG2+', 'MNH3', 'MW', 'N', 'NA+', 'NE', 'NL', 'NR', 'NT', 'NZ', 'O', 'OA', 'ODmso', 'OM', 'OMet', 'OW', 'OWT3', 'OWT4', 'P', 'S', 'SDmso', 'SI', 'ZN2+']

In some cases, the automated procedure does not yield the expected
(or wanted) match.  To fix this one can just update the database
by specify exacly the *atom names* (not the types, as it was in
versions before 0.6.1)

>>> gromos['Me1']=gromos['CMet']
>>> gromos['O2']=gromos['OMet']
>>> interface = pytim.ITIM(u,radii_dict=gromos)


.. toctree::

.. raw:: html
   :file: analytics.html
