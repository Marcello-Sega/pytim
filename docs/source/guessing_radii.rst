Choosing the atomic radii
*************************

Summary
-------
The result of interfacial atoms identification can depend markedly on the choice of atomic radii, especially with methods like :class:`~pytim.gitim.GITIM`.
We review how `pytim` chooses which values to use, and how this can be overridden by the user.

Example
-------

**The problem**

Configuration file formats such as the gromos one do not bear
information neither about the atom types nor about the atomic radii. 
In this case :py:obj:`~Pytim` (starting from v 0.6.1) chooses the corresponding radius using the
closest lexycographic match (with higher weight on the first letter)
to the Gromos 43A1 forcefield types. Alternatively, it is possible 
to choose parameters from other forcefields provided with :py:obj:`Pytim`
(Amber 03 and Charmm 27, as of v 0.6.1), or to provide a custom one.
We make here several examples using a methanol box:


>>> import MDAnalysis as mda
>>> import numpy as np
>>> import pytim
>>> from pytim.datafiles import *
>>> u = mda.Universe(METHANOL_GRO)
...

You can safely ignore any warning about missing atomtypes / masses in
:py:obj:`MDAnalysis`, and deal with atomtypes only through the :py:obj:`Pytim` interface.


**Case 1: built-in forcefields**

As there is no specific information in the gromos input file regarding atomic types,  these are 
assigned by :py:obj:`MDAnalysis` using the first letter of the atom name.

>>> print u.residues[0].atoms.types
['M' 'O' 'H']

>>> print u.residues[0].atoms.names
['Me1' 'O2' 'H3']

When the interface is initialized, :py:obj:`Pytim` searches by default for the best 
match from the gromos 43A1 forcefield. It is possible to see explicitly the
choices by passing the option `warnings=True`

>>> interface = pytim.ITIM(u, warnings=True)
guessed radii:  {'H3': 0.0, 'Me1': 1.8230510193786977, 'O2': 1.312927017714457} You can override this by using, e.g.: pytim.ITIM (u,radii_dict={ 'H3':1.2 , ... } )

The Gromos 43A1, Amber 03 and Charmm 27 forcefields are accessible throught labels provided in  the :py:mod:`~pytim.datafiles` module. The radii can be extracted
using the :py:func:`pytim_data.vdwradii()`  function 

>>> gromos  = pytim_data.vdwradii(G43A1_TOP)

>>> amber   = pytim_data.vdwradii(AMBER03_TOP)

>>> charmmm = pytim_data.vdwradii(CHARMM27_TOP)

and can be passed to :py:obj:`Pytim` through the `radii_dict` option, for example:

>>> interface = pytim.ITIM(u, radii_dict = amber )


**Case 2: custom dictionary of radii**

The custom dictionary should provide the atomtypes as keys 
and the radii as values. This database will override the standard one (new behavior in 0.6.1):

>>> u = mda.Universe(METHANOL_GRO)
>>> mydict    = {'Me':1.6, 'O':1.5, 'H':0.0}
>>> interface = pytim.ITIM(u,radii_dict=mydict)
>>> print u.residues[0].atoms.radii
[ 1.6  1.5  0. ]

**Case 3: overriding the default match**

The values of the van der Waals radii present in a topology can be read
using the `pytim_data.vdwradii()`  function:

>>> u = mda.Universe(METHANOL_GRO)
>>> # Load the radii database from the G43A1 forcefield
>>> gromos = pytim_data.vdwradii(G43A1_TOP)
>>> # check which types are available
>>> print gromos.keys()
['CU1+', 'CDmso', 'OWT3', 'CA2+', 'FE', 'BR', 'MG2+', 'HC', 'OWT4', 'CL-', 'NL', 'CCl4', 'CLCl4', 'CL', 'NE', 'SDmso', 'CH1', 'NZ', 'CH3', 'CH4', 'ODmso', 'NR', 'NT', 'C', 'HChl', 'DUM', 'F', 'H', 'CChl', 'CMet', 'CLChl', 'O', 'N', 'MW', 'P', 'S', 'AR', 'IW', 'CU2+', 'OM', 'CR1', 'MNH3', 'NA+', 'OA', 'SI', 'CH2', 'OW', 'ZN2+', 'OMet']

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
