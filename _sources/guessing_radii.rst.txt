Topologies without atomi radii
******************************

Summary
-------
if one starts from a configuration/trajectory for which
the atom types/radii are not known, there are two main ways of
feeding the relevant information to the interface constructor.

Example
-------

**The problem**
Configuration file formats such as the gromos one do not bear
*atomtype* nor *atomic radius* information. In this case  Pytim
chooses the corresponding radius using the closest lexycographic
match to MDAnalysis types, `['C', 'F', 'H', 'O', 'N', 'P', 'S']`
or their average value otherwise. This is usually a good enough
compromise, but it is possible to have full control over this. Let's
start with a "problematic" configuration file:

>>> import MDAnalysis as mda
>>> import numpy as np
>>> import pytim
>>> from pytim.datafiles import *

By initializing the universe, MDAnalysis will immediately warn you
that it failed at guessing the mass atom type 'M'

>>> u = mda.Universe(METHANOL_GRO)
...

You can safely ignore any warning about missing atomtypes in
MDAnalysis, and deal with atomtypes only through the Pytim interface.

As there is no specific information in the topology,  the atom types
assigned by MDAnalysis are the first letter in the atom name:

>>> print u.residues[0].atoms.types
['M' 'O' 'H']

By initializing the interface, Pytim will associate standard radii
to the matching types, and an average radius to the non-matching
ones, issuing no warning, unless specified as shown here:

>>> interface = pytim.ITIM(u,warnings=True)
guessed radii:  {'H': 0.4, 'M': 1.207, 'O': 1.05} You can override this by using, e.g.: pytim.ITIM (u,radii_dic={ 'H':1.2 , ... } )

In this case no reasonabe match was found for atomtype 'M', and the
average value of all radii in the database was associated to it.

If we want to control which radius is being used, we have mainly
two options:

**Case 1**
Create an ad-hoc database of radii and supply it to Pytim.

This is done by creating a dictionary with the atomtypes as keys 
and radii as values. This database will be used to update the standard
one, which will be therefore extended with the new values. Radii for 
existing atomtypes will be overwritten.

>>> u = mda.Universe(METHANOL_GRO)
>>> mydict    = {'M':1.6, 'O':1.5, 'H':0.0}
>>> interface = pytim.ITIM(u,radii_dict=mydict)
>>> print u.residues[0].atoms.radii
[ 1.6  1.5  0. ]

**Case 2**
Load values from an existing forcefield.

The values of the van der Waals radii present in a topology can be read
using the `pytim_data.vdwradii()`  function:

>>> u = mda.Universe(METHANOL_GRO)
>>> # Load the radii database from the G43A1 forcefield
>>> gromos = pytim_data.vdwradii(G43A1_TOP)
>>> # check which types are available
>>> print gromos.keys()
['CU1+', 'CDmso', 'OWT3', 'CA2+', 'FE', 'BR', 'MG2+', 'HC', 'OWT4', 'CL-', 'NL', 'CCl4', 'CLCl4', 'CL', 'NE', 'SDmso', 'CH1', 'NZ', 'CH3', 'CH4', 'ODmso', 'NR', 'NT', 'C', 'HChl', 'DUM', 'F', 'H', 'CChl', 'CMet', 'CLChl', 'O', 'N', 'MW', 'P', 'S', 'AR', 'IW', 'CU2+', 'OM', 'CR1', 'MNH3', 'NA+', 'OA', 'SI', 'CH2', 'OW', 'ZN2+', 'OMet']

Often, the atomtypes strings do not correspond to the one used in the 
coordinate file or in the trajectory. To fix this one can just update
the database 

>>> gromos['M']=gromos['CMet']
>>> gromos['O']=gromos['OMet']
>>> interface = pytim.ITIM(u,radii_dict=gromos)


.. toctree::

.. raw:: html
   :file: analytics.html
