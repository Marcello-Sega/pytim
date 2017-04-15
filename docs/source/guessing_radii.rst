Topologies without atomi radii
******************************

Summary
-------
if one starts from a configuration/trajectory for which
the atom types are not known, there are two main ways of
feeding the relevant information to the interface constructor.

Example
-------

**The problem**
    Configuration file formats such as the gromos one do not bear *atomtype* nor *atomic radius* information.
    If the atom names in the configuration file start with one of the letters `['C', 'F', 'H', 'O', 'N', 'P', 'S']`, Pytim chooses the corresponding
    radius using the MDAnalysis values, and their average otherwise. This is usually a good enough compromise, but it is possible to have full control
    over this. Let's start with a "problematic" configuration file:

    >>> import MDAnalysis as mda
    >>> import pytim
    >>> from pytim.datafiles import *
    >>> u = mda.Universe(METHANOL_GRO)
    ...

    the atom types assigned by MDAnalysis are the first letter in the
    atom name:

    >>> print u.residues[0].atoms.types
    ['M' 'O' 'H']

    By initializing the interface, Pytim will associate standard radii to the matching types, and an average radius to the non-matching ones.

    >>> u = mda.Universe(METHANOL_GRO)
    ...

    >>> interface = pytim.ITIM(u,cluster_cut=5.0)
    !!                              WARNING
    !! No appropriate radius was found for the atomtype M
    !! Using the average radius (1.20714285714) as a fallback option...
    !! Pass a dictionary of radii (in Angstrom) with the option radii_dict
    !! for example: r={'M':1.2,...} ; inter=pytim.ITIM(u,radii_dict=r)


    >>> print u.residues[0].atoms.radii
    [ 1.20714286  1.05        0.4       ]

 If this is not the wanted behavior, one has usually the following two options:

**Case 1**
    We know exactly what the atomic radii should be.

**Solution**
    build a dictionary of radii corresponding to the first letter
    of the atom name

    >>> u = mda.Universe(METHANOL_GRO)
    >>> mydict    = {'M':1.6, 'O':1.5, 'H':0.0}
    >>> interface = pytim.ITIM(u,cluster_cut=5.0,radii_dict=mydict)
    >>> print u.residues[0].atoms.radii
    [ 1.6  1.5  0. ]

**Case 2**
    We don't know exactly the atomic radii, but we don't want to use just the standard values.
**Solution**
    Use an existing topology, and overwrite the types guessed by MDAnalysis

    >>> u = mda.Universe(METHANOL_GRO)
    >>> # Load the radii
    >>> radii_gromos = pytim_data.vdwradii(G43A1_TOP)
    >>> # check which types are available
    >>> print radii_gromos.keys()
    ['CU1+', 'CDmso', 'OWT3', 'CA2+', 'FE', 'BR', 'MG2+', 'HC', 'OWT4', 'CL-', 'NL', 'CCl4', 'CLCl4', 'CL', 'NE', 'SDmso', 'CH1', 'NZ', 'CH3', 'CH4', 'ODmso', 'NR', 'NT', 'C', 'HChl', 'DUM', 'F', 'H', 'CChl', 'CMet', 'CLChl', 'O', 'N', 'MW', 'P', 'S', 'AR', 'IW', 'CU2+', 'OM', 'CR1', 'MNH3', 'NA+', 'OA', 'SI', 'CH2', 'OW', 'ZN2+', 'OMet']
    >>> # Overwrite the types
    >>> u.atoms.types[u.atoms.types=='M']='CMet'
    >>> u.atoms.types[u.atoms.types=='O']='OMet'
    >>> # compute the interface
    >>> interface = pytim.ITIM(u,cluster_cut=5.0,radii_dict=radii_gromos)



More in detail
--------------


If we do not specify any external dictionary for the radii,
the default one is that of MDAnalysis, which can be accessed by
pytim.tables.vdwradii:

>>> print pytim.tables.vdwradii.keys()
['C', 'F', 'H', 'O', 'N', 'P', 'S']

Our atom names are

>>> print u.residues[0].atoms.names
['Me1' 'O2' 'H3']

>>> print u.residues[0].atoms.types
['M' 'O' 'H']

When building the interface, Pytim will try to guess atom types and radii. Since `M`
does not match any atom type in `tables.vdwradii`, the following warning is issued:

>>> u = mda.Universe(METHANOL_GRO)
>>> interface = pytim.ITIM(u,cluster_cut=5.0)
!!                              WARNING
!! No appropriate radius was found for the atomtype M
!! Using the average radius (1.20714285714) as a fallback option...
!! Pass a dictionary of radii (in Angstrom) with the option radii_dict
!! for example: r={'M':1.2,...} ; inter=pytim.ITIM(u,radii_dict=r)


The default dictionary `tables.vdwradii` can be extended directly, for example:

>>> pytim.tables.vdwradii['M']=1.6

now the warning is not issued anymore:
>>> interface = pytim.ITIM(u,cluster_cut=5.0)

Of course, one can provide all the radii explicitly:

>>> u = mda.Universe(METHANOL_GRO)
>>> radii = dict({'O':1.5,'M':1.6,'H':0.0})
>>> interface = pytim.ITIM(u,cluster_cut=5.0,radii_dict=radii)

... or use one of the provided forcefields:

>>> print pytim_data.topol
['G43A1_TOP']

We can extract the radii for the GROMOS 43A1 topology , like this:

>>> u = mda.Universe(METHANOL_GRO)
>>> radii_gromos=pytim_data.vdwradii(G43A1_TOP)
>>> interface = pytim.ITIM(u,cluster_cut=5.0,radii_dict=radii_gromos)

The atom types supplied are:

>>> print radii_gromos.keys()
['CU1+', 'CDmso', 'OWT3', 'CA2+', 'FE', 'BR', 'MG2+', 'HC', 'OWT4', 'CL-', 'NL', 'CCl4', 'CLCl4', 'CL', 'NE', 'SDmso', 'CH1', 'NZ', 'CH3', 'CH4', 'ODmso', 'NR', 'NT', 'C', 'HChl', 'DUM', 'F', 'H', 'CChl', 'CMet', 'CLChl', 'O', 'N', 'MW', 'P', 'S', 'AR', 'IW', 'CU2+', 'OM', 'CR1', 'MNH3', 'NA+', 'OA', 'SI', 'CH2', 'OW', 'ZN2+', 'OMet']


these do not correspond exactly to ours, unless we
of the corresponding atom types.

**However**, the automatic selection will anyway make a choice:

>>> print u.residues[0].atoms.radii
[ 1.31292702  1.31292702  0.        ]

If one wonder to which atom types this set of radii corresponds, it is possible to figure it out:

>>> reverse = dict((v, k) for k, v in radii_gromos.iteritems())
>>> print [reverse[r] for r in u.residues[0].atoms.radii.tolist()]
['OM', 'OM', 'MNH3']

Not really what we wanted. So, once one has a topology at hand, it is best to map the atomtypes
to those of the topology, like

>>> myradii=dict()
>>> myradii['M']=radii_gromos['CMet']
>>> myradii['O']=radii_gromos['OMet']
>>> myradii['H']=radii_gromos['H']

Let's see what we've got:

>>> print myradii
{'H': 0.0, 'M': 1.8230510193786977, 'O': 1.4774209166212664}

Better. Now we can feed them to the interface constructor:

>>> u = mda.Universe(METHANOL_GRO)
>>> interface = pytim.ITIM(u,cluster_cut=5.0,radii_dict=myradii)

An equivalent result can be obtained faster, by overwriting the types:

>>> u.atoms.types[u.atoms.types=='M']='CMet'
>>> u.atoms.types[u.atoms.types=='O']='OMet'
>>> u = mda.Universe(METHANOL_GRO)
>>> interface = pytim.ITIM(u,cluster_cut=5.0,radii_dict=radii_gromos)
>>> print interface.layers[:]
[[<AtomGroup with 555 atoms>]
 [<AtomGroup with 564 atoms>]]


.. toctree::

.. raw:: html
   :file: analytics.html
