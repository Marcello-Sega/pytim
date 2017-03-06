To center or not to center?
***************************

Summary
-------
Pytim provides different ways to center the interface in the simulation box, and how to compute profiles. 

This short tutorial shows the different behaviors.

Example: pdb output for a planar interface
------------------------------------------


**Initial file**

The waterbox example file :file:`WATER_GRO` consists of 4000 water molecules, with all 
atoms in the unit cell, and the liquid phase crossing the periodic boundary conditions:

.. image:: centering1.png
   :width: 35%
   :align: center


If you want to save a local copy of the file, you just need to use this code fragment:

.. code-block:: python

    import MDAnalysis as mda                 
    import pytim 
    u         = mda.Universe(pytim.datafiles.WATER_GRO)
    u.atoms.write('centering1.pdb')

**Default, centering at the origin**

When initializing/computing the interface, Pytim internally centers the interface in different ways, depending whether :class:`~pytim.itim.ITIM` or :class:`~pytim.gitim.GITIM` is used. For planar interfaces, by default the middle of the liquid phase is moved to the origin along the surface normal, while the positions along other directions are placed in the box. In our case, the result is:

.. code-block:: python

    interface = pytim.ITIM(u)
    interface.writepdb('centering2.pdb')


.. image:: centering2.png
   :width: 35%
   :align: center

This comes handy to quickly discriminate between upper and lower surface atoms, to spot immediately the middle of the liquid phase, or to verify that the automatic centering method behaved as expected.

**No centering**

If the option `centered='no'` is passed to :meth:`~pytim.itim.ITIM..writepdb`, then the first atom of the system is kept at its original position (i.e., no shift is applied) although the system is always put into the basic cell. 

.. code-block:: python

    interface.writepdb('centering3.pdb',centered='no')


This is like the initial case, however with information on the surface molecules added to the pdb.

.. image:: centering3.png
   :width: 35%
   :align: center

**Centering in the middle**

If the option `centered='middle'` is passed  to :meth:`~pytim.itim.ITIM.writepdb`, instead, then the liquid part of the system is placed in the middle of the box along the surface normal:

.. code-block:: python

    interface.writepdb('centering4.pdb',centered='middle')

.. image:: centering4.png
   :width: 35%
   :align: center



....


.. toctree::

.. raw:: html
   :file: analytics.html

