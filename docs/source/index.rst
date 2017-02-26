.. Pytim documentation master file, created by
   sphinx-quickstart on Wed Feb  1 00:55:02 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Pytim: a Quick Tour
*******************

A simple example that shows how to use the :class:`ITIM` class
for planar interfaces is the following:

.. code-block:: python

    import MDAnalysis as mda                 
    import pytim 
    from pytim.datafiles import *

    u         = mda.Universe(WATER_GRO)
    oxygens   = u.select_atoms("name OW") 

    interface = pytim.ITIM(u,itim_group=oxygens)

    interface.assign_layers()
    interface.writepdb()

This code block first imports MDAanalysis, pytim, and useful
datafiles, then initializes the universe in the usual way, using
as an input file one of the structures provided by the package, in
this case a water/vapor interface in a gromos file format (the variable
WATER_GRO is made available by the pytim.datafiles module, and points 
to 'data/water.gro')

The oxygens are then chosen as a reference atom group to perform
the interfacial analysis, and the interface calculation is initialized
using the :class:`ITIM` class.

The last two lines are used to calculate the interfacial atoms, and to 
save the whole configuration (with surface atoms having a beta
factor equal to the numer of the layer they belong to)

The result of the calculation can be seen in the following picture,
where surface oxygen atoms are highlighted in blue.


.. image:: example_water.png
   :scale: 70%
   :align: center


Table of Contents Tree
**********************

.. toctree::
    ITIM
    GITIM
    observables
    utilities
    testsuite


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. raw:: html
   :file: analytics.html

