A Python Tool for Interfacial Molecules Analysis
================================================

Pytim is a cross-platform python implementation of several methods
for the detection of fluid interfaces in molecular simulations: it
allows to identify interfacial molecules from the trajectories of
major molecular dynamics simulation packages, and run some analyses
specifically conceived for interfacial molecules, such as intrinsic
profiles.

So far the following methods have been implemented:

* ITIM
* GITIM 
* SASA
* Willard Chandler
* DBSCAN filtering

----

Pytim relies on the MDAnalysis package for reading/writing trajectories,
and work therefore seamlessly for a number of popular trajectory
formats, including:

* GROMACS
* CHARMM/NAMD
* LAMMPS
* AMBER
* DL_Poly


