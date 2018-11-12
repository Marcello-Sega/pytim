[What is Pytim](#what-is-pytim) | [Examples](#example) | [More info](#more-info)  | [How to Install](#installation) | [References](#references)


[![Build Status](https://travis-ci.org/Marcello-Sega/pytim.svg?branch=master)](https://travis-ci.org/Marcello-Sega/pytim)
[![GitHub tags](https://img.shields.io/github/tag/Marcello-Sega/pytim.svg)](https://github.com/Marcello-Sega/pytim/)
[![GitHub issues](https://img.shields.io/github/issues/Marcello-Sega/pytim.svg)](https://github.com/Marcello-Sega/pytim/issues)
[![codecov](https://codecov.io/gh/Marcello-Sega/pytim/branch/master/graph/badge.svg)](https://codecov.io/gh/Marcello-Sega/pytim)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5f3f2e7be75b46c1a6e4a3d44e3bb900)](https://www.codacy.com/app/Marcello-Sega/pytim?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Marcello-Sega/pytim&amp;utm_campaign=Badge_Grade)
[![Code Climate](https://codeclimate.com/github/Marcello-Sega/pytim/badges/gpa.svg)](https://codeclimate.com/github/Marcello-Sega/pytim)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<sub>**Disclaimer**: Pytim is in **beta-stage** right now and while a systematic testing system has been set up, this has not yet total coverage. We will keep rolling out more examples and, still, some new features. If you try this software out and have some suggestions, remarks, or bugfixes, feel free to comment here on GitHub and/or make a pull request. </sub>

 <sub>**News**: The paper about pytim has been published on J. Comput. Chem. It is open access and you can [download the pdf](https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.25384) from Wiley <img  src="https://licensebuttons.net/l/by-nc/4.0/80x15.png"> (see also the [references](#refs))

# What is Pytim

[Pytim](https://marcello-sega.github.io/pytim/) is a cross-platform python implementation of several methods for the detection of fluid interfaces in molecular simulations. It is based on [`MDAnalysis`](https://www.mdanalysis.org/), but it integrates also seamlessly with `MDTraj`, and can be even used for *online* analysis during an `OpenMM` simulation (see further down for examples [with `MDTraj`](#mdtraj-example) and [with `OpenMM`](#openmm-example)).

So far the following interface/phase identification methods have been implemented:
<img src="https://github.com/Marcello-Sega/pytim/raw/IMAGES/_images/micelle_cut.png" width="380" align="right" style="z-index:999;">


* ITIM
* GITIM 
* SASA
* Willard Chandler
* Chacon Tarazona
* DBSCAN filtering

## Supported formats
Pytim relies on the [MDAnalysis](http://www.mdanalysis.org/)
package for reading/writing trajectories, and work therefore seamlessly for a number of popular trajectory formats, including:  
* GROMACS
* CHARMM/NAMD
* LAMMPS
* AMBER
* DL_Poly

as well as common structure file formats such as XYZ or PDB (have a look at the [complete list](https://pythonhosted.org/MDAnalysis/documentation_pages/coordinates/init.html#id1))

## Install from PyPi or Anaconda - [![PyPI version](https://badge.fury.io/py/pytim.svg)](https://badge.fury.io/py/pytim) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/pytim/badges/version.svg)](https://anaconda.org/conda-forge/pytim)

PyPi:     ``` pip install --user --upgrade pytim ```

Anaconda: ``` conda install -c conda-forge pytim ```



# <a name="example"></a> Show me an example usage, now!

Ok, ok ... have a look below: this example is about computing molecular layers of a flat interface:

### Step 1: interface identification


```python
import MDAnalysis as mda
import pytim
from pytim.datafiles import WATER_GRO

# load the configuration in MDAnalysis
u = mda.Universe(WATER_GRO)

# compute the interface using ITIM. Identify 4 layers.
inter = pytim.ITIM(u,max_layers=4)
```

### That's it. There's no step 2!

Now interfacial atoms are accessible in different ways, pick the one you like:

1. Through the `atoms` group accessible as

```python
inter.atoms.positions # this is a numpy array holding the position of atoms in the layers
```

    array([[ 22.10000038,  16.05999947,  94.19633484],
           [ 22.43999863,  16.97999954,  93.96632385],
           ...,
           [ 33.70999908,  49.02999878,  62.52632904],
           [ 34.06999969,  48.18000031,  61.16632843]], dtype=float32)

2. Using the label that each atom in the `MDAnalysis` universe now has, which specifies in which layer it is found: 

```python
u.atoms.layers  # -1 if not in any layer, 1 if in the first layer, ...
```

3. Using the layers groups, stored as a list (of lists, in case of upper/lower layers in flat interfaces) of groups: 

```pythin
inter.layers

array([[<AtomGroup with 780 atoms>, <AtomGroup with 690 atoms>,
        <AtomGroup with 693 atoms>, <AtomGroup with 660 atoms>],
       [<AtomGroup with 777 atoms>, <AtomGroup with 687 atoms>,
        <AtomGroup with 663 atoms>, <AtomGroup with 654 atoms>]], dtype=object)
```

## Visualisation

Pytim can export in different file formats: the `PDB` format with the `beta` field used to tag the layers, `VTK`, `cube` for both continuous surfaces and particles, and, of course, all formats supported by `MDAnalysis`. 

In [`VMD`](www.ks.uiuc.edu/Research/vmd/), for example, using `beta == 1` allows you to select all atoms in the first interfacial layer. Just save your `PDB` file with layers information using

```python
inter.writepdb('myfile.pdb')
```


In a `jupyter` notebook, you can use `nglview` like this:


```python
import nglview
v = nglview.show_mdanalysis(u)
v.camera='orthographic'
v.center()
system = v.component_0
colors = ['','red','orange','yellow','white']

for n in [1,2,3,4]:
    system.add_spacefill(selection = inter.atoms[inter.atoms.layers==n].indices, color=colors[n] )

system.add_spacefill(selection = (u.atoms - inter.atoms).indices, color='gray' )
```
```python
# this must go in a separate cell
v.display()
```
<p align="center">

<img src="https://github.com/Marcello-Sega/pytim/raw/IMAGES/_images/output_13_0.png" width="60%" align="center" style="z-index:999;">
</p>

## Analysing trajectories (`MDAnalysis` and `mdtraj`)

Once one of the pytim classes (`ITIM`,`GITIM`,`WillardChandler`,...) has been initialised, whenever a new frame is loaded, the interfacial properties will be calculated automatically without need of doing anything else

### MDAnalysis example

```python
import MDAnalysis as mda
import pytim 
from pytim.datafiles import WATER_GRO, WATER_XTC 

u = mda.Universe(WATER_GRO,WATER_XTC)
inter = pytim.ITIM(u)
for step in u.trajectory[:]:
    print "surface atoms:", repr(inter.atoms)
```

### mdtraj example

Under the hood, `pytim` uses `MDAnalysis`, but this is made (almost completely) transparent to the user, so that interoperability with other software is easy to implement. For example, to analyse a trajectory loaded with [`MDTraj`](https://mdtraj.org), it is enough to do the following:

```python
import mdtraj
import pytim                     
from pytim.datafiles import WATER_GRO, WATER_XTC 

t = mdtraj.load_xtc(WATER_XTC,top=WATER_GRO) 
inter = pytim.ITIM(t) 
for step in t[:]:
        print "surface atoms:" , repr(inter.atoms.indices)
```

### openmm example

Another example is using `pytim` to perform *online* interfacial analysis during an [`OpenMM`](https://openmm.org/) simulation:
```python
# openmm imports
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
# pytim
import pytim
from pytim.datafiles import WATER_PDB

# usual openmm setup, we load one of pytim's example files
pdb = PDBFile(WATER_PDB)
forcefield = ForceField('amber99sb.xml', 'spce.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

# just pass the openmm Simulation object to pytim
inter = pytim.ITIM(simulation)
print repr(inter.atoms)

# the new interfacial atoms will be computed at the end
# of the integration cycle
simulation.step(10)
print repr(inter.atoms)

```



## <a name="non-flat-interfaces"></a> What if the interface is not flat? 

You could use GITIM, or the Willard-Chandler method. 
Let's have a look first at **GITIM**:

```python
import MDAnalysis as mda
import pytim
from   pytim.datafiles import *

u       = mda.Universe(MICELLE_PDB)
g       = u.select_atoms('resname DPC')

interface = pytim.GITIM(u,group=g,molecular=False,
                        symmetry='spherical',alpha=2.5)
layer = interface.layers[0]
interface.writepdb('gitim.pdb',centered=False)
print repr(layer)
<AtomGroup with 872 atoms>
```

`nglview` can be used to show a section cut of the micelle in a `jupyter` notebook:

```python
import nglview
import numpy as np

v = nglview.show_mdanalysis(u)
v.camera='orthographic'
v.center()
v.add_unitcell()
system = v.component_0
system.clear()

selection = g.atoms.positions[:,2]>30.
system.add_spacefill(selection =g.atoms.indices[selection])

selection = np.logical_and(inter.atoms.layers==1,inter.atoms.positions[:,2]>30.)
system.add_spacefill(selection =inter.atoms.indices[selection], color='yellow' )
```

```python
v.display()
```

<p align="center">

<img src="https://github.com/Marcello-Sega/pytim/raw/IMAGES/_images/micelle-gitim.png" width="40%" align="middle">
</p>

The **Willard-Chandler** method can be used, instead to find out isodensity surfaces:

```python
import MDAnalysis as mda
import pytim
from pytim.datafiles import MICELLE_PDB
import nglview

u = mda.Universe(MICELLE_PDB)
g = u.select_atoms('resname DPC')
```

```python
interface = pytim.WillardChandler(u, group=g, mesh=1.5, alpha=3.0)
interface.writecube('data.cube')
```

Done, the density file is written in `.cube` format, we can now open it with 
programs such as [`Paraview`](https://www.paraview.org/), or visualize it in a
jupyter notebook with `nglview`

```python
view = nglview.show_mdanalysis(u.atoms) # the atoms, this will be component_0 in nglview
view.add_component('data.cube') # the density data, this will be component_1 in nglview

view.clear() # looks like this is needed in order for view._display_image() to work correctly 
# let's center the view on our atoms, and draw them as spheres  
view.component_0.center()
view.component_0.add_spacefill(selection='DPC')

# let's add a transparent, red representation for the isodensity surface at 50% density
view.component_1.add_surface(color='red',isolevelType="value",isolevel=0.5,opacity=0.6) 

# add a nice simulation box
view.add_unitcell()
```

```python
view.display()
```

<p align="center">

<img src="https://github.com/Marcello-Sega/pytim/raw/IMAGES/_images/micelle-willard-chandler.png" width="60%" align="middle">
</p>

### Calculate multiple layers with GITIM: solvation shells of glucose
```python
import MDAnalysis as mda
import pytim
from   pytim.datafiles import GLUCOSE_PDB

u       = mda.Universe(GLUCOSE_PDB)
solvent = u.select_atoms('name OW')
glc     = u.atoms - solvent.residues.atoms

inter = pytim.GITIM(u, group=solvent, molecular=True,
                    max_layers=3, alpha=2)

for i in [0,1,2]:
    print "Layer "+str(i),repr(inter.layers[i])
```
```python
Layer 0 <AtomGroup with 81 atoms>
Layer 1 <AtomGroup with 186 atoms>
Layer 2 <AtomGroup with 240 atoms>
```

```python
import nglview
import numpy as np

v = nglview.show_mdanalysis(u)
v.camera='orthographic'
v.center()
v.add_unitcell()

v.clear()

# glucose
v.add_licorice(selection =glc.atoms.indices,radius=.35)

colors = ['yellow','blue','purple']
# hydration layers
for layer in [0,1,2]:
    selection = np.logical_and(inter.atoms.layers==layer+1 ,inter.atoms.positions[:,2]>9)
    v.add_spacefill(selection =inter.atoms.indices[selection], color=colors[layer] )

```
```python
v.display()
```
<p align="center">

<img src="https://github.com/Marcello-Sega/pytim/raw/IMAGES/_images/glc-gitim.png" width="60%" align="middle">
</p>

When calculating surfaces with `GITIM`, it can happen that several disconnected, closed surfaces are found in a simulation box. To restrict the analysis to the largest, clustered interfacial atoms (also when calculating multiple layers), one can pass the `biggest_cluster_only` option, as in:

```python
inter = pytim.GITIM(u, group=solvent, molecular=True, max_layers=3, alpha=2, 
                    biggest_cluster_only=True, cluster_cut = 3.5)
```
In order for this option to have any effect, a `cluster_cut` value should also be passed.

# More info

Have a look at some jupyter notebooks:

1. [An introduction to Pytim](https://github.com/Marcello-Sega/pytim/blob/master/notebooks/An%20introduction%20to%20Pytim.ipynb) 
2. [The Willard-Chandler method]( https://github.com/Marcello-Sega/pytim/blob/master/notebooks/Willard-Chandler%20and%20Cube%20format.ipynb)

Browse the examples in the online manual:

3. [Pytim Online Manual](https://marcello-sega.github.io/pytim/quick.html)

Check out the Pytim Poster from the 10th Liquid Matter Conference 

4. [Available on ResearchGate](http://dx.doi.org/10.13140/RG.2.2.18613.17126)  DOI:10.13140/RG.2.2.18613.17126


# <a name="installation"></a> How to install the package and the documentation? 

## From the PyPI

this will install the latest release present in the Python Package Index:

```
pip install --user --upgrade pytim
```

## From Github
1. Make sure you have an up-to-date version of cython, numpy, scipy and MDAnalysis:

``` 
pip install --user --upgrade cython numpy scipy MDAnalysis
```

2. Download and install pytim

```
git clone https://github.com/Marcello-Sega/pytim.git
cd pytim
python setup.py install --user
```

## Setting the `PYTHONPATH` variable

If you install with the option `--user` (which you have to do if you don't have administrator rights) you shouldn't forget to tell python where to look for the module by setting the `PYTHONPATH` environment variable. 

Under Linux, you could do, for example:
```
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages
```

Under OS-X, instead, use something like:
```
export PYTHONPATH=$HOME/Library/Python/2.7/lib/python/site-packages
```

You can search for the exact path by issuing the following command:

```
python -c "import site; print(site.USER_SITE)"
```

If this for some reason doesn't work, get some hint using:

```
find $HOME -name site-packages
```


## Trouble installing? 

Some of the most common issues are the following:

**Problem**: The system does not let me write (even using `sudo`) some files

**Solution**: You're most likely running under a recent version of OS-X. Always install packages as user (`pip install <package> --user`

**Problem**: cKDTree complains about the `boxsize` parameter

**Solution**: the version of `scipy` must be >= 0.18


**Problem**: Even though I've upgraded `scipy`, I keep getting problems about `boxsize`

**Solution**: You should tell python where to find packages by setting the variable `$PYTHONPATH` 

**Problem**: some error message mentioning `... file was built for x86_64 which is not the architecture being linked (i386)`

**Solution**: use `export ARCHFLAGS='-arch x86_64'` before installing

**Problem**: I'm getting an annoying message like "UserWarning: Module pytim_dbscan was already imported from [...]"

**Solution**: You've installed pytim, and are launching python within the pytim package directory. Move away from there :)



---------------------------

# References  <img src="https://raw.githubusercontent.com/Marcello-Sega/gitim/ITIM/media/soot1small.png" width="180" align="right" style="z-index:999;"> <a name="refs">

The [pytim paper is avaliable](https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.25384)  (under the terms of the Creative Commons BY-NC 4.0 licence) from Wiley. Please cite it if you use pytim for your research:

[M. Sega, S. G. Hantal, B. Fabian and P. Jedlovszky, _J. Comp. Chem._ **39**, 2118-2125 (2018)](http://dx.doi.org/10.1002/jcc.25384) Pytim: A python package for the interfacial analysis of molecular simulations


```
@article{sega2018pytim,
  title={Pytim: A python package for the interfacial analysis of molecular simulations},
  author={Sega, M. and Hantal, G. and F{\'a}bi{\'a}n, B. and Jedlovszky, P.},
  journal={J. Comput. Chem.},
  pages={2118--2125},
  volume={39},
  year={2018},
  publisher={Wiley Online Library}
}
```

Depending on which algorithm you are using, you might also want to cite the following: 

[M. Sega, S. S. Kantorovich, P. Jedlovszky and M. Jorge, _J. Chem. Phys._ **138**, 044110 (2013)](http://dx.doi.org/10.1063/1.4776196) The generalized identification of truly interfacial molecules (ITIM) algorithm for nonplanar interfaces.

[L. B. Pártay, G. Hantal, P. Jedlovszky, Á. Vincze and G. Horvai, _J. Comp. Chem._ **29**, 945 (2008)](http://dx.doi.org/10.1002/jcc.20852) A new method for determining the interfacial molecules and characterizing the surface roughness in computer simulations. Application to the liquid–vapor interface of water

[M. Sega and G. Hantal._Phys. Chem. Chem. Phys._ **29**, 18968-18974 (2017)](https://doi.org/10.1039/C7CP02918G) Phase and interface determination in computer simulations of liquid mixtures with high partial miscibility.

[E. Chacón, P. Tarazona, Phys. Rev. Lett. **91**, 166103 (2003)](http://dx.doi.org/10.1103/PhysRevLett.91.166103) Intrinsic profiles beyond the capillary wave theory: A Monte Carlo study.

[A. P. Willard, D. Chandler, J. Phys. Chem. B **114**,1954 (2010)](http://dx.doi.org/10.1021/jp909219k) Instantaneous Liquid Interfaces

