import MDAnalysis as mda
import pytim
from pytim import  *
from pytim.datafiles import *
u         = mda.Universe(WATER_GRO)
u=mda.Universe(_TEST_PROFILE_GRO)
print u.atoms.positions
o=observables.Number()
print u.dimensions[:3]
p=observables.Profile(direction='x',group=u.atoms,observable=o)
p.sample()
low,up,avg =  p.get_values(binwidth=1.0)
print low
print avg
print p.do_rebox
