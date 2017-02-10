# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
""" Module: observables
    ===================
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import stats
from MDAnalysis.analysis import rdf
from MDAnalysis.lib import distances
from MDAnalysis.core.AtomGroup import *
from itertools import chain
import pytim
import utilities
# we try here to have no options passed
# to the observables, so that classes are
# not becoming behemoths that do everything.
# Simple preprocessing (filtering,...)
# should be done by the user beforehand,
# or implemented in specific classes.

#TODO: error handling is horrible
#TODO: comment each function

class AnalysisBase(object):
    """ Instantiate an observable. This is derived from the MDAnalysis AnalysisBase class

    """
    __metaclass__ = ABCMeta
     
    def __init__(self,universe,options='',):
        self.u=universe
        self.options=options

    #TODO: add proper whole-molecule reconstruction
    def fold_atom_around_first_atom_in_residue(self,atom):
        # let's thake the first atom in the residue as the origin
        box=self.u.trajectory.ts.dimensions[0:3]
        pos = atom.position - atom.residue[0].position
        pos[pos>= box/2.]-=box[pos>= box/2.] 
        pos[pos<-box/2.]+=box[pos<-box/2.] 
        return pos
     
    def fold_around_first_atom_in_residue(self,inp):
        pos=[]
        t = type(inp)
          
        if t is Atom:
                pos.append(self.fold_atom_around_first_atom_in_residue(inp))
        elif t is AtomGroup or \
                t is Residue or \
                t is ResidueGroup:
                for atom in inp.atoms:
                    pos.append(self.fold_atom_around_first_atom_in_residue(atom))
        else:
            print "input not valid for fold_around_first_atom_in_residue():",t,type(self.u.trajectory.ts),exit()
        return np.array(pos)

    @abstractmethod
    def compute(self,input):
        print "Not implemented"


class Number(AnalysisBase):
    def compute(self,inp):
        return np.ones(len(inp))


            
class Orientation(AnalysisBase):
    def compute(self,pos):

        flat = pos.flatten()
        pos  = flat.reshape(len(flat)/3,3)
        a = pos[1::3] - pos[0::3]   
        b = pos[2::3] - pos[0::3]   
        # TODO: can this be vectorized? 
        if 'normal' in self.options:
            v = np.cross(a,b)
        else:
            v = np.array(a+b)
        v =  np.array([el/np.sqrt(np.dot(el,el)) for el in v])
        return v

class MolecularOrientation(AnalysisBase):
    def compute(self,inp):
        t = type(inp)
        # TODO: checks for other types?
        if t is AtomGroup and len(inp) != 3*len(inp.residues):
            #TODO: we take automatically the first three if more than three are supplied?
            inp=inp.residues
        pos = self.fold_around_first_atom_in_residue(inp)
        return Orientation(self.u,self.options).compute(pos)



class Profile(object):
    def __init__(self,universe,group,direction='z',observable=None,interface=None,center_group=None):
        _dir = {'x':0,'y':1,'z':2}
        self.universe      = universe
        self.group         = group
        self.center_group  = center_group
        if observable is None:
            self.observable = Number()
        self.observable    = observable
        self._dir          =_dir[direction]
        self.binsize       = 0.1 # this is used for internal calculations, the output binsize can be specified in self.profile()

        self.interface     = interface
        self.sampled_values=[]
        self.sampled_bins  =[]
        self.pos=[utilities.get_x,utilities.get_y,utilities.get_z]

    def sample(self):
        # TODO: implement progressive averaging to handle very long trajs
        # TODO: implement memory cleanup
        self._box=self.universe.dimensions[self._dir]
        if self.interface is None:
            _pos    = self.pos[self._dir](self.group)      
            _values = self.observable(self.group)
            _nbins  = int(self.universe.trajectory.ts.dimensions[self._dir]/self.binsize)
            _avg, _bins, _binnumber = stats.binned_statistic(_pos, _values, 
                                                             range=[-self._box/2.,self._box/2.],
                                                             statistic='mean',bins=_nbins)
            _avg[np.isnan(_avg)]=0.0
            self.sampled_values.append(_avg)
            self.sampled_bins.append(_bins[1:]-self.binsize/2.) # these are the bins midpoints
    
    def profile(self,binwidth=None,nbins=None):
        assert self.sampled_values, "No profile sampled so far."
        # we use the largest box (largest number of bins) as reference. 
        # Statistics will be poor at the boundaries, but like that we don't loose information
        _max_bins  = np.max(map(lambda x: len(x),self.sampled_bins))
        _max_size  = _max_bins * self.binsize
        if(binwidth==None and nbins==None):
            _nbins = _max_bins
        else:
            if binwidth==None:
                _nbins = nbins
            else:
                _nbins = _max_size/binwidth

        # TODO sanity check on binwidth and nbins missing

        _avg,_bins,_binnumber = stats.binned_statistic(list(chain.from_iterable(self.sampled_bins  )),
                                            list(chain.from_iterable(self.sampled_values)),
                                            range=[-self._box/2.,self._box/2.],
                                            statistic='mean',bins=_nbins )
        _avg[np.isnan(_avg)]=0.0
        _binsize = _bins[1]-_bins[0]
        return [  (_bins[1:] - _binsize/2.) , _avg ]



class InterRDF(rdf.InterRDF):
    """ Calculates a radial distribution function of some observable from two groups. 

        The two functions must return an array (of scalars or of vectors)
        having the same size of the group. The scalar product between the
        two functions is used to weight the distriution function.

        :param AtomGroup g1:            1st group
        :param AtomGroup g2:            2nd group
        :param int nbins:               number of bins
        :param ??? exclusion_block:
        :param int start:               first frame
        :param int stop:                last frame
        :param int step:                frame stride
        :param char excluded_dir:       project position vectors onto the plane orthogonal to 'z','y' or 'z' (TODO not used here, check & remove)
        :param array function:          function applied to the atoms in g1
        :param array function2:         function applied to the atoms in g2
        :param array weights:           weights to be applied to the distribution function (mutually exclusive with function/function2)

        .. math::

              g(r) = \\frac{1}{N}\left\langle \sum_{i\\neq j} \delta(r-|r_i-r_j|) f_1(r_i,v_i)\cdot f_2(r_j,v_j) \\right\\rangle

        TODO add a MolecularOrientation example
        
        Example:

        >>> import MDAnalysis as mda
        >>> import numpy as np
        >>> import pytim 
        >>> from pytim.datafiles import *
        >>> from pytim.observables import * 
        >>> 
        >>> u = mda.Universe(WATER_GRO,WATER_XTC)
        >>> L = np.min(u.dimensions[:3])
        >>> oxygens = u.select_atoms("name OW") 
        >>> radii=pytim_data.vdwradii(G43A1_TOP)
        >>> 
        >>> interface = pytim.ITIM(u,alpha=2.,itim_group=oxygens,max_layers=4,multiproc=True,radii_dict=radii,cluster_groups=oxygens,cluster_cut=3.5)
        >>> 
        >>> for ts in u.trajectory[::50] : 
        ...     interface.assign_layers()
        ...     layer=interface.layers('upper',1)	
        ...     if ts.frame==0 :
        ...         rdf=InterRDF2D(layer,layer,range=(0.,L/2.),nbins=120)
        ...     rdf.sample(ts)
        >>> rdf.normalize()
        >>> rdf.rdf[0]=0.0
        >>> np.savetxt('RDF.dat', np.column_stack((rdf.bins,rdf.rdf)))  #doctest:+SKIP


        This results in the following RDF:

        .. plot::

            import MDAnalysis as mda
            import numpy as np
            import pytim 
            from pytim.datafiles import *
            from pytim.observables import * 
            u = mda.Universe(WATER_GRO,WATER_XTC)
            L = np.min(u.dimensions[:3])
            oxygens = u.select_atoms("name OW") 
            radii=pytim_data.vdwradii(G43A1_TOP)
            interface = pytim.ITIM(u,alpha=2.,itim_group=oxygens,max_layers=4,multiproc=True,radii_dict=radii,cluster_groups=oxygens,cluster_cut=3.5)
            for ts in u.trajectory[::5] :
                interface.assign_layers()
                layer=interface.layers('upper',1)	
                if ts.frame==0 :
                    rdf=InterRDF2D(layer,layer,range=(0.,L/2.),nbins=120)
                rdf.sample(ts)
            rdf.normalize()
            rdf.rdf[0]=0.0
            plt.plot(rdf.bins, rdf.rdf)
            plt.show()


        Example: dipole-dipole correlation on the surface (TODO)



    """

    def __init__(self, g1, g2,
                 nbins=75, range=(0.0, 15.0), exclusion_block=None,
                 start=None, stop=None, step=None,excluded_dir='z',
                 function=None,function2=None,weights=None):
        rdf.InterRDF.__init__(self, g1, g2, nbins=nbins, range=range,
                              exclusion_block=exclusion_block,
                              start=start, stop=stop, step=step)
        self.nsamples=0
        self.function=function
        self.function2=function2
        self.weights=weights
     
    def _single_frame(self):
        if (self.function is not None or 
            self.function2 is not None) and \
            self.weights is not None:
            print "Error, cannot specify both a function and weights" 
        if self.function is not None or self.function2 is not None:
            if self.function2 is None:
                self.function2 = self.function

                fg1 = self.function(self.g1)
                fg2 = self.function2(self.g2)
                if len(fg1)!=len(self.g1) or len(fg2)!=len(self.g2):
                    print "Error, the function should output an array (of scalar or vectors) the same size of the group",exit()
                # both are (arrays of) scalars
                if len(fg1.shape)==1 and len(fg2.shape)==1:
                    _weights = np.outer(fg1,fg2)
                # both are (arrays of) vectors
                elif len(fg1.shape)==2 and len(fg2.shape)==2:
                # TODO: tests on the second dimension...
                    _weights = np.dot(fg1,fg2.T)
                else :  
                    print "Erorr, shape of the function output not handled",exit()
                # numpy.histogram accepts negative weights
                self.rdf_settings['weights']=_weights
        if self.weights is not None:
            print "Weights not implemented yet",exit()
        
        #
        rdf.InterRDF._single_frame(self)
     
    def sample(self,ts):
        self._ts=ts
        self._single_frame()
        self.nsamples+=1
     
    def normalize(self):
        self._conclude() # TODO fix variable group size; remove blocks support
            # undo the normalization in InterRDF._conclude()
        if self.nsamples>0:
                self.rdf *= self.nframes**2 / self.nsamples**2 ;


class InterRDF2D(InterRDF):
    def __init__(self, g1, g2,
                 nbins=75, range=(0.0, 15.0), exclusion_block=None,
                 start=None, stop=None, step=None,excluded_dir='z',
                 true2D=False, function=None):
        InterRDF.__init__(self, g1, g2,nbins=nbins, range=range,
                          exclusion_block=exclusion_block,
                          start=start, stop=stop, step=step,
                          function=function)
        self.true2D=true2D
        if excluded_dir is 'z':
                self.excluded_dir=2
        if excluded_dir is 'y':
                self.excluded_dir=1
        if excluded_dir is 'x':
                self.excluded_dir=0
     
    def _single_frame(self):
        excl=self.excluded_dir
        p1=self.g1.positions
        p2=self.g2.positions
        if self.true2D:
                p1[:,excl]=0
                p2[:,excl]=0
        self.g1.positions=p1
        self.g2.positions=p2
        InterRDF._single_frame(self)
        # TODO: works only for rectangular boxes
        # we subtract the volume added for the 3d case,
        # and we add the surface
        self.volume += self._ts.volume*(1./self._ts.dimensions[excl]-1.)
     
    def _conclude(self):
        InterRDF._conclude(self)
        correction = 4./3.*np.pi * (np.power(self.edges[1:], 3) -
                                    np.power(self.edges[:-1], 3))
        correction /= np.pi * (np.power(self.edges[1:], 2) -
                               np.power(self.edges[:-1], 2))
        rdf = self.rdf * correction
        self.rdf = rdf


#
