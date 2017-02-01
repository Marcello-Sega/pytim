from abc import ABCMeta, abstractmethod
import numpy as np
from MDAnalysis.analysis import rdf
from MDAnalysis.lib import distances
from MDAnalysis.core.AtomGroup import *
# we try here to have no options passed
# to the observables, so that classes are
# not becoming behemoths that do everything.
# Simple preprocessing (filtering,...)
# should be done by the user beforehand,
# or implemented in specific classes.

#TODO: error handling is horrible
#TODO: comment each function

class Observable(object):
    """ Instantiate an observable. This is derived from the MDAnalysis Observable class
    """
    __metaclass__ = ABCMeta
     
    def __init__(self,universe,options=''):
        self.u=universe
        self.options=options

    #TODO: add proper whole-molecule reconstruction
    def fold_atom_around_first_atom_in_residue(self,atom):
        # let's thake the first atom in the residue as the origin
        box=self.u.trajectory.ts.dimensions[0:3]
        pos = atom.position - atom.residue[0].position
        pos[pos> box/2.]-=box[pos> box/2.] 
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
            
class Orientation(Observable):
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

class MolecularOrientation(Observable):
    def compute(self,inp):
        t = type(inp)
        # TODO: checks for other types?
        if t is AtomGroup and len(inp) != 3*len(inp.residues):
            #TODO: we take automatically the first three if more than three are supplied?
            inp=inp.residues
        pos = self.fold_around_first_atom_in_residue(inp)
        return Orientation(self.u,self.options).compute(pos)


class InterRDF(rdf.InterRDF):
    def __init__(self, g1, g2,
                 nbins=75, range=(0.0, 15.0), exclusion_block=None,
                 start=None, stop=None, step=None,excluded_dir='z',
                 function=None,function2=None,weights=None,obs=None):
        rdf.InterRDF.__init__(self, g1, g2, nbins=nbins, range=range,
                              exclusion_block=exclusion_block,
                              start=start, stop=stop, step=step)
        self.nsamples=0
        self.function=function
        self.function2=function2
        self.weights=weights
        self.obs=obs
     
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
        self._conclude()
            # undo the normalization in InterRDF._conclude()
        if self.nsamples>0:
                self.rdf *= self.nframes**2 / self.nsamples**2 ;


class InterRDF2D(InterRDF):
    def __init__(self, g1, g2,
                 nbins=75, range=(0.0, 15.0), exclusion_block=None,
                 start=None, stop=None, step=None,excluded_dir='z',
                 true2D=False, function=None,obs=None):
        InterRDF.__init__(self, g1, g2,nbins=nbins, range=range,
                          exclusion_block=exclusion_block,
                          start=start, stop=stop, step=step,
                          function=function,obs=obs)
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
