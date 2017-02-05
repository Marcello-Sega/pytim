#!/usr/bin/python
# vim: set expandtab:
# vim: set tabstop=4:
""" Module: pytim
    =============
"""

from timeit import default_timer as timer
#from threading import Thread
from multiprocessing import Process, Queue
import numpy as np
from MDAnalysis.core.AtomGroup   import *
from pytim.datafiles import *

class ITIM():
    """ Identifies the interfacial molecules at macroscopically 
        flat interfaces.

        :param Universe universe:      the MDAnalysis universe
        :param float mesh:             the grid spacing used for the testlines
        :param float alpha:            the probe sphere radius
        :param AtomGroup itim_group:   identify the interfacial molecules from this group
        :param dict radii_dict:        dictionary with the atomic radii of the elements in the itim_group. 
                                       If None is supplied, the default one (from GROMOS 43a1) will be used.
        :param int max_layers:         the number of layers to be identified
        :param bool info:              print additional info 
        :param bool multiproc:         parallel version (default: True. Switch off for debugging)
 
        Example:

        >>> import MDAnalysis as mda                 
        >>> import pytim 
        >>> from pytim.datafiles import *
        >>> 
        >>> u         = mda.Universe(WATER_GRO)
        >>> oxygens   = u.select_atoms("name OW") 
        >>>  
        >>> interface = pytim.ITIM(u, alpha=2.0, max_layers=4)
        >>> interface.assign_layers()
        >>> 
        >>> print interface.layers('upper',1)  # first layer, upper
        <AtomGroup with 161 atoms>

        # TODO Add here an example on how to use the variuos other options
    """
 
    def __init__(self,universe,mesh=0.4,alpha=2.0,itim_group=None,radii_dict=None,
                 max_layers=1,info=False,multiproc=True):

        self.universe=universe
        self.target_mesh=mesh
        self.alpha=alpha
        self.max_layers=max_layers
        self.info=info
        self.all_atoms = self.universe.select_atoms('all')

        try:
            if(itim_group==None):
                self.itim_group =  self.all_atoms
            else:
                self.itim_group =  itim_group
            types = np.copy(self.itim_group.types)

            if (radii_dict==None): # let's use the default one
                radii_dict=pytim_data.vdwradii(G43A1_TOP)
            radii = np.zeros(len(self.itim_group.types))
            # let's select unique atom types, and 
            for type in np.unique(types):
                try:
                    radii[types==type]=radii_dict[type]
                except:
                    pass # those types which are not listed in the dictionary 
                         # will have radius==0. TODO handle this

            self.itim_group.radii=radii
            del radii
            del types
        except:
            print ("Error (generic) while initializing ITIM")

        self._sanity_checks()

        self.grid=None
        self.use_threads=False
        self.use_multiproc=multiproc
        self.tic=timer()

    def _sanity_checks(self):
        ALPHA_NEGATIVE = "parameter alpha in %s.%s must be positive" % ( (__name__) , (self.__class__.__name__) )
        ALPHA_LARGE= "parameter alpha in %s.%s must be smaller than the smaller box side" % ( (__name__) , (self.__class__.__name__) )

        MESH_NEGATIVE = "parameter mesh in %s.%s must be positive" % ( (__name__) , (self.__class__.__name__) )
        MESH_LARGE= "parameter mesh in %s.%s must be smaller than the smaller box side" % ( (__name__) , (self.__class__.__name__) )

        # these are done at the beginning to prevent burdening the inner loops
        assert self.alpha > 0,                                           ALPHA_NEGATIVE
        assert self.alpha < np.amin(self.universe.dimensions[:3]),       ALPHA_LARGE

        assert self.target_mesh > 0,                                     MESH_NEGATIVE
        assert self.target_mesh < np.amin(self.universe.dimensions[:3]), MESH_LARGE

        try:
            np.arange(int(self.alpha/self.target_mesh))
        except:
            print("Error while initializing ITIM: alpha (%f) too large or mesh (%f) too small" % self.alpha,self.target_mesh)
            raise ValueError

    def lap(self):
        toc=timer()
        print "LAP >>> ",toc-self.tic
        self.tic=toc

    def _x(self,group=None):
        if (group==None) :
            group = self.all_atoms
        return group.positions[:,0]

    def _y(self,group=None):
        if (group==None) :
            group = self.all_atoms
        return group.positions[:,1]

    def _z(self,group=None):
        if (group==None) :
            group = self.all_atoms
        return group.positions[:,2]

    def _rebox(self,x=None,y=None,z=None):
        dim = self.universe.coord.dimensions
        stack=False
        shift=np.array([0,0,dim[2]/2.])

        # we rebox the atoms in universe, and not a vector
        if x is None and y is None and z is None:
            stack=True
            x=self._x()
            y=self._y()
            z=self._z()
        for index, val in enumerate((x,y,z)):
            try:
                val[val> dim[index]-shift[index]]-=dim[index]
                val[val< 0        -shift[index]]+=dim[index]
            except:
                pass
        if stack:
            self.universe.coord.positions=np.column_stack((x,y,z))

    def writepdb(self,filename='layers.pdb',multiframe=True):
        """ Write the frame to a pdb file, marking the atoms belonging
            to the layers with different beta factor.

	    :param filename:   string  -- the output file name
            :param multiframe: boolean -- append to pdb file if True

            Example:

            >>> interface.writepdb('layers.pdb',multiframe=False)

        """

        self.itim_group.atoms.bfactors=self.bfactors
        try:
            PDB=MDAnalysis.Writer(filename, multiframe=True, bonds=False,
                            n_atoms=self.universe.atoms.n_atoms)
            PDB.write(self.universe.atoms)
        except:
            print("Error writing pdb file")


    def center(self):
        """ 
        Centers the liquid slab in the simulation box.

        The algorithm tries to avoid problems with the definition
        of the center of mass. First, a rough density profile
        (10 bins) is computed. Then, the support group is shifted
        and reboxed until the bins at the box boundaries have a
        density lower than a threshold delta

        
        """
        #TODO: implement shifting back for final coordinate writing as an option
        dim = self.universe.coord.dimensions
        shift=dim[2]/100. ;# TODO, what about non ortho boxes?
        total_shift=0
        self._rebox()

        _z_itim_group = self._z(self.itim_group)
        _x = self._x()
        _y = self._y()
        _z = self._z()

        histo,edges=np.histogram(_z_itim_group, bins=10,
                                 range=(-dim[2]/2.,dim[2]/2.), density=True) ;
            #TODO handle norm!=z
        max=np.amax(histo)
        min=np.amin(histo)
        delta=min+(max-min)/3. ;# TODO test different cases

        # let's first avoid crossing pbc with the liquid phase. This can fail:
        # TODO handle failure
        while(histo[0]>delta or histo[-1]> delta):
            if self.info: print "shifting by ",shift
            total_shift+=shift
            _z_itim_group +=shift
            self._rebox(z=_z_itim_group);
            histo,edges=np.histogram(_z_itim_group, bins=10,
                                     range=(-dim[2]/2.,dim[2]/2.), density=True);
        #TODO: clean up
        center=np.average(_z_itim_group)

        _z += total_shift - center
        # finally, we copy everything back
        self.universe.coord.positions=np.column_stack((_x,_y,_z))


    def _assign_mesh(self):
        """ determine a mesh size for the testlines that is compatible with the simulation box
        """
        self.mesh_nx=int(np.ceil(self.universe.coord.dimensions[0]/
                         self.target_mesh))
        self.mesh_ny=int(np.ceil(self.universe.coord.dimensions[1]/
                         self.target_mesh))
        self.mesh_dx=self.universe.coord.dimensions[0]/self.mesh_nx
        self.mesh_dy=self.universe.coord.dimensions[1]/self.mesh_ny
        self.delta=np.minimum(self.mesh_dx,self.mesh_dy)/10.

    def _touched_lines(self,atom,_x,_y,_z,_radius):
        # TODO: speedup this part of the code.
        _range=self.alpha+self.delta
        index_x = np.array(range(
            int(np.floor((_x[atom]-_radius[atom]-_range)/self.mesh_dx)),
            int(np.ceil ((_x[atom]+_radius[atom]+_range)/self.mesh_dx))
            ))
        index_y = np.array(range(
            int(np.floor((_y[atom]-_radius[atom]-_range)/self.mesh_dy)),
            int(np.ceil ((_y[atom]+_radius[atom]+_range)/self.mesh_dy))
            ))
        distmap = ( (index_x*self.mesh_dx-_x[atom]).reshape(len(index_x),1)**2+
                    (index_y*self.mesh_dy -_y[atom])**2 )
        _xx, _yy  = np.where(distmap<=(self.alpha+_radius[atom])**2)
        # now we need to go back to the real space map. Whenever
        # index_x (or index_y) is < 0 || > box we need to wrap it to
        # the other end of the box.
        sel_x = index_x[_xx]
        sel_y = index_y[_yy]
        sel_x[sel_x<0]+=self.mesh_nx
        sel_y[sel_y<0]+=self.mesh_ny
        sel_x[sel_x>=self.mesh_nx]-=self.mesh_nx
        sel_y[sel_y>=self.mesh_ny]-=self.mesh_ny
        return np.column_stack((sel_x,sel_y))

    def _assign_one_side(self,uplow,sorted_atoms,_x,_y,_z,
                        _radius,queue=None):

        for layer in range(0,self.max_layers) :
            mask = self.mask[uplow][layer]
            inlayer=[]
            count=0
            for atom in sorted_atoms:
                count+=1
                if self._seen[atom] != 0 :
                    continue
                    # TODO: would a KD-Tree be faster for small boxes ?
                    # TODO: document this
                touched_lines  = self._touched_lines(atom,_x,_y,_z,_radius)
                submask_=[]
                for i,j in touched_lines:
                    submask_.append(mask[i,j])
                # 
                submask = np.array(submask_)
                if(len(submask[submask==0])==0):
                    # no new contact, let's move to the next atom
                    continue
                # let's mark now:
                # 1) the touched lines
                for i,j in touched_lines:
                    mask[i,j] = 1
                # 2) the sorted atom
                self._seen[atom]=layer+1 ; # start counting from 1, 0 will be
                                           # unassigned, -1 for gas phase TODO: to be
                                           # implemented
                # 3) let's add the atom id to the list of atoms in this layer
                inlayer.append(atom)
                if len(mask[mask==0])==0: # no more untouched lines left
                    self.layers_ids[uplow].append(inlayer)
                    break
        if queue != None:
            queue.put(self._seen)
            queue.put(self.layers_ids[uplow])

    def _define_layers_groups(self):
        _layers=[[],[]]
        for i,uplow in enumerate(self.layers_ids):
            for j,layer in enumerate(uplow):
                _layers[i].append(self.itim_group.atoms[layer])
        self._layers=np.array(_layers)

    def assign_layers(self):
        """ Determine the ITIM layers. 

        """
        # TODO: speedup this part of the code.
        self._assign_mesh()
        group=self.itim_group ; delta = self.delta ;
        mesh_dx = self.mesh_dx ; mesh_dy = self.mesh_dy
        up=0 ; low=1
        self.layers_ids=[[],[]] ;# upper, lower
        self.mask=np.zeros((2,self.max_layers,self.mesh_nx,self.mesh_ny),
                            dtype=int);
        self.center()

        _radius=group.radii
        self._seen=np.zeros(len(self._x(group)))

        _x=self._x(group)
        _y=self._y(group)
        _z=self._z(group)
        
        sort = np.argsort( _z + _radius * np.sign(_z) )
        # NOTE: np.argsort returns the sorted *indices*

        if self.use_multiproc:
            # so far, it justs exploit a simple scheme splitting
            # the calculation between the two sides. Would it be
            # possible to implement easily 2d domain decomposition? 
            proc=[[],[]] ; queue=[[],[]] ; seen=[[],[]]
            queue[up]=Queue()
            proc[up]  = Process(target=self._assign_one_side,
                                args=(up,sort[::-1],_x,_y,_z,_radius,
                                queue[up]))
            queue[low]=Queue()
            proc[low] = Process(target=self._assign_one_side,
                                args=(low,sort,_x,_y,_z,_radius,
                                queue[low]))
            for p in proc: p.start()
            for q in queue: self._seen+=q.get()
            self.layers_ids[low]+=queue[low].get()
            self.layers_ids[up]+=queue[up].get()
            for p in proc: p.join()
        else:
            self._assign_one_side(up,sort[::-1],_x,_y,_z,_radius)
            self._assign_one_side(low,sort,_x,_y,_z,_radius)
        self._define_layers_groups()
        self.bfactors = self._seen

    def layers(self,side='both',*ids):
        """ Select one or more layers.

        :param str side: 'upper', 'lower' or 'both'
        :param slice ids: the slice corresponding to the layers to be selcted (starting from 0) 

        The slice can be used to select a single layer, or multiple, e.g. (using the example of the :class:`ITIM` class) :

        >>> interface.layers('upper')  # all layers, upper side
        array([<AtomGroup with 161 atoms>, <AtomGroup with 202 atoms>,
               <AtomGroup with 280 atoms>, <AtomGroup with 272 atoms>], dtype=object)

        >>> interface.layers('lower',1)  # first layer, lower side
        <AtomGroup with 144 atoms>

        >>> interface.layers('both',0,3) # 1st - 3rd layer, on both sides
        array([[<AtomGroup with 161 atoms>, <AtomGroup with 202 atoms>,
                <AtomGroup with 280 atoms>],
               [<AtomGroup with 144 atoms>, <AtomGroup with 178 atoms>,
                <AtomGroup with 267 atoms>]], dtype=object)

        >>> interface.layers('lower',0,4,2) # 1st - 4th layer, with a stride of 2, lower side 
        array([<AtomGroup with 144 atoms>, <AtomGroup with 267 atoms>], dtype=object)


        """
        _options={'both':slice(None),'upper':0,'lower':1}
        _side=_options[side]
        if len(ids) == 0:
            _slice = slice(None)
        if len(ids) == 1:
            _slice = slice(ids[0])
        if len(ids) == 2:
            _slice = slice(ids[0],ids[1])
        if len(ids) == 3:
            _slice = slice(ids[0],ids[1],ids[2])

        if len(ids) == 1 and side != 'both':
            return self._layers[_side,_slice][0]

        if len(ids) == 1 :
            return self._layers[_side,_slice][:,0]

        return self._layers[_side,_slice]



if __name__ == "__main__":
    import argparse
    from matplotlib import pyplot as plt
    from observables import *
    parser = argparse.ArgumentParser(description='Description...')
    #TODO add series of pdb/gro/...
    parser.add_argument('--top'                                       )
    parser.add_argument('--trj'                                       )
    parser.add_argument('--info'     , action  = 'store_true'         )
    parser.add_argument('--alpha'    , type = float , default = 1.0   )
    parser.add_argument('--selection', default = 'all'                )
    parser.add_argument('--layers'   , type = int   , default = 1     )
    parser.add_argument('--dump'   ,default = False
                                                     ,help="Output to pdb trajectory")
    # TODO: add noncontiguous sampling
    args = parser.parse_args()

    u=None
    if args.top is None and args.trj is None:
        parser.print_help()
        exit()
    else:
        try:
            u = Universe(args.top,args.trj)
        except:
            pass

        if args.top is None:
            u = Universe(args.trj)
        if args.trj is None:
            u = Universe(args.top)

    if u is None:
        print "Error loadinig input files",exit()

    g = u.select_atoms(args.selection)

    interface = ITIM(u,
                info=args.info,
                alpha=args.alpha,
                max_layers = args.layers,
                itim_group = g
                )
    rdf=None
    rdf2=None
    orientation=MolecularOrientation(u)
    orientation2=MolecularOrientation(u,options='normal')

    all1 = u.select_atoms("name OW")
    all2 = u.select_atoms("name OW")
    for frames, ts in enumerate(u.trajectory[::50]) :
        print "Analyzing frame",ts.frame+1,\
              "(out of ",len(u.trajectory),") @ ",ts.time,"ps"
        interface.assign_layers()
        g1=interface.layers('upper',1) 
        g2=interface.layers('upper',1)
        if(args.dump):
            interface.writepdb()
        
        tmp = InterRDF(all1,all2,range=(0.,ts.dimensions[0]/2.),function=orientation.compute)
#        tmp = InterRDF2D(g1,g2,range=(0.,ts.dimensions[0]/2.))
        tmp.sample(ts)
        tmp.normalize()

#        tmp2 = InterRDF2D(g1,g2,range=(0.,ts.dimensions[0]/2.),function=orientation2.compute)
#        tmp2.sample(ts)
#        tmp2.normalize()


        if rdf  is None:
            rdf  =tmp.rdf
#            rdf2 =tmp2.rdf
        else:
            rdf +=tmp.rdf
#            rdf2+=tmp2.rdf

    np.savetxt('angle3d.dat',np.column_stack((tmp.bins,rdf/(frames+1))))




#
