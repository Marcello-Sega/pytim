#!/usr/bin/python

from timeit import default_timer as timer
#from threading import Thread
from multiprocessing import Process, Queue
import numpy as np
from MDAnalysis.core.AtomGroup   import *

class ITIM():

    def __init__(self,universe,mesh=0.4,alpha=1.0,itim_group="all",
                 max_layers=1,pdb="layers.pdb",info=False):
        #TODO:CRITICAL handle the radii...
        try:
            types = np.copy(universe.select_atoms(itim_group).types)
            radii = np.zeros(len(universe.select_atoms(itim_group).types))
            radii[types=='OW']=3.5/2.
            radii[types=='H']=0.
            universe.select_atoms(itim_group).radii=radii
            del radii
            del types
        except:
            print "Error while selecting the group:",itim_group

        self.universe=universe
        self.target_mesh=mesh
        self.alpha=alpha
        self.max_layers=max_layers
        self.itim_group=itim_group ; # TODO add analysis groups
        self.grid=None
        self.info=info
        self.use_threads=False
        self.use_multiproc=True
        try:
            self.PDB=Writer(pdb, multiframe=True, bonds=False,
                            n_atoms=self.universe.atoms.n_atoms)
        except:
            self.PDB=None

            self.tic=timer()

    def lap(self):
        toc=timer()
        print "LAP >>> ",toc-self.tic
        self.tic=toc


    def x(self,group="all"):
        return self.universe.select_atoms(group).positions[:,0]

    def y(self,group="all"):
        return self.universe.select_atoms(group).positions[:,1]

    def z(self,group="all"):
        return self.universe.select_atoms(group).positions[:,2]

    def _rebox(self,x=None,y=None,z=None):
        dim = self.universe.coord.dimensions
        stack=False
        shift=np.array([0,0,dim[2]/2.])

        # we rebox the atoms in universe, and not a vector
        if x is None and y is None and z is None:
            stack=True
            x=self.x()
            y=self.y()
            z=self.z()
        for index, val in enumerate((x,y,z)):
            try:
                val[val> dim[index]-shift[index]]-=dim[index]
                val[val< 0        -shift[index]]+=dim[index]
            except:
                pass
        if stack:
            self.universe.coord.positions=np.column_stack((x,y,z))

    def writepdb(self,_seen):
        self.universe.atoms.bfactors=_seen
        try:
            self.PDB.write(self.universe.atoms)
        except:
            pass


    def center(self):
        #TODO: implement shifting back for final coordinate writing as an option
        dim = self.universe.coord.dimensions
        shift=dim[2]/100. ;# TODO, what about non ortho boxes?
        total_shift=0
        self._rebox()

        _z_itim_group = self.z(self.itim_group)
        _x = self.x()
        _y = self.y()
        _z = self.z()

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


    def assign_mesh(self):
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

    def assign_one_side(self,uplow,layer,sorted_atoms,_x,_y,_z,
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

    def define_layers_groups(self):
        self.layers=[[],[]]
        for i,uplow in enumerate(self.layers_ids):
            for j,layer in enumerate(uplow):
                self.layers[i].append(self.universe.atoms[layer])

    def assign_layers(self):
        # TODO: speedup this part of the code.
        self.assign_mesh()
        group=self.itim_group ; delta = self.delta ;
        mesh_dx = self.mesh_dx ; mesh_dy = self.mesh_dy
        layer=0 ; up=0 ; low=1
        self.layers_ids=[[],[]] ;# upper, lower
        self.mask=np.zeros((2,self.max_layers,self.mesh_nx,self.mesh_ny),
                            dtype=int);
        self.center()

        _radius=self.universe.select_atoms(group).radii
        self._seen=np.zeros(len(self.x(group)))

        _x=self.x(group)
        _y=self.y(group)
        _z=self.z(group)

        sort = np.argsort( _z + _radius * np.sign(_z) )

        if self.use_multiproc:
            proc=[[],[]] ; queue=[[],[]] ; seen=[[],[]]
            queue[up]=Queue()
            proc[up]  = Process(target=self.assign_one_side,
                                args=(up,layer,sort[::-1],_x,_y,_z,_radius,
                                queue[up]))
            queue[low]=Queue()
            proc[low] = Process(target=self.assign_one_side,
                                args=(low,layer,sort,_x,_y,_z,_radius,
                                queue[low]))
            for p in proc: p.start()
            for q in queue: self._seen+=q.get()
            self.layers_ids[low]+=queue[low].get()
            self.layers_ids[up]+=queue[up].get()
            for p in proc: p.join()
        else:
            self.assign_one_side(up,layer,sort[::-1],_x,_y,_z,_radius)
            self.assign_one_side(low,layer,sort,_x,_y,_z,_radius)
        self.define_layers_groups()
        self.writepdb(self._seen)




if __name__ == "__main__":
    import argparse
    from matplotlib import pyplot as plt
    from observables import *
    parser = argparse.ArgumentParser(description='Description...')
    #TODO add series of pdb/gro/...
    parser.add_argument('--top'                        )
    parser.add_argument('--trj'                        )
    parser.add_argument('--info'   ,action  = 'store_true')
    parser.add_argument('--alpha'  ,default = 1.0         )
    parser.add_argument('--selection',default = 'all'     )
    parser.add_argument('--layers'    ,default = 1        )
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

    itim = ITIM(u,
                info=args.info,
                pdb=args.dump,
                alpha=args.alpha,
                max_layers = args.layers,
                itim_group = args.selection
                )
    rdf=None
    rdf2=None
    orientation=MolecularOrientation(u)
    orientation2=MolecularOrientation(u,options='normal')

    all1 = u.select_atoms("name OW")
    all2 = u.select_atoms("name OW")
    for frames, ts in enumerate(u.trajectory[::50]) :
        print "Analyzing frame",ts.frame,\
              "(out of ",len(u.trajectory),") @ ",ts.time,"ps"
        itim.assign_layers()
        g1=itim.layers[0][0]
        g2=itim.layers[0][0]
        
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

    np.savetxt('angle3d.dat',np.column_stack((tmp.bins,rdf/frames)))




#
