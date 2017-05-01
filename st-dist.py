#!/usr/bin/python

import numpy as np
import time
from scipy.interpolate import griddata
import MDAnalysis as md
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm,colors
import mpl_toolkits
print dir(mpl_toolkits.mplot3d)
cmap = cm.PRGn


########
do_plot=0
########

u = md.Universe("box.gro" , "traj.trr")

theta=list()
phi=list()
G=list()
bins=40
#plt.ion()
plt.hold(True)
image=None 
step=0

pdbtrj = "adk_distance_bfac.pdb"
with md.Writer(pdbtrj, multiframe=True, bonds=False, n_atoms=u.atoms.n_atoms) as PDB:
	for ts in u.trajectory:  
		ST=list()
		print "time:",ts.time
		step+=1
		try:
			bins = int(open('bins').read())
		except:
			pass
		for mol in u.residues:
			com = mol.center_of_mass()	
			rO= mol[0].position - com
			rH1 = mol[1].position - com
			rH2 = mol[2].position - com
			v1=(rH1+rH2)
			v1 = v1 / np.sqrt(v1.dot(v1))
			v2 = np.cross(rH1,rH2)
			v2 = v2 / np.sqrt(v2.dot(v2))
			theta.append(np.arccos( v1[2] ))
			phi.append(np.arccos(v2[2]/np.sqrt(1-v1[2]*v1[2])))
			K=np.zeros(3)
			Vcm=np.zeros(3)
			totm=0
			for atom in mol:
				Vcm+=atom.mass*atom.velocity	
				totm+=atom.mass
			Vcm/=totm
			for atom in mol:
				k=atom.mass*(atom.velocity-Vcm)**2
				K+=k
				ST.append ( k[2]-0.5*(k[0]+k[1]) )
			G.append ( K[2]-0.5*(K[0]+K[1]) )
		st=np.array(ST)
		min=np.amin(st)
		max=np.amax(st)
		u.atoms.set_bfactors((st-min)/(max-min))	
		PDB.write(u.atoms)
		H, Tedge, Pedge = np.histogram2d(theta, phi, bins=bins, weights=G)
		H=H/step
		if step==50:
			np.savetxt("tab.dat",H)
			exit()
		X, Y = np.meshgrid(Tedge[0:-1], Pedge[0:-1])
		if do_plot:
			if image is None:
				norm = cm.colors.Normalize(vmax=abs(H).max(), vmin=-abs(H).max())
				#image = plt.imshow(H, interpolation='nearest', origin='low',extent=[Tedge[0], Tedge[-1], Pedge[0], Pedge[-1]],animated=True)
				plt.draw()
			else:
				plt.clf()
				image.set_data(H)
				plt.draw()
		
		if step%5==0 and do_plot: 
			plt.clf()
			c=plt.contourf(X, Y, H,64,alpha=1,antialiased=False,linewidths=2,linestyle='solid',extend3d=True)
			plt.draw()
	
		#	theta.append(np.arccos( v1[2] ))
		#	phi.append(np.pi+np.arccos(v2[2]/np.sqrt(1-v1[2]*v1[2])))
		#	G.append ( K[2]-0.5*(K[0]+K[1]) )
	#		for atom in mol:
	#			[x,y,z] = atom.position
	#			if z > ts.dimensions[2]/4. and z < ts.dimensions[2]/2.:
	#				[x,y,z] = atom.position  - com
	#				theta.append( np.arccos(z/np.sqrt(x*x+y*y+z*z)) )
	#				phi.append( np.arctan2(y,x) ) 
	#				K=atom.mass*atom.velocity*atom.velocity
	#				G.append ( K[2]-0.5*(K[0]+K[1]) ) 
	
	
	
	#	if ts.time >= 1.:
	#		H, Tedge, Pedge = np.histogram2d(theta, phi, bins=40, weights=G)
	#		#H, Tedge, Pedge = np.histogram2d(theta, phi, bins=40)
	#
	#		bbins=200
	#		points = np.array(np.meshgrid(Tedge[0:-1],Pedge[0:-1])).T.reshape(-1,2)
	#		gx, gy = np.mgrid[0:np.pi:bbins*1j, 0:2.*np.pi:bbins*1j]
	#		histo = H.flatten()
	#		gz = griddata(points, histo, (gx, gy), method='nearest')
	#
	#
	#		fig = plt.figure()
	#		ax = fig.add_subplot(111, projection='3d')
	#		ax.view_init(45,60)
	#		ax.plot_surface(np.sin(gx)*np.cos(gy),np.sin(gx)*np.sin(gy),np.cos(gx), facecolors=cm.Oranges(gz))
	#		plt.figure()
	#	#	ax.plot_surface(np.sin(xedges)*np.cos(yedges),np.sin(xedges)*np.sin(yedges),np.cos(xedges), facecolors=cm.Oranges(H))
	#	#	ax = fig.add_subplot(131)
	#		im = plt.imshow(H, interpolation='nearest', origin='low',extent=[Tedge[0], Tedge[-1], Pedge[0], Pedge[-1]])
	#		plt.show()
	#		break
	
