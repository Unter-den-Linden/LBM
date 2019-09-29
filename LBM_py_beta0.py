# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 23:44:01 2019

@author: Unter den Linden

Lattice Boltzmann Code - Flow around 2D Squre Cylinder
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
#Initialization
NY = 80; NX = 200   # Grid size
mu = 0.005	# viscosity
om = 1 / (3*mu + 0.5)     # relaxation time
u0 = 0.1    # flow speed
f=np.empty([9,NY,NX])
c = np.array([(x,y) for x in [0,-1,1] for y in [0,-1,1]])

t = 4*np.ones(9)/9
t[1:9]=t[1:9]/4
t[4:6]=t[4:6]/4; t[7:9]=t[7:9]/4

rho = np.ones([NY,NX])
ux=np.ones([NY,NX])*u0
ux[0:40,0:10]=0.105
uy=np.zeros([NY,NX])
uu = 1 - 1.5*(ux*ux+uy*uy)
cu = np.dot(c,np.vstack((ux.reshape([NX*NY]),uy.reshape([NX*NY])))).reshape(9,NY,NX)

f = np.array([t[i] * rho * (uu+3*cu[i]+4.5*cu[i]**2) for i in range(9)])

#%%
# Mask solid
solid = np.zeros((NY,NX), bool)					
solid[int(NY/2)-6:int(NY/2)+6, int(NY/2)-6:int(NY/2)+6] = True

ba = np.ones([9,NY,NX])
ba = ba > 5

ba[2] = np.roll(solid,  1, axis=0)					
ba[1] = np.roll(solid, -1, axis=0)					
ba[6] = np.roll(solid,  1, axis=1)
ba[3] = np.roll(solid, -1, axis=1)
ba[8] = np.roll(ba[2],  1, axis=1)
ba[5] = np.roll(ba[2], -1, axis=1)
ba[7] = np.roll(ba[1],  1, axis=1)
ba[4] = np.roll(ba[1], -1, axis=1)

BC = [0,2,1,6,8,7,3,5,4]
#%%
rho = np.sum(f,axis=0)
ux=np.dot(c[:,0].T,f.reshape([9,NX*NY])).reshape([NY,NX]) / rho
uy=np.dot(c[:,1].T,f.reshape([9,NX*NY])).reshape([NY,NX]) / rho

#%%
# Stream step
def stream(f):
	f[1:9]=np.array([np.roll(np.roll(f[i],c[i,0],axis=1),c[i,1],axis=0) for i in range(1,9)])
	for i in range(1,9):
		f[i][ba[i]]=f[BC[i]][solid] # boundary
	return f
	
#%%
# Collision step
def collision(f):
    rho = np.sum(f,axis=0)
    ux=np.dot(c[:,0].T,f.reshape([9,NX*NY])).reshape([NY,NX]) / rho
    uy=np.dot(c[:,1].T,f.reshape([9,NX*NY])).reshape([NY,NX]) / rho
    uu = 1 - 1.5*(ux*ux+uy*uy)			
    cu = np.dot(c,np.vstack((ux.reshape([NX*NY]),uy.reshape([NX*NY])))).reshape(9,NY,NX)

    for i in range(9):
        f[i] = (1-om)*f[i] + om * t[i] * rho * (uu+3*cu[i]+4.5*cu[i]**2)
    
    for i in range(3,9):
        f[i,:,0] = t[i]*(1+3*c[i,0]*u0+3*u0**2)
    return f,ux,uy

#%%
def vort(ux, uy):
	return np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1) - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0)

fig= plt.figure(figsize=(int(NX/10),int(NY/10)))
X,Y = np.meshgrid(np.arange(0,1,1/NX),np.arange(0,1,1/NY))

SolidIm = np.ones([NY, NX])	
SolidIm[solid] = 0

#%%
def main(arg):							
	global f
	plt.cla()
	for step in range(200):
		f=stream(f)
		f,ux,uy=collision(f)
	levels = np.arange(-0.2,0.2,0.005)
	vor=vort(ux,uy)
	vor[solid] = 0
	plt.contour(X,Y,vor,levels,cmap='jet',vmin=-0.2,vmax=0.2)
	plt.contourf(X,Y,SolidIm,levels=[-1,0,1],colors=['black','white','blue'])

animate = anime.FuncAnimation(fig, main, interval=100)
plt.show()
