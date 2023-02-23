import numpy as np 
import matplotlib.pyplot as plt
import pickle

from oracles import *
from Algorithms_for_plots import *

import time

plt.close('all')

############## 3D plot of the J ################


from mpl_toolkits import mplot3d

X = np.outer(np.linspace(-2,2, 30), np.ones(30))
Y = X.copy().T # transpose
J =  X**4 - c0*X**2 + Y**2


fig = plt.figure(figsize=(12,12))
ax = plt.axes(projection='3d')



############# Color surface by convexity #############
concavity = (X>-np.sqrt(c0/6)) * (X<np.sqrt(c0/6))
concavity = np.array(concavity,dtype=float)

from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import Normalize, SymLogNorm

concavity[concavity==0]=-1
color_value = concavity*(J-np.min(J))
norm = Normalize(vmin=color_value.min().min(), vmax=color_value.max().max())
my_color = cm.bwr(norm(color_value))
#####################"##############################

ax.plot_surface(X, Y, J,cmap='cool', edgecolor='none',facecolors = my_color)


#Stable Manifold #
x=np.zeros(30)
y = np.linspace(-1.95,1.95,30)
z = y**2
ax.plot(x,y,z,linewidth=8,color='dimgrey',zorder=9,label=r'Stable Manifold of $(0,0)$',alpha=0.9)

# Get rid of background color #
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False


# Change ticks #
ticks = np.array([-2,-1.,0.,1.,2])
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ticks = np.array([-4,-2,0.,2.,4])
ax.set_zticks(ticks)
ax.set_xlabel(r'$\theta_1$',fontsize=25)
ax.set_ylabel(r'$\theta_2$',fontsize=25)
ax.set_zlabel(r'$\mathcal{J}(\theta_1,\theta_2)$',fontsize=25)


theta0 = np.array([0.,1.9]) 
list_J_INNA_grad0,_,theta_INNA_grad  = do_one_exp('INNA',gamma0,theta0,nmax,2.,1.)
theta0 = np.array([0,-1.9])
list_J_INNA_spiral0,_,theta_INNA_spiral  = do_one_exp('INNA',gamma0,theta0,nmax,0.5,1.)

plot_traj(theta_INNA_grad,ax,'blue',r'INNA $ \alpha\beta>1$',alpha=1,linestyle='dotted')
plot_traj(theta_INNA_spiral,ax,'limegreen',r'INNA $ \alpha\beta<1$ ',alpha=1,linestyle='dotted')

#plt.legend()

#plt.ion()
ax.view_init(azim=-51,elev=56)
#plt.show()
fig.tight_layout()
fig.savefig('3D_stable_manifold1.pdf')

################################################################








############## 2D contour plot of J ###################################


from mpl_toolkits import mplot3d


X = np.outer(np.linspace(-2,2, 30), np.ones(30))
Y  = np.outer(np.ones(30),np.linspace(-2,2, 30)) # transpose
J =  X**4 - c0*X**2 + Y**2


fig2 = plt.figure(figsize=(12,12))
ax2 = plt.axes(projection='3d')


############# Color surface by convexity #############
concavity = (X>-np.sqrt(c0/6)) * (X<np.sqrt(c0/6))
concavity = np.array(concavity,dtype=float)

from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import Normalize, SymLogNorm

concavity[concavity==0]=-1
color_value = concavity*(J-np.min(J))
norm = Normalize(vmin=color_value.min().min(), vmax=color_value.max().max())
my_color = cm.bwr(norm(color_value))
#####################"##############################

ax2.plot_surface(X, Y, J,cmap='cool', edgecolor='none',facecolors = my_color)


#Stable Manifold #
x=np.zeros(30)
y = np.linspace(-1.95,1.95,30)
z = y**2
ax2.plot(x,y,z,linewidth=8,color='dimgrey',zorder=9,label=r'Stable Manifold of $(0,0)$',alpha=0.9)

# Get rid of background color #
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False


# Change ticks #
ticks = np.array([-2,-1.,0.,1.,2])
ax2.set_xticks(ticks)
ax2.set_yticks(ticks)
ticks = np.array([-4,-2,0.,2.,4])
ax2.set_zticks(ticks)
ax2.set_xlabel(r'$\theta_1$',fontsize=25)
ax2.set_ylabel(r'$\theta_2$',fontsize=25)
ax2.set_zlabel(r'$\mathcal{J}(\theta_1,\theta_2)$',fontsize=25)



theta0 = np.array([-0.05,1.9]) 
list_J_INNA_grad,_,theta_INNA_grad  = do_one_exp('INNA',gamma0,theta0,nmax,1.,1.3)
theta0 = np.array([0.05,-1.9])
list_J_INNA_spiral,_,theta_INNA_spiral  = do_one_exp('INNA',gamma0,theta0,nmax,0.3,0.9)

plot_traj(theta_INNA_grad,ax2,'blue',r'INNA $ \alpha\beta>1$',alpha=1,linestyle='dotted')
plot_traj(theta_INNA_spiral,ax2,'limegreen',r'INNA $ \alpha\beta<1$ ',alpha=1,linestyle='dotted')


ax2.view_init(azim=-51,elev=70)
fig2.tight_layout()
fig2.savefig('3D_stable_manifold2.pdf')



legend = plt.legend(fontsize=14,loc='upper right',ncol=4,bbox_to_anchor=[1.5, 1.5])


def export_legend(legend, filename="legend_horizontal.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


export_legend(legend,'legend_inna_hor.pdf')
