import numpy as np 
import matplotlib.pyplot as plt
import pickle

from oracles import *
from Algorithms import *

import time

plt.close('all')
#########Main Parameters############
nmax = 2000
save_every = 1000
num_exp = 200

from os import walk 
_, _, filenames = next(walk('Results'))

#####################
# List of Algorithms:
# * Gradient descent 
# * INNA (0.5,1.)
# * INNA (1.,1.1)
#####################


list_params = []


full_Hist = {}
for file in filenames:
    f = file
    temp = f.replace('_',' ').replace('.pkl','').split()
    if temp[0] == 'INNA':
        full_Hist[temp[0],temp[4],temp[6]] = pickle.load(open('Results/'+file,'rb'))
        list_params.append((temp[0],temp[4],temp[6]))
    else:
        full_Hist[temp[0],None,None] = pickle.load(open('Results/'+file,'rb'))
        list_params.append((temp[0],None,None))

Tab = {}
for key in list_params:
    Tab[key] = {}

for key in list_params:
    cpt = {0:0,1:0,2:0}
    mean_iters = 0.
    num_average = 0
    print('Number of exp with diverging iterates:',np.sum(1*np.array(full_Hist[key]['is_diverging'])))
    closest_points = full_Hist[key]['closest_point']
    for item in closest_points.items():
        if item[1][-1][1] == item[1][-1][1]: #check for NaNs
            cpt[item[1][-1][1]] += 1 #Count which point is reached each time
            #Estimate time to escape saddles
            temp = np.array([ t[1] for t in item[1]])
            if temp[0] == 2  and (temp[-1]==0 or temp[-1]==1):
                num_average += 1 #Count this exp as one where there is an escape
                mean_iters+= (np.sum(temp[:-1]==2))*(nmax/save_every)
    Tab[key]['point reached'] = cpt
    Tab[key]['mean iters to escape'] = int(mean_iters/num_average) if num_average>0 else nmax

print('\nClosest point after 2000 iterations\n')
for key in list_params:
    print(key, Tab[key]['point reached'])
print('\nAverage number of iterations to escape saddle points\n')
for key in list_params: 
    print(key, Tab[key]['mean iters to escape'])


############## 3D plot of the function ################

'''
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
ax.plot(x,y,z,linewidth=4,color='black',zorder=10,label='Stable Manifold',alpha=0.9)

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
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_zlabel(r'$\mathcal{J}(\theta_1,\theta_2)$')


plt.savefig('3D_stable_manifold.pdf')
'''