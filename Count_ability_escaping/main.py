import sys
algo = str(sys.argv[1])
try:
    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
except:
    alpha = None
    beta = None

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
num_exp = 1000

L = 50 #Estimated local Lipschitz constant
gamma0 = 1./L 


#####################
# List of Algorithms:
# * Gradient descent 
# * INNA (0.5,1.)
# * INNA (1.,1.1)
#####################



def do_one_exp(algo,gamma,theta0,nmax,alpha=None,beta=None):
    if algo=='INNA':
        list_J,list_gradJ,theta = INNA(theta0,alpha,beta,gamma,oracle,niter=nmax)
    else:
        list_J,list_gradJ,theta = GD(theta0,gamma,oracle,niter=nmax)
    list_J = np.ravel(list_J)
    list_gradJ = np.ravel(list_gradJ)
    return list_J,list_gradJ,theta


loc_min_1 = np.array([1.,0.])
loc_min_2 = np.array([-1.,0.])
saddle_1 = np.array([0.,0.])



def main_loop(algo,gamma,nmax,num_exp,alpha=None,beta=None):
    print(algo,alpha,beta)
    closest_point,is_diverging,value_reached = {},[],[]
    
    for exp in range(num_exp):
        if exp%100 == 0:
            print(algo,' experiment number ', exp)
        theta0 = saddle_1 + 1e-12*np.random.randn(P)     

        list_J,list_gradJ,list_clp = do_one_exp(algo,gamma,theta0,nmax,alpha,beta)
        closest_point[exp] = list_clp
        is_diverging.append(np.max(list_J)>1e5)
        value_reached.append(np.min(list_J))
        if exp%50 == 0: #Save Hist every 50 experiments
            Hist = {}
            Hist['closest_point'] = closest_point
            Hist['is_diverging'] = is_diverging
            Hist['value_reached'] = value_reached
            if alpha is not None and beta is not None:
                pickle.dump(Hist,open('Results/'+algo+'_gamma_'+str(gamma)+'_alpha_'+str(alpha)+'_beta_'+str(beta)+'.pkl','wb'))
            else:
                pickle.dump(Hist,open('Results/'+algo+'_gamma_'+str(gamma)+'.pkl','wb'))
    Hist = {}
    Hist['closest_point'] = closest_point
    Hist['is_diverging'] = is_diverging
    Hist['value_reached'] = value_reached
    if alpha is not None and beta is not None:
        pickle.dump(Hist,open('Results/'+algo+'_gamma_'+str(gamma)+'_alpha_'+str(alpha)+'_beta_'+str(beta)+'.pkl','wb'))
    else:
        pickle.dump(Hist,open('Results/'+algo+'_gamma_'+str(gamma)+'.pkl','wb'))
    #return Hist
    pass

if alpha is not None and beta is not None:
    gamma = min( beta , (alpha+beta*L)/(2*L) + np.sqrt((alpha+beta*L)**2-4*L)/(2*L)  )
    gamma = min(gamma,gamma0)
else:
    gamma = gamma0
print('step-size: ',gamma)
main_loop(algo,gamma,nmax,num_exp,alpha=alpha,beta=beta)
