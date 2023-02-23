import numpy as np 
from oracles import *

##################################################################
save_every = 10

loc_min_1 = np.array([1.,0.])
loc_min_2 = np.array([-1.,0.])
saddle_1 = np.array([0.,0.])

def INNA(x0,alpha,beta,gamma,oracle,niter=1000):
    list_J,list_gradJ, list_theta = [],[],[]

    theta=np.copy(x0)
    #Initialize psi
    _,gradJ = oracle(theta)
    psi=(1-alpha*beta)*theta - (beta**2-beta)*gradJ
    n=0
    while n < niter:
        J,gradJ = oracle(theta)
        list_J.append(J)
        list_gradJ.append(norm(gradJ))
        list_theta.append(theta)
        
        theta_old,psi_old = np.copy(theta),np.copy(psi)

        psi = psi_old + gamma * ( (1./beta-alpha)*theta_old -1./beta * psi_old )
        theta = theta_old + gamma * ( (1./beta-alpha)*theta_old -1./beta * psi_old - beta*gradJ)
        
        n+=1

    return list_J, list_gradJ, list_theta

def GD(x0,gamma,oracle,niter=1000):
    list_J,list_gradJ, list_theta = [],[],[]

    theta=np.copy(x0)

    n=0
    while n < niter:
        J,gradJ = oracle(theta)
        list_J.append(J)
        list_gradJ.append(norm(gradJ))
        list_theta.append(theta)
        
        theta = theta - gamma *  gradJ
        
        n+=1

    return list_J, list_gradJ, list_theta


nmax = 1000

L = 50 #Estimated Lipschitz constant
gamma0 = 1./L

def do_one_exp(algo,gamma,theta0,nmax,alpha=None,beta=None):
    list_theta=[]
    if algo=='INNA':
        list_J,list_gradJ,theta = INNA(theta0,alpha,beta,gamma,oracle,niter=nmax)
    else:
        list_J,list_gradJ,theta = GD(theta0,gamma,oracle,niter=nmax)
    list_J = np.ravel(list_J)
    list_gradJ = np.ravel(list_gradJ)
    return list_J,list_gradJ,theta



def plot_traj(theta,ax,color,label,alpha=0.95,linestyle='-'):
    x = np.ravel(theta)[::2]
    y = np.ravel(theta)[1::2]
    J = x**4 - c0*x**2 + y**2
    ax.plot(x,y,J,linewidth=8,color=color,zorder=10,label=label,alpha=alpha,linestyle=linestyle)