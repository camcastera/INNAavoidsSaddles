import numpy as np 
from oracles import *

##################################################################
save_every = 10

loc_min_1 = np.array([1.,0.])
loc_min_2 = np.array([-1.,0.])
saddle_1 = np.array([0.,0.])

def INNA(x0,alpha,beta,gamma,oracle,niter=1000):
    list_J,list_gradJ, closest_point = [],[],[]

    theta=np.copy(x0)
    #Initialize psi
    _,gradJ = oracle(theta)
    psi=(1-alpha*beta)*theta - (beta**2-beta)*gradJ
    n=0
    while n < niter:
        J,gradJ = oracle(theta)
        list_J.append(J)
        list_gradJ.append(norm(gradJ))
        
        theta_old,psi_old = np.copy(theta),np.copy(psi)

        psi = psi_old + gamma * ( (1./beta-alpha)*theta_old -1./beta * psi_old )
        theta = theta_old + gamma * ( (1./beta-alpha)*theta_old -1./beta * psi_old - beta*gradJ)
        
        
        if n%save_every == 0:
            distances = np.array([norm(theta-loc_min_1),norm(theta-loc_min_2),norm(theta-saddle_1)])
            try:
                clp = (int(np.min(distances)*100),np.argmin(distances))
            except:
                clp = (np.nan,np.nan)
            closest_point.append(clp)
        n+=1

    distances = np.array([norm(theta-loc_min_1),norm(theta-loc_min_2),norm(theta-saddle_1)])
    try:
        clp = (int(np.min(distances)*100),np.argmin(distances))
    except:
        clp = (np.nan,np.nan)
    closest_point.append(clp)
    return list_J, list_gradJ, closest_point

def GD(x0,gamma,oracle,niter=1000):
    list_J,list_gradJ, closest_point = [],[],[]

    theta=np.copy(x0)

    n=0
    while n < niter:
        J,gradJ = oracle(theta)
        list_J.append(J)
        list_gradJ.append(norm(gradJ))
        #norm_thetaP.append(np.sqrt(theta[-1]**2))
        
        theta = theta - gamma *  gradJ
        
        if n%save_every == 0:
            distances = np.array([norm(theta-loc_min_1),norm(theta-loc_min_2),norm(theta-saddle_1)])
            try:
                clp = (int(np.min(distances)*100),np.argmin(distances))
            except:
                clp = (np.nan,np.nan)
            closest_point.append(clp)
        n+=1
        
    distances = np.array([norm(theta-loc_min_1),norm(theta-loc_min_2),norm(theta-saddle_1)])
    try:
        clp = (int(np.min(distances)*100),np.argmin(distances))
    except:
        clp = (np.nan,np.nan)
    closest_point.append(clp)
    return list_J, list_gradJ, closest_point