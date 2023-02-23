import numpy as np 
import pickle

P = 2

c0=4

def loss(theta):
    J= theta[0]**4 - c0*theta[0]**2 + theta[1]**2
    return  J

def grad(theta):
    grad = np.zeros_like(theta)
    grad[0] = 4*theta[0]**3 - 2*c0*theta[0]
    grad[1] = 2*theta[1]
    return grad

def Hess(theta):
    P = theta.shape[0]
    d = np.zeros(P)
    d[0] = 12*theta[0]**2-2*c0
    d[1] = 2
    H = np.diag(d)
    return H

def oracle(theta):
    return loss(theta), grad(theta)

def norm(theta):
    if len(theta.shape)==2:
        return np.sqrt(theta.T.dot(theta))
    else:
        return np.sqrt(theta.dot(theta))