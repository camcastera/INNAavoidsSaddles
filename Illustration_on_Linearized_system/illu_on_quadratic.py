import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

from IPython.display import display, Math

plt.close('all')



def norm(y):
  return np.sqrt(y.dot(y))

def f(y):
    val = a*y[0]**2 + b*y[1]**2 - c*y[0]*y[1]
    return val

def fprime(y):
  grad = np.zeros_like(y)
  grad[0] = a*2*y[0] -c*y[1]
  grad[1] = b*2*y[1] -c*y[0]
  return grad


def fsec(y):
  Hess = np.identity(y.shape[0])
  Hess[0,0] = 2*a
  Hess[1,1] = 2*b
  Hess[0,1] = -c
  Hess[1,0] = -c
  return Hess



def oracle(y):
    return f(y),fprime(y)


def INNA(x0,alpha,beta,gamma,oracle,niter=100,epsilon=1e-10,NAG=False):
    list_J,list_gradJ = [],[]
    alpha0 = np.copy(alpha)
    theta=np.copy(x0) ; T1 = [theta0[0]] ; T2 = [theta0[1]]
    #Initialize psi
    _,gradJ = oracle(theta)
    psi=(1-alpha*beta)*theta - (beta**2-beta)*gradJ
    n=0
    while n < niter and norm(gradJ)>epsilon:
        J,gradJ = oracle(theta)
        list_J.append(J)
        list_gradJ.append(norm(gradJ))
        
        if NAG:
            alpha = alpha0/((n+1)*np.sqrt(gamma))
        else:
            alpha = alpha0

        theta_old,psi_old = theta,psi

        psi = psi_old + gamma * ( (1./beta-alpha)*theta_old -1./beta * psi_old )
        theta = theta_old + gamma * ( (1./beta-alpha)*theta_old -1./beta * psi_old - beta*gradJ)

        T1.append(theta[0]) ; T2.append(theta[1])
        
        n+=1
  
    return list_J, list_gradJ, theta, T1, T2


# Main #

a=1.
b=2.
c=0.

gamma = 1e-2
epsilon=-1
nmax=7000



fig1,ax1 = plt.subplots(1,1,figsize=(9,4.5))
fig2,ax2 = plt.subplots(1,1,figsize=(4.5,4.5))
fig3,ax3 = plt.subplots(1,1,figsize=(4.5,4.5))

fig4,ax4 = plt.subplots(1,1,figsize=(9,4.5))
fig5,ax5 = plt.subplots(1,1,figsize=(9,4.5))

colors = ['red','orange','dodgerblue','limegreen']

iters=0
for alpha,beta,num_alg in [(2,0.1,0),(2,1,1),(2,0.1,2),(2,1,3)]:
    if alpha*beta<=1:
        lambda_min = (2-alpha*beta - 2*np.sqrt(1-alpha*beta))/(beta**2)
        lambda_max = (2-alpha*beta + 2*np.sqrt(1-alpha*beta))/(beta**2)
        print('range for problematic eigvals ',lambda_min,lambda_max)
    
    print('(alpha,beta) = ',alpha,beta )

    theta0 = np.array([1.,1.]) 
    NAG = False if num_alg<2. else True
    list_J,list_gradJ,theta, T1, T2 = INNA(theta0,alpha,beta,gamma,oracle,niter=nmax,epsilon=epsilon,NAG=NAG)
    color = colors[num_alg]
    if num_alg>1:
        label = r'$(\alpha,\beta)=$'+str(alpha)+r'$/t$,'+str(beta)
    else:
        label = r'$(\alpha,\beta)=$'+str(alpha)+','+str(beta)
    ax1.plot(T1,T2,color=color,linewidth=4,label=label)
    ax2.plot(list_J,color=color,label=label,linewidth=4)
    ax3.plot(np.sqrt((np.array(T1)-0.)**2+(np.array(T2)-0.)**2),color=color,label=label,linewidth=4)
    linewidth = 5 if num_alg !=2 else 2.5
    zorder = 10-num_alg if num_alg !=2 else 1
    ax4.plot(T1,T2,color=color,linewidth=linewidth,label=label,zorder=zorder)
    ax5.plot(T1,T2,color=color,linewidth=linewidth,label=label,zorder=zorder)
    iters+=1


x=np.linspace(-1.,1.,200)
y=np.linspace(-1.,1.,200)
X,Y = np.meshgrid(x,y)
Z = a*X**2 + b*Y**2 - c*X*Y
levels = np.logspace(-10,np.log10(np.max(Z)),100)
ct = ax1.contour(X, Y, Z,levels = levels)
ax1.legend(loc='upper left')
ax1.set_xlabel(r'$\theta_{k,1}$')
ax1.set_ylabel(r'$\theta_{k,2}$')

ax2.set_yscale('log')
ax2.legend(loc='lower left')
ax2.set_xlabel(r'number of iteration $k$')
ax2.set_ylabel(r'$\mathcal{J}(\theta_k)$')



ax3.set_yscale('log')
ax3.legend(loc='lower left')
ax3.set_xlabel(r'number of iteration $k$')
ax3.set_ylabel(r'$\Vert \theta_k \Vert$')


x=np.linspace(-0.01,0.01,200)
y=np.linspace(-0.015,0.015,200)
X,Y = np.meshgrid(x,y)
Z = a*X**2 + b*Y**2 - c*X*Y
levels = np.logspace(np.log10(np.min(Z)),np.max(Z),50)
ct = ax4.contour(X, Y, Z,levels = levels)
ax4.set_xlim(-0.01,0.01)
ax4.set_ylim(-0.015,0.015)
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)


x=np.linspace(-0.00005,0.00005,200)
y=np.linspace(-0.000035,0.000035,200)
X,Y = np.meshgrid(x,y)
Z = a*X**2 + b*Y**2 - c*X*Y
levels = np.logspace(np.log10(np.min(Z)),np.max(Z),50)
ct = ax5.contour(X, Y, Z,levels = levels)
ax5.set_xlim(-0.00005,0.00005)
ax5.set_ylim(-0.000035,0.000035)
ax5.get_xaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)

fig1.tight_layout() ; fig2.tight_layout() ; fig3.tight_layout() ; fig4.tight_layout() ; fig5.tight_layout() ; 

fig1.savefig('2D_example.pdf') ; fig4.savefig('2D_example_zoom1.pdf') ; fig5.savefig('2D_example_zoom2.pdf') ; 
fig2.savefig('2D_example_loss.pdf') ; fig3.savefig('2D_example_grad.pdf') ;

plt.ion()
plt.show()

    

    

