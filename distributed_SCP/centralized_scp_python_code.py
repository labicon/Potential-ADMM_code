#code for centralized version of scp 
import numpy as np
import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import dpilqr as dec
# from animations import animate_cartpole


@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))



T_duration= 15
dt= 0.01

range_possible_pos= np.array([[-3, 3], [-3, 3], [-3, 3]])
drone_distance= 1 #meter
max_acceleration= 3.
num_drones=4
#initial positions for 4 drones 
pi1= np.array([[1, 1, 1]])
pi2= np.array([[1.5, 1.5, 1.5]])
pi3= np.array([[-1, -1, -1]])
pi4=np.array([[-1.5, -1.5, -1.5]])
   
#final positions

    pf1=np.array([[-1, -1, -1]])
    pf2=np.array([[-1.5, -1.5, -1.5]])
    pf3=np.array([[1, 1., 1]])
    pf4=np.array([[1.5, 1.5, 1.5]])

def linearize(fd:callable, s: jnp.ndarray, u:jnp.ndarray):
    
    A = jax.jacfwd(fd,0)(s,u)
    B = jax.jacfwd(fd,1)(s,u)
    c = fd(s,u)-A@s-B@u
    
    
    return A, B, c


def initAllSolutions(po, pf, h, K):
    po = np.squeeze(po)
    pf = np.squeeze(pf)
    N = po.shape[1]
    t = np.arange(0, K*h, h)
    p = np.zeros((po.shape[0], len(t), N))
    for i in range(N):
        diff = pf[:,i] - po[:,i]
        for k in range(len(t)):
            p[:,k,i] = po[:,i] + t[k]*diff/((K-1)*h)
    return p





    #i need to convert velocity to acceleration now 
def collision_constraints(A, b):
   #define collision constraints  


def centralizedSCP (po,pf,h,K,N,pmin,pmax,rmin,alim,A_p, A_v,E1,E2,order):
    prev_pos= initAllSolutions(po, pf, h, K)

    upper_bound= max_acceleration

    #you need the solve QP part and inequality constraints 