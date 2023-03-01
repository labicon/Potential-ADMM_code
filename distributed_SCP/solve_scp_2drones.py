"""
Adapted from the problem "Cart-pole swing-up with limited actuation".
Autonomous Systems Lab (ASL), Stanford University
"""

import numpy as np
import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
from animations import animate_cartpole
import dpilqr as dec


@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def linearize(fd: callable,
              s: jnp.ndarray,
              u: jnp.ndarray):

    """Linearize the dynamics function `fd(s,u)` around nominal `(s,u)`."""
    # Use JAX to linearize `fd` around `(s,u)`.

    A = jax.jacfwd(fd,0)(s,u)
    B = jax.jacfwd(fd,1)(s,u)
    c = fd(s,u)-A@s-B@u

    return A, B, c


def solve_swingup_scp(fd: callable,
                      P: np.ndarray,
                      Q: np.ndarray,
                      R: np.ndarray,
                      Q1: np.ndarray, 
                      R1: np.ndarray,
                      N: int,
                      Z:int, #new variable
                      s_goal: np.ndarray,
                      s1_goal:np.ndarray,
                      s0: np.ndarray,
                      s20:np.ndarray,
                      ru: float,
                      ρ: float,
                      tol: float,
                      tol1:float,
                      max_iters: int):
    """Solve the cart-pole swing-up problem via SCP (the outer loop)."""
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    #added code:
    n1= Q1.shape[0]
    m1=R1.shape[0]

    # Initialize nominal trajectories
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))

    #added code: 
    u2_bar=np.zeros(Z, m1)
    s2_bar=np.zeros(Z+1, n1)

    s_bar[0] = s0

    #added code:
    s2_bar[0]=s20

    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k])
        
        #added code:
        s2_bar[k+1]=fd(s2_bar[k], u2_bar[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    #added code: 
    converged2= False

    obj_prev = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, s0,
                                  ru, ρ)
        diff_obj = np.abs(obj - obj_prev)
        prog_bar.set_postfix({'objective change': '{:.5f}'.format(diff_obj)})

        if diff_obj < tol:
            converged = True
            print('SCP converged after {} iterations.'.format(i))
            break
        else:
            obj_prev = obj
            np.copyto(s_bar, s)
            np.copyto(u_bar, u)

    if not converged:
        raise RuntimeError('SCP did not converge!')
        

    #added code
    for k in (prog_bar := tqdm(range(max_iters))):
        s2, u2, obj1 = scp_iteration(fd, P, Q, R, N, s2_bar, u2_bar, s1_goal, s20,
                                  ru, ρ)
        diff_obj = np.abs(obj1 - obj_prev)
        prog_bar.set_postfix({'objective change': '{:.5f}'.format(diff_obj)})

        if diff_obj < tol1:
            converged = True
            print('SCP converged after {} iterations.'.format(k))
            break
        else:
            obj_prev = obj
            np.copyto(s2_bar, s2)
            np.copyto(u2_bar, u2)

    if not converged:
        raise RuntimeError('SCP did not converge!')
        

    return s, u, s2, u2


def scp_iteration(fd: callable, P: np.ndarray, P1: np.ndarray, Q: np.ndarray, R: np.ndarray, Q1: np.ndarray, R1:np.ndarray,
                  N: int, Z:int, s_bar: np.ndarray, u_bar: np.ndarray, s1_bar: np.ndarray, u1_bar: np.ndarray,
                  s_goal: np.ndarray, s0: np.ndarray, s1_goal:np.ndarray,
                  ru: float, ρ: float, x_dims: list):
    """Solve a single SCP sub-problem for the cart-pole swing-up problem."""
    A, B, c = linearize(fd, s_bar[:-1], u_bar)
    A, B, c = np.array(A), np.array(B), np.array(c)

    n = Q.shape[0]
    m = R.shape[0]
    
    #added code:
    n1=Q1.shape[0]
    m2=R1.shape[0]

    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))
    
    #added code: 
    s1_cvx=cvx.Variable(Z+1, n1)
    u1_cvx=cvx.Variable(Z, m1)

    # ####################### PART (c): YOUR CODE BELOW #######################
    # Construct and solve the convex sub-problem for SCP.

    objective = 0.

    #added code: 
    objective1=0

    constraints = []
    constraints +=[s_cvx[0,:] == s_bar[0]]

    #added code: 
    constraints2=[]
    constraints2 += [s1_cvx[0, :]==s1_bar[0]]

    for k in range(N):
      

      objective += cvx.quad_form(s_cvx[k,:] - s_goal, Q) + cvx.quad_form(u_cvx[k,:], R) 

      constraints += [s_cvx[k+1,:] == A[k]@s_cvx[k,:] + B[k]@u_cvx[k,:] + c[k]]

      constraints += [-ru <= u_cvx[k,:], u_cvx[k,:] <= ru]

      constraints += [cvx.pnorm(s_cvx[k,:]-s_bar[k,:],'inf') <= ρ]

      constraints += [cvx.pnorm(u_cvx[k,:]-u_bar[k,:],'inf') <= ρ]

    objective += cvx.quad_form(s_cvx[-1,:] - s_goal, P)

    #added code: 
    for j in range(Z):
      

      objective1 += cvx.quad_form(s_cvx[k,:] - s_goal, Q) + cvx.quad_form(u_cvx[k,:], R) 

      constraints2 += [s1_cvx[j+1,:] == A[k]@s_cvx[j,:] + B[k]@u_cvx[j,:] + c[k]]

      constraints2 += [-ru <= u_cvx[j,:], u_cvx[j,:] <= ru]

      constraints2 += [cvx.pnorm(s_cvx[j,:]-s1_bar[j,:],'inf') <= ρ]

      constraints2 += [cvx.pnorm(u_cvx[k,:]-u1_bar[k,:],'inf') <= ρ]

    objective1 += cvx.quad_form(s_cvx[-1,:] - s1_goal, P1)

    # ############################# END PART (c) ##############################

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()  
    
    #added code: 
    prob1= cvx.Problem(cvx.Minimize(objective1), constraints2)
    prob1.solve()

    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)

    #added code: 
    if prob1.status !='optimal':
        raise RuntimeError('SCP solve failed.Problem_status:'+prob1.status)

    s = s_cvx.value
    u = u_cvx.value

    #added code: 
    s1=s1_cvx.value
    u1=u1_cvx.value

    obj = prob.objective.value

    #added code:
    obj1= prob1.objective1.value

    return s, u, obj, s1, u1, obj1
