#code for centralized version of scp 
import numpy as np
import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import dpilqr as dec
from scipy.optimize import quadratic_assignment
from scipy.sparse import csr_matrix
from scipy.optimize import quadratic_assignment
# from animations import animate_cartpole

#use 6 dim dynamics

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


#def initAllSolutions(po, pf, h, K):  #change to nonlinear
    #po = np.squeeze(po)
    #pf = np.squeeze(pf)
    #N = po.shape[1]
    #t = np.arange(0, K*h, h)
    #p = np.zeros((po.shape[0], len(t), N))
    #or i in range(N):
        #diff = pf[:,i] - po[:,i]
        #for k in range(len(t)):
           # p[:,k,i] = po[:,i] + t[k]*diff/((K-1)*h)
    #return p

def initAllSolutions(fd:callable, s:jnp.ndarray, u:jnp.ndarray):
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k])
    return s_bar



    #i need to convert velocity to acceleration now 
#def solve_quadratic_program(H, Ain_total, Aeqtot):  #take this from cartpole example
 #   H_sparse = csr_matrix(H)
  #  Ain_total_sparse = csr_matrix(Ain_total)
   # Aeqtot_sparse = csr_matrix(Aeqtot)

    #result = quadratic_assignment(H_sparse, Ain_total_sparse)

    #atot, f0, exitflag = result[0], result[1], result[3]

    #if (len(atot) == 0 or exitflag == 0):
     #   p = []
      #  v = []
       # a = []
        #success = 0

    #return result

def solve_scp(fd: callable,
                      P: np.ndarray,
                      Q: np.ndarray,
                      R: np.ndarray,
                      N: int,
                      s_goal: np.ndarray,
                      s0: np.ndarray,
                      ru: float,
                      ρ: float,
                      tol: float,
                      max_iters: int,
                      n_drones: int):
    """This function is used for one-shot optimization"""
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension
    converged = False
    obj_prev = np.inf
    iterate = 0
    s_prev = None
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, s0,
                                  ru, ρ, iterate, s_prev, n_drones)
        
        iterate+=1

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
        
        s_prev = s

    if not converged:
        raise RuntimeError('SCP did not converge!')

    return s, u


def states (po, a, A_p, A_v, k, n):

    v_o = np.array([0, 0, 0, 0, 0, 0,0])
    po = np.squeeze(po)

    p = np.zeros((6*K, N))
    v = np.zeros((6*K, N))
    a = np.zeros((6, K, N))

    for i in range(N):
        ai = atot[6*K*(i-1):6*K*i]
        a[:, :, i] = np.reshape(ai, (6, K), order='F')
        new_p = A_p.dot(ai)
        new_v = A_v.dot(ai)
        pi = np.vstack((po[:, i], new_p + np.tile(po[:, i], (K-1, 1))))
        vi = np.vstack((vo, new_v))
        p[:, i] = np.reshape(pi, (6*K,), order='F')
        v[:, i] = np.reshape(vi, (6*K,), order='F')

    return p, v, a

#success = exitflag
#p, v, a = getStates(po, atot, A_p, A_v, K, N)
#prev_p = p
#criteria = abs(prev_f0 - f0)
#prev_f0 = f0
#k += 1


def centralizedSCP (po,pf,h,K,N,pmin,pmax,rmin,alim,A_p, A_v,E1,E2,order):

    n = Q.shape[0]  
    m = R.shape[0]

    converged = False
    obj_prev = np.inf
    iterate = 0
    s_prev = None
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, s0,
                                 ρ, iterate, s_prev, n_drones)
        
        iterate+=1

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
            ρ = ρ * 0.85
        
        s_prev = s

    if not converged:
        raise RuntimeError('SCP did not converge!')

    

    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k])




    prev_pos= initAllSolutions(po, pf, h, K)

    upper_bound= max_acceleration

    return s, u

    #you need the solve QP part and inequality constraints 


def scp_iteration(fd: callable, P: np.ndarray, Q: np.ndarray, R: np.ndarray,
                  N: int, s_bar: np.ndarray, u_bar: np.ndarray,
                  s_goal: np.ndarray, s0: np.ndarray,
                  ru: float, ρ: float, iterate: int, s_prev: np.ndarray):
    """Solve a single SCP sub-problem for the cart-pole swing-up problem."""
    A, B, c = linearize(fd, s_bar[:-1], u_bar)
    A, B, c = np.array(A), np.array(B), np.array(c)

    n = Q.shape[0]
    m = R.shape[0]
    n_drones = m//3
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    objective = 0.

    constraints = []
    constraints +=[s_cvx[0,:] == s_bar[0]]

    for k in range(N):
        
        objective += cvx.quad_form(s_cvx[k,:] - s_goal, Q) + cvx.quad_form(u_cvx[k,:], R) 
        constraints += [s_cvx[k+1,:] == A[k]@s_cvx[k,:] + B[k]@u_cvx[k,:] + c[k]]

        #Note: without the following convex trust regions, the solution blows up 
        #just after a few iterations    
    
        for i in range(n_drones):
            constraints += [cvx.pnorm(s_cvx[k,i*3:(i+1)*3]-s_bar[k,i*3:(i+1)*3],'inf') <= ρ]
            constraints += [cvx.pnorm(u_cvx[k,i*3:(i+1)*3]-u_bar[k,i*3:(i+1)*3],'inf') <= ρ] 
            
            #actuator constraints
            constraints += [-np.array([-np.pi/6, -np.pi/6, 0]) <= u_cvx[k, i*3:(i+1)*3], \
                        u_cvx[k, i*3:(i+1)*3] <= np.array([np.pi/6, np.pi/6, 20])]
        
            # we want nonlinear collision avoidance constraints
            #if iterate > 0:
                #prev_pos = [s_prev[k][id:id+6] for id in range(0, len(s_prev[k]), 6)]
                #curr_pos = [s_cvx[k][id:id+6] for id in range(0, len(s_cvx[k]), 6)]
                #for j in range(n_drones):
                 #   if j != i:
                  #      constraints+= [cvx.norm(prev_pos[i][0:3]-prev_pos[j][0:3]) + \
                   #                 (prev_pos[i][0:3].T-prev_pos[j][0:3].T)/cvx.norm(prev_pos[i][0:3]-prev_pos[j][0:3]) \
                    #                * (curr_pos[i][0:3]-curr_pos[j][0:3]) >= COLLISION_RADIUS]
            #ρ=        
    
      

    objective += cvx.quad_form(s_cvx[-1,:] - s_goal, P)
    

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()  
  
    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)

    s = s_cvx.value
    u = u_cvx.value

    obj = prob.objective.value

    return s, u, obj