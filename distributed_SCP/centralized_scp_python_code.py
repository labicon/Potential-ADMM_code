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
def solve_quadratic_program(H, Ain_total, Aeqtot):
    H_sparse = csr_matrix(H)
    Ain_total_sparse = csr_matrix(Ain_total)
    Aeqtot_sparse = csr_matrix(Aeqtot)

    result = quadratic_assignment(H_sparse, Ain_total_sparse)

    atot, f0, exitflag = result[0], result[1], result[3]

    if (len(atot) == 0 or exitflag == 0):
        p = []
        v = []
        a = []
        success = 0

    return result

success = exitflag
p, v, a = getStates(po, atot, A_p, A_v, K, N)
prev_p = p
criteria = abs(prev_f0 - f0)
prev_f0 = f0
k += 1


def centralizedSCP (po,pf,h,K,N,pmin,pmax,rmin,alim,A_p, A_v,E1,E2,order):
    prev_pos= initAllSolutions(po, pf, h, K)

    upper_bound= max_acceleration

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
        
            # linearized collision avoidance constraints
            if iterate > 0:
                prev_pos = [s_prev[k][id:id+6] for id in range(0, len(s_prev[k]), 6)]
                curr_pos = [s_cvx[k][id:id+6] for id in range(0, len(s_cvx[k]), 6)]
                for j in range(n_drones):
                    if j != i:
                        constraints+= [cvx.norm(prev_pos[i][0:3]-prev_pos[j][0:3]) + \
                                    (prev_pos[i][0:3].T-prev_pos[j][0:3].T)/cvx.norm(prev_pos[i][0:3]-prev_pos[j][0:3]) \
                                    * (curr_pos[i][0:3]-curr_pos[j][0:3]) >= COLLISION_RADIUS]

    
      

    objective += cvx.quad_form(s_cvx[-1,:] - s_goal, P)
    

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()  
  
    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)

    s = s_cvx.value
    u = u_cvx.value

    obj = prob.objective.value

    return s, u, obj