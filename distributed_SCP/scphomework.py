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
def linearize(fd: callable,
              s: jnp.ndarray,
              u: jnp.ndarray):
    """Linearize the dynamics function `fd(s,u)` around nominal `(s,u)`."""

    # Use JAX to linearize `fd` around `(s,u)`.

    A = jax.jacfwd(fd,0)(s,u)
    B = jax.jacfwd(fd,1)(s,u)
    c = fd(s,u)-A@s-B@u

    return A, B, c

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
                      max_iters: int):
    """Solve the cart-pole swing-up problem via SCP. This function is used for one-shot optimization"""
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize nominal trajectories
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
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

    return s, u


def scp_iteration(fd: callable, P: np.ndarray, Q: np.ndarray, R: np.ndarray,
                  N: int, s_bar: np.ndarray, u_bar: np.ndarray,
                  s_goal: np.ndarray, s0: np.ndarray,
                  ru: float, ρ: float):
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

      for id in range(n_drones):

        constraints += [-np.array([-np.pi/6, -np.pi/6, 0]) <= u_cvx[k,id*3:(id+1)*3], \
                        u_cvx[k,id*3:(id+1)*3] <= np.array([np.pi/6, np.pi/6, 20])]
    
      #Note: without the following convex trust regions, the solution blows up 
      #just after a few iterations
        # constraints += [cvx.pnorm(s_cvx[k,id*3:(id+1)*3]-s_bar[k,id*3:(id+1)*3],'inf') <= ρ]
        # constraints += [cvx.pnorm(u_cvx[k,id*3:(id+1)*3]-u_bar[k,id*3:(id+1)*3],'inf') <= ρ] 

    objective += cvx.quad_form(s_cvx[-1,:] - s_goal, P)
    

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()  
  
    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)

    s = s_cvx.value
    u = u_cvx.value

    obj = prob.objective.value

    return s, u, obj

# def Quad_Dynamics(s, u):

#     x, y, z, vx, vy, vz = s

#     theta, phi, tau = u

#     x_d = jnp.array([
#         vx,
#         vy,
#         vz,
#         g*jnp.tan(theta),
#         -g*jnp.tan(phi),
#         tau-g

#     ])
    
#     return x_d

def multi_Quad_Dynamics(s, u):

    num_quadrotors = len(x_dims)

     # Define the constants of the quadrotor dynamics
    g = 9.81
    
    # Split the state `s` and control input `u` into individual components
    xs = [s[i:i+6] for i in range(0, len(s), 6)]
    us = [u[i:i+3] for i in range(0, len(u), 3)]
    
    x_ds = []
    for i in range(num_quadrotors):
        x, y, z, vx, vy, vz = xs[i]
        theta, phi, tau = us[i]
        
        x_d = jnp.array([
            vx,
            vy,
            vz,
            g*jnp.tan(theta),
            -g*jnp.tan(phi),
            tau-g
        ])
        
        x_ds.append(x_d)
    
    # print(f'shape is {jnp.concatenate(x_ds).shape}')
    return jnp.concatenate(x_ds)

def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator


"""Run SCP in centralized receding-horizon:"""

n_agents = 2
n_states = 6
n_controls = 3
g = 9.81
n = 12                                # state dimension
m = 6                                # control dimension
s_goal = np.array([-1.5, -1.5, 1.2, 0,  0 , 0, 2.5, 1.5, 1.5, 0, 0 , 0])  # desired state
s0 = np.array([0, 0, 0.8, 0, 0, 0, 1.5, 1.0, 1.0, 0, 0, 0])          # initial state
dt = 0.1                             # discrete time resolution
# T = 15.                              # total simulation time

# Dynamics
x_dims = [6, 6]
fd = jax.jit(discretize(multi_Quad_Dynamics,dt))

# SCP parameters

P = 1e3*np.eye(n)                    # terminal state cost matrix
Q = np.eye(n)*10  # state cost matrix
R = 1e-3*np.eye(m)                   # control cost matrix
ρ = 3.                               # trust region parameter
ru = 8.                              # control effort bound
tol = 5e-1                           # convergence tolerance
max_iters = 100                      # maximum number of SCP iterations

# Solve the swing-up problem with SCP
# t = np.arange(0., T + dt, dt)
# N = t.size - 1
N = 15
converged = False

count = 0
si = s0
obj_list = []

# scp_iteration(fd: callable, P: np.ndarray, Q: np.ndarray, R: np.ndarray,
#                   N: int, s_bar: np.ndarray, u_bar: np.ndarray,
#                   s_goal: np.ndarray, s0: np.ndarray,
#                   ru: float, ρ: float):


u_bar = np.zeros((N, m))
s_bar = np.zeros((N + 1, n))
s_bar[0] = s0
for k in range(N):
    s_bar[k+1] = fd(s_bar[k], u_bar[k])


X_trj =  np.zeros((0, n))  #Full trajectory over entire problem horizon (not just a single prediction horizon)
STEP_SIZE=1
"""TODO: the problem is blowing up: objective value grows unbounded somehow"""
while not np.all(dec.distance_to_goal(si, s_goal.reshape(1,-1), n_agents = 2,n_states = 6,n_d= 3) < 0.1) :
    count +=1
    s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, si, ru, ρ)
    obj_list.append(obj)
    
    print(f'current objective value is {obj}!')

    si = s[STEP_SIZE,:]
    X_trj = np.r_[X_trj, s[:STEP_SIZE]]

    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = si
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k])

    if count > 50:
        print('max. number of outer iterations reached!')
        break

# Simulate open-loop control
for k in range(N):
    s[k+1] = fd(s[k], u[k])

# Plot state and control trajectories
print(f'Full trajectory has shape {X_trj.shape}')
fig = plt.figure(dpi=200)
dec.plot_solve(X_trj,0,s_goal,x_dims,n_d = 3)
plt.savefig('2_quad_SCP.png')
