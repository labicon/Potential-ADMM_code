import cvxpy as cp
import matplotlib.pyplot as plt
import dpilqr
import numpy as np
from scipy import sparse
from tqdm import tqdm
import jax
import jax.numpy as jnp
from functools import partial
from time import perf_counter
from solve_scp import objective

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
                      ρ: float,
                      tol: float,
                      max_iters: int,
                      n_drones: int,
                      coll_radius: float):
    """This function is used for one-shot optimization"""
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize nominal trajectories
    u_bar_base = np.tile(np.array([0., 0., 0.]), (1, n_drones))  #initialze as hover condition
    # u_bar = np.zeros((N, m))
    u_bar = np.tile(u_bar_base,(N, 1))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    obj_prev = np.inf
    iterate = 0
    s_prev = None
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, s0,
                                 ρ, iterate, s_prev, n_drones, coll_radius)
        
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
            # ρ = ρ * 0.85
        
        s_prev = s

    if not converged:
        raise RuntimeError('SCP did not converge!')

    return s, u


def scp_iteration(fd: callable, P: np.ndarray, Q: np.ndarray, R: np.ndarray,
                  N: int, s_bar: np.ndarray, u_bar: np.ndarray,
                  s_goal: np.ndarray, s0: np.ndarray,
                  ρ: float, iterate: int, s_prev: np.ndarray, n_drones: int, collision_radius: float,
                  ):
    """Solve a single SCP sub-problem for the cart-pole swing-up problem."""
    A, B, c = linearize(fd, s_bar[:-1], u_bar)
    A, B, c = np.array(A), np.array(B), np.array(c)
    # A, B = linear_kinodynamics(dt, n_agent)
    print(f'current iteration is {iterate}')
    n = Q.shape[0]
    m = R.shape[0]
    
    s_cvx = cp.Variable((N + 1, n))
    u_cvx = cp.Variable((N, m))

    objective = 0.

    constraints = []
    constraints +=[s_cvx[0,:] == s_bar[0]]
    
    for k in range(N-1):
        objective += cp.quad_form(u_cvx[k+1,:]-u_cvx[k,:], np.eye(m))
    

    for k in range(N):
        
        objective += cp.quad_form(s_cvx[k,:] - s_goal, Q) + cp.quad_form(u_cvx[k,:], R) 
        constraints += [s_cvx[k+1,:] == A[k]@s_cvx[k,:] + B[k]@u_cvx[k,:] ]

        #Adding constraints for each quadrotor:
        if n_drones > 1:

            for i in range(n_drones):
                """Convex trust region"""
                # constraints += [cp.pnorm(s_cvx[k,i*6:(i+1)*6]-s_bar[k,i*6:(i+1)*6],'inf') <= ρ]
                # constraints += [cp.pnorm(u_cvx[k,i*3:(i+1)*3]-u_bar[k,i*3:(i+1)*3],'inf') <= ρ] 
                
                constraints += [np.array([-np.pi/6, -np.pi/6, -3]) <= u_cvx[k, i*3:(i+1)*3], \
                            u_cvx[k, i*3:(i+1)*3] <= np.array([np.pi/6, np.pi/6, 3])]
                
                # #state constraints:
                # constraints+= [np.array([-5., -5. , 0.9, -np.pi/3, -np.pi/3, -np.pi/3, -5., -5., -5., -1.5, -1.5, -1.5]) <= s_cvx[k, i*12:(i+1)*12],\
                #             s_cvx[k, i*12:(i+1)*12] <= np.array([5. , 5. , 4., np.pi/3, np.pi/3, np.pi/3, 5., 5., 5., 1.5, 1.5, 1.5])]
            
                #linearized collision avoidance constraints
                if iterate > 0:
                    prev_pos = [s_prev[k][id:id+6] for id in range(0, len(s_prev[k]), 6)]
                    curr_pos = [s_cvx[k][id:id+6] for id in range(0, len(s_prev[k]), 6)]
                    for j in range(n_drones):
                        if j != i:
                            constraints+= [cp.norm(prev_pos[i][0:3]-prev_pos[j][0:3], 1) + \
                                        (prev_pos[i][0:3].T-prev_pos[j][0:3].T)/cp.norm(prev_pos[i][0:3]-prev_pos[j][0:3], 1)
                                            @ (curr_pos[i][0:3]-curr_pos[j][0:3]) >= collision_radius]
                            
        else: #in case we want to test our code with a single drone
   
            constraints += [np.array([-np.pi/6, -np.pi/6, -3]) <= u_cvx[k, :], \
                            u_cvx[k, :] <= np.array([np.pi/6, np.pi/6, 3])]
       
            constraints += [cp.pnorm(s_cvx[k,:]-s_bar[k,:],'inf') <= ρ]
            constraints += [cp.pnorm(u_cvx[k,:]-u_bar[k,:],'inf') <= ρ] 


    objective += cp.quad_form(s_cvx[-1,:] - s_goal, P)
    print(f'total number of constraints is {len(constraints)}')
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(verbose = True)  
    
    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)

    s = s_cvx.value
    u = u_cvx.value

    obj = prob.objective.value

    return s, u, obj



def linear_kinodynamics(dt,n_agent):
    #Decision vector is a = [a_x, a_y, a_z]
    #State vector is X = [p_x, p_y, p_z, v_x, v_y, v_z]
    #Discretization time step is dt
    A_tot = sparse.lil_matrix((6*n_agent, 6*n_agent))
    B_tot = sparse.lil_matrix((6*n_agent, 3*n_agent))
    A = sparse.csc_matrix([[1, 0, 0, dt, 0, 0],
                           [0, 1, 0, 0 , dt ,0],\
                           [0, 0, 1, 0, 0 , dt],\
                           [0, 0, 0, 1, 0 ,0],\
                           [0, 0, 0, 0, 1 ,0],\
                           [0, 0, 0, 0, 0, 1]])
    B = sparse.csc_matrix([[dt**2/2, 0, 0],\
                           [0, dt**2/2, 0],\
                           [0, 0, dt**2/2],\
                           [dt, 0, 0 ],\
                           [0, dt , 0],\
                           [0, 0, dt]])

    for i in range(n_agent):
        A_tot[i*6:(i+1)*6,i*6:(i+1)*6] = A
        B_tot[i*6:(i+1)*6,i*3:(i+1)*3] = B
        
    
    return A_tot, B_tot


def single_quad_dynamics(s, u, x_dims):
    #x_dims is just a place holder to make it consistent with multi_Quad_Dynamics
    g = 9.81
    theta, phi, tau = u[0], u[1], u[2]
    px_dot = s[3]
    py_dot = s[4]
    pz_dot = s[5]
    vx_dot = g * jnp.tan(theta)
    vy_dot = -g *  jnp.tan(phi)
    vz_dot = tau - g

    x_d = jnp.array([
        px_dot,
        py_dot,
        pz_dot,
        vx_dot,
        vy_dot,
        vz_dot
    ])
        
    return x_d

def multi_Quad_Dynamics(s, u, x_dims):
    """Constants such as the mass and half-body length, etc, are hard coded into the model below"""

    num_quadrotors = len(x_dims)
    g = 9.81
    n_states = 6
    n_inputs = 3
    # Split the state `s` and control input `u` into indiviual components
    xs = [s[i:i+n_states] for i in range(0, len(s), n_states)]
    us = [u[i:i+n_inputs] for i in range(0, len(u), n_inputs)]
    
    x_ds = []
    for i in range(num_quadrotors):
        theta, phi, tau= us[i][0], us[i][1], us[i][2]
   
        px_dot = xs[i][3]
        py_dot = xs[i][4]
        pz_dot = xs[i][5]
        vx_dot = g * jnp.tan(theta)
        vy_dot = -g *  jnp.tan(phi)
        vz_dot = tau - g

        x_d =  jnp.array([
        px_dot,
        py_dot,
        pz_dot,
        vx_dot,
        vy_dot,
        vz_dot
    ])
        
        x_ds.append(x_d)

    return jnp.concatenate(x_ds)

def discretize(f, dt, x_dims):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u, x_dims)
        k2 = dt * f(s + k1 / 2, u, x_dims)
        k3 = dt * f(s + k2 / 2, u, x_dims)
        k4 = dt * f(s + k3, u, x_dims)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator

def run_scp_rhc(n_agents, n_states, n_inputs, N, dt, s_goal, s0, step_size):

    count = 0

    obj_list = []

    P = 1e3*np.eye(n_agents*n_states)                    # terminal state cost matrix
    Q = np.eye(n_agents*n_states)*10  # state cost matrix
    R = 1e-3*np.eye(n_agents*n_inputs)                   # control cost matrix
    ρ = 200                

    u_try = np.tile(np.array([0., 0., 0.]), (1, n_agents))

    x_dims = [n_states]*n_agents
    fd = jax.jit(discretize(multi_Quad_Dynamics, dt, x_dims))

    u_bar = np.tile(u_try ,(N,1))
    s_bar = np.zeros((N + 1, n_agents*n_states))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k])

    X_trj =  np.zeros((0, n_agents*n_states))  #Full trajectory over entire problem horizon (not just a single prediction horizon)
    U_trj =  np.zeros((0, n_agents*n_inputs))

    i_trial = 0
    iterate = 0
    s_prev = None
    obj_prev = np.inf
    tol = 5e-1
    radius = 0.35
    si = s0

    t_solve_list = []
    converged = False
    while not np.all(dpilqr.distance_to_goal(si, s_goal.reshape(1,-1), n_agents ,n_states,n_d= 3) < 0.1) :
        try:
            t_solve_start = perf_counter()
            s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, si,  ρ, iterate, s_prev, n_agents, radius)
            t_solve_per_step = perf_counter()-t_solve_start
            t_solve_list.append(t_solve_per_step)
            
        except RuntimeError:

            print('current trial failed')
            i_trial +=1
            
        s_prev = s
        
        diff_obj = np.abs(obj - obj_prev)
        print(f'current diff_obj is {diff_obj}')
        if diff_obj < tol:
                
            print('SCP converged')
            i_trial +=1
            converged = True
            break
        
        else:
            obj_prev = obj
            #Re-initialize nominal trajectory to shift prediction horizon
            s_bar = np.zeros((N + 1, n_agents*n_states))
            s_bar[0] = s[step_size]
            si = s_bar[0]
            u_bar = np.tile(u_try ,(N,1))
            u_bar[0] = u[step_size-1]

            count +=1
            
            obj_list.append(obj)
            
            print(f'current objective value is {obj}!\n')

            X_trj = np.r_[X_trj, s[:step_size]]
            U_trj = np.r_[U_trj, u[:step_size]]
            print(f'X_trj has shape {X_trj.shape}\n')

            if count >=60:
                print('max iteration reached')
                i_trial +=1
                break
    
    # if converged:
    #     objective_val = objective(X_trj,U_trj[1:],s_goal, N, Q, R, P)  
    #     t_solve_step = np.mean(t_solve_list)
    #     distance_to_goal = dpilqr.distance_to_goal(X_trj[-1], s_goal, n_agents, n_states, n_d=3)
    
    # else:
    #     objective_val = None
    #     t_solve_step = None

    return X_trj, U_trj




if __name__ == '__main__':
    
    N = 10

    cost = 0
    constr = []
    
    n_states = 6
    n_inputs = 3
    x0 = np.array([0, 0, 0.5, 0.01, 0.01, 0.01, 1.2, 0.7, 0.45, -0.01, -0.01, -0.01])
    xr = np.array([1.5, 1.0, 1.1, 0., 0., 0., -1, -0.4, 0.6, 0, 0, 0])

    nsim = 100
    radius = 0.5
    n_agents = 2
    nx = n_agents * n_states
    nu = n_agents * n_inputs
    X_full = np.zeros((0,nx))
    X_full = np.r_[X_full,x0.reshape(1,-1)]
    xi = x0.flatten()
    xr = xr.flatten()
    # Ad,Bd = linear_kinodynamics(0.1, n_agents)


    Q = np.diag([10., 10., 10., 10., 10., 10.]*n_agents)
    QN = Q*10
    P = 1e3*np.eye(nx) 
    R = 0.1*np.eye(nu)

    ρ = 3.                               # trust region parameter
    tol = 5e-1                           # convergence tolerance
    max_iters = 100                      # maximum number of SCP iterations

    iter = 0
    x_dims = [n_states]*n_agents
    n_dims = [3]*n_agents
    
    dt = 0.1
    
    X_trj, U_trj = run_scp_rhc(2, 6, 3, N, dt, xr, x0, 1)
    """The following QP method fails; switching to sequential convex programming!"""
    # while not np.all(dpilqr.distance_to_goal(xi,xr,n_agents,6,3) < 0.1) and (iter < nsim):
    # for _ in range(nsim):
        # x = cp.Variable((N+1,nx))
        # u = cp.Variable((N,nu))
        # cost = 0
        # constr = []
        
        # for t in range(N):

        #     cost += cp.quad_form(x[t+1,:],Q) + cp.quad_form(u[t,:],R)
    
        #     # # Linearized collision avoidance constraints
        #     if iter > 0:
        #         prev_pos = [x_prev[t][id:id+6] for id in range(0, len(x_prev[t]), 6)]
        #         curr_pos = [x[t][id:id+6] for id in range(0, len(x_prev[t]), 6)]
        #         for i in range(n_agents):
        #             for j in range(n_agents):
        #                 if j != i:
        #                     # for k in range(N+1):
        #                         # constr += [cp.norm(x_prev[k,j*n_states:j*n_states+3]-x_prev[k,i*n_states:i*n_states+3], 1) + \
        #                         #         (x_prev[k,j*n_states:j*n_states+3].T- x_prev[k,i*n_states:i*n_states+3].T)/cp.norm(x_prev[k,j*n_states:j*n_states+3]\
        #                         #         -x_prev[k,i*n_states:i*n_states+3], 1)@\
        #                         #         (x[k,j*n_states:j*n_states+3]-x[k,i*n_states:i*n_states+3]) >= radius]
        #                     constr += [cp.norm(prev_pos[i][0:3]-prev_pos[j][0:3], 1) + \
        #                             (prev_pos[i][0:3].T-prev_pos[j][0:3].T)/cp.norm(prev_pos[i][0:3]-prev_pos[j][0:3], 1)
        #                                 @ (curr_pos[i][0:3]-curr_pos[j][0:3]) >= radius]
            
        #     constr += [x[t + 1,:] == Ad @ x[t,:] + Bd @ u[t, :]]
        #     constr += [u[t,:] <= np.tile(np.array([5, 5, 5]),(n_agents,)) ]
        #     constr += [np.tile(np.array([-5, -5, -5]),(n_agents,)) <= u[t, :]]
        #     # constr += [x[t, :] <= np.tile(np.array([np.inf, np.inf, np.inf, 2, 2, 2]),(n_agents,))]
        #     # constr += [np.tile(-np.array([np.inf, np.inf, np.inf, 2, 2, 2]),(n_agents,)) <= x[t,:]]

        # cost += cp.quad_form((x[-1,:]-xr.flatten()),Q*1000)    
        # constr += [x[0, :] == xi]

        # problem = cp.Problem(cp.Minimize(cost), constr)
        # problem.solve(verbose=True)

        # s, u = solve_scp(fd, P, Q, R, N, xr, xi, ρ, tol, max_iters, n_agents, 0.35)
        
        # u_trj = u
        # x_trj = s
        # x_prev = x_trj
        # ctrl = u_trj[1]
        # xi = Ad@xi + Bd@ctrl
        # # X_full[iter+1,:] = xi
        # X_full = np.r_[X_full,xi.reshape(1,-1)]
        
        # iter +=1
        
        
        
    """Plot trajectory"""
    plt.figure(dpi=150)
    dpilqr.plot_solve(X_trj,0,xr,x_dims,True,3)
    # plt.show()
    plt.savefig('single_agent_QP.png')
    
    """Plot pair-wise distances"""
    plt.figure(dpi=150)
    plt.plot(dpilqr.compute_pairwise_distance(X_trj,x_dims,3))
    plt.hlines(0.35, 0, X_trj.shape[0], 'r')
    plt.savefig('Pairwise_distance(2_agents).png')