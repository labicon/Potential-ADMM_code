import numpy as np
import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
#import dpilqr as dec


def objective (s, u, s_goal, N, Q, R, P): #calculates the objective function 
    m= R.shape[0]
    n=Q.shape[0]
    u= cvx.Variable(N, m)
    obj=0
    for k in range (0, N-1): #cost for each time step
        c= (s[k, :]- s_goal).T@Q@(s[k, :]-s_goal)
        c2= u[k, :].T@R@u[k, :]
        c3= s[-1, :]-s_goal.T@P@(s[-1, :]-s_goal)
        
        obj=c+c2+c3
        
    return obj

def linearize (fd: callable, s, u): #makes the A, B, C matrices
    A= jax.jacfwd(fs, 0)(s, u)
    B= jax.jacfwd(fs, 1)(s, u)
    C= fd(s, u) - A@s -B@u
    
    return A, B, C

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

    u_bar_base = np.zeros((n_drones, 4))
    u_bar_base[:, -1] = 4.9
    
    u_bar_base = np.array([0, 0, 0, 4.9])  
    u_bar = np.broadcast_to(u_bar_base, (N, n_drones, 4))
    
    u_bar_reshaped = u_bar.reshape(N * n_drones, 4) #look over this line
    
    s_bar = np.zeros((N + 1, n_drones, 12)) 

    
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
                  ρ: float, iterate: int, s_prev: np.ndarray, n_drones: int, collision_radius: float):
    """Solve a single SCP sub-problem for the cart-pole swing-up problem."""
    A, B, c = linearize(fd, s_bar[:-1], u_bar)
    A, B, c = np.array(A), np.array(B), np.array(c)
    print(f'current iteration is {iterate}')
    n = Q.shape[0]
    m = R.shape[0]
    
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    objective = 0.

    constraints = []
    constraints +=[s_cvx[0,:] == s_bar[0]]

    for k in range(N):
        objective += cvx.quad_form(s_cvx[k,:] - s_goal, Q) + cvx.quad_form(u_cvx[k,:], R) 

        constraints += [s_cvx[k+1,:] == A[k]@s_cvx[k,:] + B[k]@u_cvx[k,:] + c[k]]

        constraints += [-ru <= u_cvx[k,:], u_cvx[k,:] <= ru]
        
        
        if n_drones>1:
            for i in range (0, n_drones):
                constraints += [cvx.pnorm(s_cvx[k,i*6:(i+1)*6]-s_bar[k,i*6:(i+1)*6],'inf') <= ρ]
                constraints += [cvx.pnorm(u_cvx[k,i*4:(i+1)*4]-u_bar[k,i*4:(i+1)*4],'inf') <= ρ] 
                constraints += [np.array([-1, -1, -1, 0]) <= u_cvx[k, i*4:(i+1)*4], u_cvx[k, i*4:(i+1)*4] <= np.array([1, 1, 1, 10])]
                

            if iterate>0: 
                for j in range(n_drones):
                    for i in range (n_drones):
                        if j != i:
                            prev_pos_i= s_prev[k][i*6:(i+1)*6][:3]
                            prev_pos_j= s_prev[k][j*6:(j+1)*6][:3]
                            curr_pos_i= s_cvx[k][i*6:(i+1)*6][:3]
                            curr_pos_j= s_cvx[k][j*6:(j+1)*6][:3]

                            distance= cvx.norm(prev_pos_i - prev_pos_j, 1)
                            relative_velocity= (prev_pos_i - prev_pos_j) / distance @ (curr_pos_i - curr_pos_j)

                            constraints+= [distance + relative_velocity >= collision_radius]

            else: 
                constraints += [np.array([ -1, -1, -1, 0]) <= u_cvx[k, :], \
                            u_cvx[k, :] <= np.array([1, 1, 1, 10])]
                constraints += [cvx.pnorm(s_cvx[k,:]-s_bar[k,:],'inf') <= ρ]
                constraints += [cvx.pnorm(u_cvx[k,:]-u_bar[k,:],'inf') <= ρ] 
    
    
        objective += cvx.quad_form(s_cvx[-1,:] - s_goal, P)
    
    print(f'total number of constraints is {len(constraints)}')
    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve(verbose = True)  
    
    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)

    s = s_cvx.value
    u = u_cvx.value

    obj = prob.objective.value

    return s, u, obj
    


def single_quad_dynamics(s, u, x_dims):
    #x_dims is just a place holder to make it consistent with multi_Quad_Dynamics
    tau_x, tau_y, tau_z, f_z = u[0], u[1], u[2], u[3]
    psi = s[3]
    theta = s[4]
    phi = s[5]
    v_x = s[6]
    v_y = s[7]
    v_z = s[8]
    w_x = s[9]
    w_y = s[10]
    w_z = s[11]

    #model is derived after plugging into physical parameters found via sys ID
    #see https://github.com/tbretl/ae353-sp22/blob/main/projects/04_drone/DeriveEOM-Template.ipynb
    x_d = jnp.array([
        v_x*jnp.cos(psi)*jnp.cos(theta) + v_y*(jnp.sin(phi)*jnp.sin(theta)*jnp.cos(psi) - jnp.sin(psi)*jnp.cos(phi)) + v_z*(jnp.sin(phi)*jnp.sin(psi) + jnp.sin(theta)*jnp.cos(phi)*jnp.cos(psi)),
        v_x*jnp.sin(psi)*jnp.cos(theta) + v_y*(jnp.sin(phi)*jnp.sin(psi)*jnp.sin(theta) + jnp.cos(phi)*jnp.cos(psi)) - v_z*(jnp.sin(phi)*jnp.cos(psi) - jnp.sin(psi)*jnp.sin(theta)*jnp.cos(phi)), 
        -v_x*jnp.sin(theta) + v_y*jnp.sin(phi)*jnp.cos(theta) + v_z*jnp.cos(phi)*jnp.cos(theta), 
        (w_y*jnp.sin(phi) + w_z*jnp.cos(phi))/jnp.cos(theta), 
        w_y*jnp.cos(phi) - w_z*jnp.sin(phi), 
        w_x + w_y*jnp.sin(phi)*jnp.tan(theta) + w_z*jnp.cos(phi)*jnp.tan(theta), 
        v_y*w_z - v_z*w_y + 981.*jnp.sin(theta)/100., 
        -v_x*w_z + v_z*w_x - 981.*jnp.sin(phi)*jnp.cos(theta)/100., 
        2*f_z + v_x*w_y - v_y*w_x - 981.*jnp.cos(phi)*jnp.cos(theta)/100., 
        10000.*tau_x/23. - 17.*w_y*w_z/23., 
        10000.*tau_y/23. + 17.*w_x*w_z/23., 
        250.*tau_z
    ])
        
    return x_d

def multi_Quad_Dynamics(s, u, x_dims):
    """Constants such as the mass and half-body length, etc, are hard coded into the model below"""

    num_quadrotors = len(x_dims)

    g = 9.81
    n_states = 12
    n_inputs = 4
    # Split the state `s` and control input `u` into indiviual components
    xs = [s[i:i+n_states] for i in range(0, len(s), n_states)]
    us = [u[i:i+n_inputs] for i in range(0, len(u), n_inputs)]
    
    x_ds = []
    for i in range(num_quadrotors):
        
        tau_x, tau_y, tau_z, f_z = us[i][0], us[i][1], us[i][2], us[i][3]
        psi = xs[i][3]
        theta = xs[i][4]
        phi = xs[i][5]
        v_x = xs[i][6]
        v_y = xs[i][7]
        v_z = xs[i][8]
        w_x = xs[i][9]
        w_y = xs[i][10]
        w_z = xs[i][11]

        x_d = jnp.array([
        v_x*jnp.cos(psi)*jnp.cos(theta) + v_y*(jnp.sin(phi)*jnp.sin(theta)*jnp.cos(psi) - jnp.sin(psi)*jnp.cos(phi)) + v_z*(jnp.sin(phi)*jnp.sin(psi) + jnp.sin(theta)*jnp.cos(phi)*jnp.cos(psi)),
        v_x*jnp.sin(psi)*jnp.cos(theta) + v_y*(jnp.sin(phi)*jnp.sin(psi)*jnp.sin(theta) + jnp.cos(phi)*jnp.cos(psi)) - v_z*(jnp.sin(phi)*jnp.cos(psi) - jnp.sin(psi)*jnp.sin(theta)*jnp.cos(phi)), 
        -v_x*jnp.sin(theta) + v_y*jnp.sin(phi)*jnp.cos(theta) + v_z*jnp.cos(phi)*jnp.cos(theta), 
        (w_y*jnp.sin(phi) + w_z*jnp.cos(phi))/jnp.cos(theta), 
        w_y*jnp.cos(phi) - w_z*jnp.sin(phi), 
        w_x + w_y*jnp.sin(phi)*jnp.tan(theta) + w_z*jnp.cos(phi)*jnp.tan(theta), 
        v_y*w_z - v_z*w_y + 981.*jnp.sin(theta)/100., 
        -v_x*w_z + v_z*w_x - 981*jnp.sin(phi)*jnp.cos(theta)/100., 
        2*f_z + v_x*w_y - v_y*w_x - 981.*jnp.cos(phi)*jnp.cos(theta)/100., 
        10000.*tau_x/23. - 17.*w_y*w_z/23., 
        10000.*tau_y/23. + 17.*w_x*w_z/23., 
        250.*tau_z
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


