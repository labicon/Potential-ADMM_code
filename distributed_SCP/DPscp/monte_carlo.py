import numpy as np
import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import dpilqr as dec
from solve_scp import *

import logging
from pathlib import Path
import multiprocessing as mp
from os import getpid
import os
from time import strftime, perf_counter


def run_scp_rhc(i_trial, n_agents, n_states, n_inputs, N, dt, s_goal, s0, step_size, radius):

    count = 0

    P = 1e3*np.eye(n_agents*n_states)                    # terminal state cost matrix
    Q = np.eye(n_agents*n_states)*10  # state cost matrix
    R = 1e-3*np.eye(n_agents*n_inputs)                   # control cost matrix
    ρ = 200                

    u_try = np.tile(np.array([0, 0, 0, 4.9]), (1, n_agents))

    x_dims = [n_states]*n_agents
    fd = jax.jit(discretize(multi_Quad_Dynamics, dt, x_dims))

    u_bar = np.tile(u_try ,(N,1))
    s_bar = np.zeros((N + 1, n_agents*n_states))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k])

    X_trj =  np.zeros((0, n_agents*n_states))  #Full trajectory over entire problem horizon (not just a single prediction horizon)
    U_trj =  np.zeros((0, n_agents*n_inputs))

    iterate = 0
    s_prev = None
    obj_prev = np.inf
    tol = 5e-1
    si = s0

    t_solve_list = []
    converged = False
    while not np.all(dec.distance_to_goal(si, s_goal.reshape(1,-1), n_agents ,n_states,n_d= 3) < 0.1) :
        try:
            t_solve_start = perf_counter()
            s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, si,  ρ, iterate, s_prev, n_agents, radius)
            t_solve_per_step = perf_counter()-t_solve_start
            t_solve_list.append(t_solve_per_step)

        except RuntimeError:

            print('current trial failed')
            objective_val = None
            t_solve_step_avg = None
            break

        s_prev = s
        
        diff_obj = np.abs(obj - obj_prev)
        print(f'current diff_obj is {diff_obj}')
        if diff_obj < tol:
                
            print('SCP converged')
            converged = True
            break
        
        else:
            obj_prev = obj
            #Re-initialize nominal trajectory to shift prediction horizon
            s_bar = np.zeros((N + 1, n_agents*n_states))
            s_bar[0] = s[step_size]
            u_bar = np.tile(u_try ,(N,1))
            u_bar[0] = u[step_size-1]

            count +=1
                
            print(f'current objective value is {obj}!\n')

            X_trj = np.r_[X_trj, s[:step_size]]
            U_trj = np.r_[U_trj, u[:step_size]]
            print(f'X_trj has shape {X_trj.shape}\n')

            if count >=60:
                print('max iteration reached')
                break
    
    if converged:
        objective_val = objective(X_trj,U_trj[1:],s_goal, N, Q, R, P)  
        t_solve_step_avg = np.mean(t_solve_list)
        distance_to_goal = dec.distance_to_goal(X_trj[-1], s_goal, n_agents, n_states, n_d=3)
        
    else:
        objective_val = None
        t_solve_step_avg = None
        distance_to_goal = None
        
    logging.info(
            f'{i_trial},'
            f'{n_agents},{converged},'
            f'{objective_val},{N},{dt},{radius},{t_solve_step_avg},'
            f'{distance_to_goal},'
        )

    return X_trj, U_trj


def multi_agent_run(i_trial ,n_agents, n_states, n_inputs, N, dt, step_size, radius):
    
    x0,xf = dec.random_setup(n_agents, n_states, n_d = 3, energy = n_agents*1.5)
    s0 = x0.squeeze()
    s_goal = xf.squeeze()
    

    P = 1e3*np.eye(n_agents*n_states)                    # terminal state cost matrix
    Q = np.eye(n_agents*n_states)*10  # state cost matrix
    R = 1e-3*np.eye(n_agents*n_inputs)                   # control cost matrix
    ρ = 200                

    X_trj, U_trj = run_scp_rhc(i_trial, n_agents, n_states, n_inputs, N, dt, s_goal, s0, step_size, radius)


def setup_logger():
    
    # if centralized == True:
        
    LOG_PATH = Path(__file__).parent.parent / "logs"
    LOG_FILE = LOG_PATH / strftime(
        "rhc-scp-_%m-%d-%y_%H.%M.%S_{getpid()}.csv"
    )
    if not LOG_PATH.is_dir():
        LOG_PATH.mkdir()
        
    print(f"Logging results to {LOG_FILE}")
    logging.basicConfig(filename=LOG_FILE, format="%(message)s", level=logging.INFO)
    
    logging.info(
        "i_trial,n_agents,converged,objective_val,N,dt,radius,\
        t_solve_step, dist_to_goal"
    )


def monte_carlo_analysis():
    """Benchmark to evaluate algorithm over many random initial conditions"""

    setup_logger()

    n_trials_iter = range(30)

    n_agents_iter = [3,4,5,6,7,8,9,10]
    # n_agents_iter = [8,9,10]

    dt = 0.1
    N = 10
    radius = 0.35
    n_states = 12
    n_inputs = 4
    step_size = 1
    # Change the for loops into multi-processing?
    i_trial = 0
    for n_agents in n_agents_iter:
        print(f"\tn_agents: {n_agents}")
        if n_agents >=5 and n_agents <=8:
            radius = 0.15

        if n_agents >= 9:
            radius = 0.1
            
        for i_trial in n_trials_iter:
            print(f"\t\ttrial: {i_trial}")
            
            multi_agent_run(
                    i_trial, n_agents, n_states, 
                    n_inputs, N, dt, 
                    step_size, radius)
            
            i_trial +=1


def main():
    
    monte_carlo_analysis()


if __name__ == "__main__":
    main()

