import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import casadi as cs
import dpilqr
import cvxpy as cp
from time import perf_counter
import sys

import logging
from pathlib import Path
import multiprocessing as mp
from os import getpid
import os
from time import strftime

from solvers.util import (
    compute_pairwise_distance_nd_Sym,
    define_inter_graph_threshold,
    distance_to_goal,
    split_graph, 
    objective,
)

from solvers import util
from multiprocessing import Process, Pipe
import cvxpy as cp
from dynamics import linear_kinodynamics


"""Centralized Potential-ADMM"""

def solve_iteration(n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf):
    """Define constants"""

    nx = n_states*n_agents
    nu = n_inputs*n_agents
    N = n_agents
    Ad, Bd = linear_kinodynamics(0.1, N)
    x_dims = [n_states] * N
    n_dims = [3, 3, 3]
    
    """Creating empty dicts to hold Casadi variables for each worker machine"""
    f_list = {}
    d = {} 
    states = {}

    for id in range(N):
        d["opti_{0}".format(id)] = Opti()
        
        #Augmented state : Y = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        
        states["Y_{0}".format(id)] = d[f"opti_{id}"].variable((T+1)*nx + T* nu)
        cost = 0
    
        #Quadratic tracking cost
        
        for t in range(T):
            for idx in range(nx):
                cost += (states[f"Y_{id}"][:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) *  \
                Q[idx,idx]* (states[f"Y_{id}"][:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) 
            for idu in range(nu):
                cost += (states[f"Y_{id}"][(T+1)*nx:][t*nu:(t+1)*nu][idu]) *  \
                R[idu,idu] * (states[f"Y_{id}"][(T+1)*nx:][t*nu:(t+1)*nu][idu])
        
        for idf in range(nx):
            cost += (states[f"Y_{id}"][:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf]) * \
            Qf[idf,idf] * (states[f"Y_{id}"][:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf])

        # f_list.append(cost)
        f_list["cost_{0}".format(id)] = cost

    def run_worker(agent_id, cost, pipe):
        xbar = d[f"opti_{agent_id}"].parameter((T+1)*nx + T*nu)
        d[f"opti_{agent_id}"].set_value(xbar, cs.GenDM_zeros((T+1)*nx + T*nu,1))    
        
        u = d[f"opti_{agent_id}"].parameter((T+1)*nx + T*nu)
        d[f"opti_{agent_id}"].set_value(u, cs.GenDM_zeros((T+1)*nx + T*nu,1))
        
        #This is the scaled Lagrange multiplier

        rho = 5
        cost += (rho/2)*sumsqr(states[f"Y_{agent_id}"] - xbar + u)
        
        # ADMM loop
        
        iter = 0
        while True:
            try:
                coll_cost = 0
                smooth_trj_cost = 0
                for k in range(T):
                
                    d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx] \
                                    == Ad @ states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx] \
                                        + Bd @ states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])

                    d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu] <= np.tile(np.array([3, 3, 3]),(N,)).reshape(-1,1))
                    d[f"opti_{agent_id}"].subject_to(np.tile(np.array([-3, -3, -3]),(N,)).reshape(-1,1) <= states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])
                
                    #Pair-wise Euclidean distance between each pair of agents
                    if n_agents > 1:
                        distances = util.compute_pairwise_distance_nd_Sym(states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx], x_dims, n_dims)
                        #Collision avoidance cost
                        for dist in distances:
                            # coll_cost += fmin(0,(dist - radius))**2 * 500
                            # coll_cost += fmin(0,(dist - 2*radius))**2 * 1000 #Works, but distnace falls below threshold
                            coll_cost += fmin(0,(dist - 2*radius))**2 * 1200
                    #Trajectory smoothing term
                    for ind in range(nx):
                        smooth_trj_cost += (states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx][ind]-\
                                            states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx][ind])**2
                    
                
                d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][0:nx] == x0) 
                
                cost_tot = cost + coll_cost/n_agents + smooth_trj_cost
                d[f"opti_{agent_id}"].minimize(cost_tot)
                
                d[f"opti_{agent_id}"].solver("ipopt")
                
                # if iter > 0:
                    # d[f"opti_{agent_id}"].set_initial(sol_prev.value_variables())
                
                sol = d[f"opti_{agent_id}"].solve()
                # print(f'paramete xbar has value {sol.value(xbar)}')
                # print(f'parameter u has value {sol.value(u)}')
                
                # sol_prev = sol
                pipe.send(sol.value(states[f"Y_{agent_id}"]))
                
                
                d[f"opti_{agent_id}"].set_value(xbar, pipe.recv()) #receive the averaged result from the main process.
                d[f"opti_{agent_id}"].set_value(u, sol.value( u + states[f"Y_{agent_id}"] - xbar))

                iter += 1
                print(f'Current iteration is {iter}')
                
                d[f"opti_{agent_id}"].subject_to()
                
                d[f"opti_{agent_id}"].set_initial(sol.value_variables()) #Warm start the next re-optimization
                
            except EOFError:
                print("Connection closed.")
                break
                    
    pipes = []
    procs = []
    for i in range(N):
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=run_worker, args=(i, f_list[f"cost_{i}"], remote))]
        procs[-1].start()

    MAX_ITER = 5
    solution_list = []
    iter = 0
    for i in range(MAX_ITER):
        # Gather and average xi

        xbar = sum(pipe.recv() for pipe in pipes)/N
        # print(f'xbar is {xbar}, has shape {xbar.shape}\n')

        solution_list.append(xbar)
        # print(f'average of xbar is {np.mean(xbar)}\n')

        # Scatter xbar
        for pipe in pipes:
            pipe.send(xbar)
            
        iter += 1

    [p.terminate() for p in procs]
        
        
    x_trj_converged = solution_list[-1][:(T+1)*nx].reshape((T+1,nx))
    u_trj_converged = solution_list[-1][(T+1)*nx:].reshape((T,nu))
        
    
    return x_trj_converged, u_trj_converged, iter


def solve_rhc(n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf, n_trial=None):
    centralized = True
    nx = n_states*n_agents
    nu = n_inputs*n_agents
    
    X_full = np.zeros((0, nx))
    U_full = np.zeros((0, nu))
    X_full = np.r_[X_full, x0.reshape(1,-1)]
    u_ref = np.array([0, 0, 0]*n_agents)
    
    x_curr = x0
    mpc_iter = 0
    ADMM_iters = []
    obj_history = [np.inf]
    solve_times = []
    t = 0
    dt = 0.1
    while not np.all(dpilqr.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
        t0 = perf_counter()
        x_trj_converged, u_trj_converged, iter = solve_iteration(n_states, n_inputs, n_agents, x_curr, \
                                                                 xr, T, radius, Q, R, Qf)
        solve_times.append(perf_counter()-t0)
        ADMM_iters.append(iter)
        obj_history.append(float(objective(x_trj_converged.T, u_trj_converged.T, u_ref, xr, Q, R, Qf)))
        
        x_curr = x_trj_converged[1]
        u_curr = u_trj_converged[0]
        
        X_full = np.r_[X_full, x_curr.reshape(1,-1)]
        U_full = np.r_[U_full, u_curr.reshape(1,-1)]
        
        mpc_iter += 1
        t += dt
        if mpc_iter > 50:
            print('Exiting MPC loops')
            converged = False
            break

    print(f'Final distance to goal is {dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)}')
    
    if np.all(dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
        converged = True
    
    obj_trj = float(objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf))
    
    logging.info(
    f'{n_trial},'
    f'{n_agents},{t},{converged},'
    f'{obj_trj},{T},{dt},{radius},{centralized},{np.mean(solve_times)},{distance_to_goal(X_full[-1].flatten(), xr, n_agents, n_states)},'
        )
    
    
    return X_full, U_full, \
            objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf), \
            np.mean(ADMM_iters), np.mean(solve_times)
        
        
        
def solve_distributed_rhc(ids, n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf, n_trial=None):
    
    n_dims = [3]*n_agents
    u_ref = np.array([0, 0, 0]*n_agents)
    x_dims = [n_states]*n_agents
    u_dims = [n_inputs]*n_agents
    
    nx = n_states*n_agents
    nu = n_inputs*n_agents
    X_full = np.zeros((0, nx))
    U_full = np.zeros((0, nu))
    X_full = np.r_[X_full, x0.reshape(1,-1)]
    
    distributed_mpc_iters =0
    solve_times = []
    x_curr = x0
    # obj_history = [np.inf]
    t = 0
    dt = 0.1
    converged = False
    centralized = False
    while not np.all(np.all(dpilqr.distance_to_goal(x_curr.flatten(), xr.flatten(), \
                                                    n_agents, n_states, 3) <= 0.1)):
        # rel_dists = util.compute_pairwise_distance_nd_Sym(x0,x_dims,n_dims)
        graph = util.define_inter_graph_threshold(x_curr, radius, x_dims, ids, n_dims)
        
        split_states_initial = split_graph(x_curr.T, x_dims, graph)
        split_states_final = split_graph(xr.T, x_dims, graph)
        split_inputs_ref = split_graph(u_ref.reshape(-1, 1).T, u_dims, graph)
        
        X_dec = np.zeros((nx, 1))
        U_dec = np.zeros((nu, 1))
        
        # X_trj = np.zeros((nx, T+1))
        # U_trj = np.zeros((nu, T))
        
        # t0 = perf_counter()
        for (x0_i, xf_i, u_ref_i , (prob, ids_) , place_holder) in zip(split_states_initial, 
                                       split_states_final,
                                       split_inputs_ref,
                                       graph.items(),
                                       range(len(graph))):
            print(f'Current sub-problem has {x0_i.size//n_states} agents \n')
            t0 = perf_counter()
            x_trj_converged_i, u_trj_converged_i, _ = solve_iteration(n_states, n_inputs, \
                                                                      x0_i.size//n_states,\
                                                                      x0_i.reshape(-1,1), \
                                                                      xf_i.reshape(-1,1), \
                                                                      T, radius, 
                                                                      Q[:x0_i.size,:x0_i.size], 
                                                                      R[:u_ref_i.size,:u_ref_i.size], 
                                                                      Qf[:x0_i.size,:x0_i.size])
            solve_times.append(perf_counter() - t0)
            i_prob = ids_.index(prob)
            
            #Collecting solutions from different potential game sub-problems at current time step K:
            X_dec[place_holder * n_states : (place_holder + 1) * n_states, :] = x_trj_converged_i[
                1, i_prob * n_states : (i_prob + 1) * n_states
                ].reshape(-1,1)
            
            U_dec[place_holder * n_inputs : (place_holder + 1) * n_inputs, :] = u_trj_converged_i[
                0, i_prob * n_inputs : (i_prob + 1) * n_inputs
                ].reshape(-1,1)
            
        #     X_trj[place_holder * n_states : (place_holder + 1) * n_states, :] = x_trj_converged_i[
        #         :, i_prob * n_states : (i_prob + 1) * n_states
        #         ].T
            
        #     U_trj[place_holder * n_inputs : (place_holder + 1) * n_inputs, :] = u_trj_converged_i[
        #         :, i_prob * n_inputs : (i_prob + 1) * n_inputs
        #         ].T
            
        # obj_history.append(float(objective(X_trj, U_trj, u_ref, xr, Q, R, Qf)))    
        t += dt
        
        x_curr = X_dec
        
        X_full = np.r_[X_full, X_dec.reshape(1,-1)]
        U_full = np.r_[U_full, U_dec.reshape(1,-1)]
        
        distributed_mpc_iters += 1
        
        if distributed_mpc_iters > 50:
            print(f'Max iters reached; exiting MPC loops')
            break
        
    if np.all(dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
        converged = True
        
    print(f'Final distance to goal is {dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)}')    
    
    obj_trj = float(objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf))
    
    logging.info(
    f'{n_trial},'
    f'{n_agents},{t},{converged},'
    f'{obj_trj},{T},{dt},{radius},{centralized},{np.mean(solve_times)},{distance_to_goal(X_full[-1].flatten(), xr, n_agents, n_states)},'
        )
    
    return X_full, U_full, objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf), np.mean(solve_times)
        

def setup_logger():
    
    # if centralized == True:
        
    LOG_PATH = Path(__file__).parent/ "logs"
    LOG_FILE = LOG_PATH / strftime(
        "ADMM-mpc-_%m-%d-%y_%H.%M.%S_{getpid()}.csv"
    )
    if not LOG_PATH.is_dir():
        LOG_PATH.mkdir()
    print(f"Logging results to {LOG_FILE}")
    logging.basicConfig(filename=LOG_FILE, format="%(message)s", level=logging.INFO)
    logging.info(
        "i_trial,n_agents,converged,obj_trj,T,dt,radius,\
         centralized,t_solve_avg,dist_to_goal"
    )
    
    
def multi_agent_run(trial, n_states,
                            n_inputs,
                            n_agents,
                            x0,
                            xr,
                            T,
                            radius,
                            Q,
                            R,
                            Qf):
    """simulation comparing the centralized and decentralized solvers"""
    
    if n_agents == 3:
        x0,xr = util.setup_3_quads()
        n_dims = [3]*3

    elif n_agents==4:
        x0, xr=util.setup_4_quads()
        n_dims=[3]*4

    elif n_agents == 5:
        x0,xr = util.setup_5_quads()
        n_dims = [3]*5

    elif n_agents==6:
        x0, xr=util.setup_6_quads()
        n_dims= [3]*6

    elif n_agents==7:
        x0, xr= util.setup_7_quads()
        n_dims= [3]*7

    elif n_agents==8:
        x0, xr= util.setup_8_quads()
        n_dims= [3]*8

    elif n_agents==9:
        x0, xr= util.setup_9_quads()
        n_dims= [3]*9

    elif n_agents == 10:
        x0,xr = util.setup_10_quads()
        n_dims = [3]*10
    
    ids = [100 + i for i in range(n_agents)]
    
    Q = np.diag([5., 5., 5., 1., 1., 1.]*n_agents)
    Qf = Q*500
    R = 0.1*np.eye(n_agents*n_inputs)

    X_full, U_full, obj, mean_iters, avg_SolveTime = solve_rhc(n_states,
                                                n_inputs,
                                                n_agents,
                                                x0,
                                                xr,
                                                T,
                                                radius,
                                                Q,
                                                R,
                                                Qf,
                                                trial)
    X_full, U_full, obj, avg_SolveTime = solve_distributed_rhc(ids,
                                                            n_states, 
                                                            n_inputs, 
                                                            n_agents, 
                                                            x0, 
                                                            xr, 
                                                            T, 
                                                            radius,
                                                            Q, 
                                                            R, 
                                                            Qf,
                                                            trial)


def monte_carlo_analysis():
    """Benchmark to evaluate algorithm over many random initial conditions"""

    setup_logger()

    n_trials_iter = range(30)

    n_agents_iter = [3,4,5,6,7,8,9,10]
    

    radius = 0.5
    
    # Change the for loops into multi-processing?

    for n_agents in n_agents_iter:
        print(f"\tn_agents: {n_agents}")
        # if n_agents >=5 and n_agents <=8:
        #     radius = 0.4
        
        # if n_agents >8 and n_agents <9:
        #     radius = 0.45

        # if n_agents >= 9:
        #     radius = 0.4
            
        for i_trial in n_trials_iter:
            print(f"\t\ttrial: {i_trial}")
            
            multi_agent_run(
                i_trial,n_states,
                            n_inputs,
                            n_agents,
                            x0,
                            xr,
                            T,
                            radius,
                            Q,
                            R,
                            Qf)    
            
            
if __name__ == "__main__":
    
    n_states = 6
    n_inputs = 3
    n_agents = 3
    x_dims = [n_states]*n_agents
    x0, xr = util.paper_setup_3_quads()
    T = 8
    radius = 0.5
    Q = np.diag([5., 5., 5., 1., 1., 1.]*n_agents)
    Qf = Q*500
    R = 0.1*np.eye(n_agents*n_inputs)
    
    ids = [100 + n for n in range(n_agents)] #Assigning random IDs for agents
    
    centralized = False
    logging = True
    
    if not logging:
    
        if centralized:
            X_full, U_full, obj, mean_iters, avg_SolveTime = solve_rhc(n_states,
                                                    n_inputs,
                                                    n_agents,
                                                    x0,
                                                    xr,
                                                    T,
                                                    radius,
                                                    Q,
                                                    R,
                                                    Qf)
        else:
            
            X_full, U_full, obj, avg_SolveTime = solve_distributed_rhc(ids,
                                                                n_states, 
                                                                n_inputs, 
                                                                n_agents, 
                                                                x0, 
                                                                xr, 
                                                                T, 
                                                                radius,
                                                                Q, 
                                                                R, 
                                                                Qf)
            
        print(f'The average solve time is {avg_SolveTime} seconds!')
    
        #Plot trajectory
        plt.figure(dpi=150)
        dpilqr.plot_solve(X_full, float(obj), xr, x_dims, True, 3)
        # plt.gca().set_zticks([0.8,1.2], minor=False)
        plt.legend(plt.gca().get_children()[1:3], ["Start Position", "Goal Position"])
        
        if centralized:
            plt.savefig('ADMM_mpc(centralized).png')
        
        else:
            plt.savefig('ADMM_mpc(decentralized).png')
        
        #Plot pairwise distance
        plt.figure(dpi=150)
        dpilqr.plot_pairwise_distances(X_full, x_dims, [3,3,3], radius)
        
        if centralized:
            plt.savefig('Pairwise_distances_ADMM(centralized).png')
            
        else:
            plt.savefig('Pairwise_distances_ADMM(decentralized).png')
        
    
    
    else:
        monte_carlo_analysis()
        

        