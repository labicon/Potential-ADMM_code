import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import casadi as cs
import dpilqr
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

from dpilqr.cost import GameCost, ProximityCost, ReferenceCost
from dpilqr.dynamics import (
    QuadcopterDynamics6D,
    MultiDynamicalModel,
)
from dpilqr.distributed import solve_rhc
from dpilqr.problem import ilqrProblem
from dpilqr.util import split_agents_gen, random_setup


from solvers import util
from multiprocessing import Process, Pipe
from dynamics import linear_kinodynamics

def solve_iteration(n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf, x_trj_init, state_prev, local_iter, MAX_ITER = 5):
    """Define constants"""
    #T is the horizon
    nx = n_states*n_agents
    nu = n_inputs*n_agents
    N = n_agents

    x_dims = [n_states] * N
    n_dims = [3]*N
    
    # u_ref = np.array([0, 0, 9.8] * N).reshape(-1,1)
    u_ref = np.array([0, 0, 0]*N).reshape(-1,1)
    
    """Creating empty dicts to hold Casadi variables for each worker machine"""
    f_list = {}
    d = {} 
    states = {}
    dt = 0.1
    for id in range(N):
        d["opti_{0}".format(id)] = Opti('conic')
        
        #Augmented state : Y = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        
        states["Y_{0}".format(id)] = d[f"opti_{id}"].variable((T+1)*nx + T* nu)
        cost = 0
    
        #Quadratic tracking cost
        
        for t in range(T):
            for idx in range(nx):
                cost += (states[f"Y_{id}"][:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) *  \
                Q[idx,idx]* (states[f"Y_{id}"][:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) 
            for idu in range(nu):
                cost += (states[f"Y_{id}"][(T+1)*nx:][t*nu:(t+1)*nu][idu] - u_ref[idu]) *  \
                R[idu,idu] * (states[f"Y_{id}"][(T+1)*nx:][t*nu:(t+1)*nu][idu] - u_ref[idu])
        
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

        rho = 1
        cost += (rho/2)*sumsqr(states[f"Y_{agent_id}"] - xbar + u)
        
        # ADMM loop
        iter = 0
        #open-loop rollout of dynamics
        # u_init = np.array([0, 0, 0]*N)
       
        scaling_matrix = np.diag([1, 1, 2])
        Ad,Bd = linear_kinodynamics(0.1, N)
        while True:
            try:
                coll_cost = 0
                smooth_trj_cost = 0
                # f = util.generate_f(x_dims)
                
                for k in range(T):
                    # k1 = f(states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx],states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])
                    # k2 = f(states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx]+dt/2*k1, states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])
                    # k3 = f(states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx]+dt/2*k2, states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])
                    # k4 = f(states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx]+dt*k3,   states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])
                    # x_next = states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx] + dt/6*(k1+2*k2+2*k3+k4) 
                    # d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx]==x_next) # close the gaps
                    
                    # d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu] <= np.tile(np.array([np.pi/6, np.pi/6, 20]),(N,)).reshape(-1,1))
                    # d[f"opti_{agent_id}"].subject_to(np.tile(np.array([-np.pi/6, -np.pi/6, 0]),(N,)).reshape(-1,1) <= states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])
                    
                    d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx] \
                                    == Ad @ states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx] \
                                        + Bd @ states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])

                    d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu] <= np.tile(np.array([3, 3, 3]),(N,)).reshape(-1,1))
                    d[f"opti_{agent_id}"].subject_to(np.tile(np.array([-3, -3, -3]),(N,)).reshape(-1,1) <= states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])
                    
                    
                    # #Soft collision-avoidance constraints
                    # if n_agents > 1:
                    #     distances = util.compute_pairwise_distance_nd_Sym(states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx], x_dims, n_dims)
                    #     #Collision avoidance cost
                    #     for dist in distances:
                    #         # coll_cost += fmin(0,(dist - 2*radius))**2 * 1500 #Works for centralized ADMM MPC
                    #         coll_cost += fmin(0,(dist - 2*radius))**2 * 1200
                    
                    #Linearized collision constraints:
                    if N > 1:
                        if local_iter <= 0:
                            pos_prev = x_trj_init[k]
                            print(f'pos_prev has shape {pos_prev.shape}')
                        
                        else:
                            pos_prev = state_prev[:(T+1)*nx].reshape(T+1, nx)[k]
                            
                            # pos_prev = X_full[iter-1]
                            # pos_curr = cp.reshape(y_state[:(T+1)*nx],[T+1,nx])[k]
                        
                        for i in range(N):
                            for j in range(N):
                                if j != i:
                                    #See "Generation of collision-free trajectories for a quadrocopter fleet: 
                                    # A sequential convex programming approach" for the linearization step;
                                    linearized_dist = cs.norm_2((pos_prev[j*n_states:j*n_states+3]-  \
                                            pos_prev[i*n_states:i*n_states+3])) + \
                                            (pos_prev[j*n_states:j*n_states+3].reshape(1,-1)- \
                                            pos_prev[i*n_states:i*n_states+3].reshape(1,-1))/cs.norm_2((pos_prev[j*n_states:j*n_states+3]\
                                            -pos_prev[i*n_states:i*n_states+3]))@  \
                                            (states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx][j*n_states:j*n_states+3] \
                                            -states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx][i*n_states:i*n_states+3])
              
                                    d[f"opti_{agent_id}"].subject_to(linearized_dist >= radius)
                    
                    
                    #Trajectory smoothing term
                    for ind in range(nx):
                        smooth_trj_cost += (states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx][ind]-\
                                            states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx][ind])**2
                    
                X0 = d[f"opti_{agent_id}"].parameter(x0.shape[0],1)
                # d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][0:nx] == x0) 
                d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][0:nx] == X0)
                d[f"opti_{agent_id}"].set_value(X0,x0)
                
                cost_tot = cost + coll_cost/n_agents + smooth_trj_cost
                
                d[f"opti_{agent_id}"].minimize(cost_tot)
                d[f"opti_{agent_id}"].solver("osqp")
                
                if iter > 0:
                    d[f"opti_{agent_id}"].set_initial(sol_prev.value_variables())

                sol = d[f"opti_{agent_id}"].solve()
                
                # print(f'paramete xbar has value {sol.value(xbar)}')
                # print(f'parameter u has value {sol.value(u)}')
      
                sol_prev = sol
                
                # state_prev = sol.value(states[f"Y_{agent_id}"])
                pipe.send(sol.value(states[f"Y_{agent_id}"]))
                
                d[f"opti_{agent_id}"].set_value(xbar, pipe.recv()) #receive the averaged result from the main process.
                d[f"opti_{agent_id}"].set_value(u, sol.value( u + states[f"Y_{agent_id}"] - xbar))

                # print(f'Current iteration is {iter}')
                
                d[f"opti_{agent_id}"].subject_to()
                
                iter += 1                
                
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


    solution_list = []
    admm_iter_time = []
    
    x_bar_history = [np.ones((nx, 1))*np.inf]
    iter = 0
    t0 = perf_counter()
    for i in range(MAX_ITER):
        
        # Gather and average Y_i
        xbar = sum(pipe.recv() for pipe in pipes)/N
        
        x_bar_history.append(xbar)
        solution_list.append(xbar)
        
        # Scatter xbar
        for pipe in pipes:
            pipe.send(xbar)
        
        iter += 1
        
        if np.linalg.norm(x_bar_history[iter]-x_bar_history[iter-1]) <= 1e-3:
            print(f'Consensus reached after {iter} iterations!')
            
            break

    admm_iter_time.append(perf_counter() - t0)    
    [p.terminate() for p in procs]
        
    
    x_trj_converged = solution_list[-1][:(T+1)*nx].reshape((T+1,nx))
    u_trj_converged = solution_list[-1][(T+1)*nx:].reshape((T,nu))
    
    coupling_cost = 0
    for k in range(x_trj_converged.shape[0]-1):
        distances = util.compute_pairwise_distance_nd_Sym(x_trj_converged[k,:].reshape(-1,1), x_dims, n_dims)
        for pair in distances:
            coupling_cost +=  fmin(0,(pair - 2*radius))**2 * 1200
    

    return x_trj_converged, u_trj_converged, admm_iter_time, coupling_cost/N


def solve_admm_mpc(n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf, MAX_ITER, n_trial=None):
    centralized = True
    nx = n_states*n_agents
    nu = n_inputs*n_agents
    
    X_full = np.zeros((0, nx))
    U_full = np.zeros((0, nu))
    X_full = np.r_[X_full, x0.reshape(1,-1)]
    u_ref = np.array([0, 0, 0]*n_agents)
    # u_ref = np.array([0, 0, 9.8]*n_agents)
    
    x_curr = x0
    mpc_iter = 0
    obj_history = [np.inf]
    solve_times = []
    t = 0
    dt = 0.1
    # t_kill = T*dt
    local_iter = 0
    state_prev = None
    
    u_init = np.random.rand(3*n_agents)*0.1
    x_trj_init = np.zeros((0, nx))
    x_trj_init = np.r_[x_trj_init, x0.reshape(1,-1)]
    Ad,Bd = linear_kinodynamics(0.1,n_agents)
    x_nominal = x0
    
    for _ in range(T):
        x_nominal = Ad@x_nominal + Bd@u_init.reshape(-1,1)
        x_trj_init = np.r_[x_trj_init, x_nominal.reshape(1,-1)]
        
    while not np.all(dpilqr.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
        
        x_trj_converged, u_trj_converged, admm_time,coupling_cost = solve_iteration(n_states, n_inputs, n_agents, x_curr, \
                                                                 xr, T, radius, Q, R, Qf, x_trj_init, state_prev, local_iter, MAX_ITER)
        local_iter += 1
        solve_times.append(admm_time)
        state_prev = np.hstack((x_trj_converged.flatten(),u_trj_converged.flatten()))

        # if solve_times[-1] > t_kill:
        #     converged = False
        #     print('Solve time exceeded t_kill!')
        #     break
        
        obj_history.append(float(objective(x_trj_converged.T, u_trj_converged.T, u_ref, xr, Q, R, Qf))\
                           +float(coupling_cost))
        
        x_curr = x_trj_converged[1]
        u_curr = u_trj_converged[0]
        
        X_full = np.r_[X_full, x_curr.reshape(1,-1)]
        U_full = np.r_[U_full, u_curr.reshape(1,-1)]
        
        mpc_iter += 1
        t += dt
        if mpc_iter > 30:
            print('Max MPC iters reached!Exiting MPC loops...')
            converged = False
            break

    print(f'Final distance to goal is {dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)}')
    
    if np.all(dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
        converged = True

    
    obj_trj = float(objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf))
    
    logging.info(
    f'{n_trial},'
    f'{n_agents},{t},{converged},'
    f'{obj_trj},{T},{dt},{radius},{centralized},{np.mean(solve_times)},{MAX_ITER},'
    f'{dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
    )
    
    
    return X_full, U_full, obj_trj, np.mean(solve_times), obj_history
        
# def solve_distributed_rhc(ids, n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf, MAX_ITER, n_trial=None):
    
#     n_dims = [3]*n_agents
#     u_ref = np.array([0, 0, 0]*n_agents)
#     x_dims = [n_states]*n_agents
#     u_dims = [n_inputs]*n_agents
    
#     nx = n_states*n_agents
#     nu = n_inputs*n_agents
#     X_full = np.zeros((0, nx))
#     U_full = np.zeros((0, nu))
#     X_full = np.r_[X_full, x0.reshape(1,-1)]
    
#     distributed_mpc_iters =0
#     solve_times_total = []
#     x_curr = x0
#     obj_history = [np.inf]
#     t = 0
#     dt = 0.1

#     ADMM = True
#     while not np.all(np.all(dpilqr.distance_to_goal(x_curr.flatten(), xr.flatten(), \
#                                                     n_agents, n_states, 3) <= 0.1)):
#         # rel_dists = util.compute_pairwise_distance_nd_Sym(x0,x_dims,n_dims)
#         graph = util.define_inter_graph_threshold(x_curr, radius, x_dims, ids, n_dims)
        
#         split_states_initial = split_graph(x_curr.T, x_dims, graph)
#         split_states_final = split_graph(xr.T, x_dims, graph)
#         split_inputs_ref = split_graph(u_ref.reshape(-1, 1).T, u_dims, graph)
        
#         X_dec = np.zeros((nx, 1))
#         U_dec = np.zeros((nu, 1))
        
#         # X_trj = np.zeros((nx, T+1))
#         # U_trj = np.zeros((nu, T))
        
#         solve_times = []
 
#         for (x0_i, xf_i, u_ref_i , (prob, ids_) , place_holder) in zip(split_states_initial, 
#                                        split_states_final,
#                                        split_inputs_ref,
#                                        graph.items(),
#                                        range(len(graph))):
#             print(f'Current sub-problem has {x0_i.size//n_states} agents \n')
            
#             t0 = perf_counter()
#             x_trj_converged_i, u_trj_converged_i, iter_time_i = solve_iteration(n_states, n_inputs, \
#                                                                       x0_i.size//n_states,\
#                                                                       x0_i.reshape(-1,1), \
#                                                                       xf_i.reshape(-1,1), \
#                                                                       T, radius, 
#                                                                       Q[:x0_i.size,:x0_i.size], 
#                                                                       R[:u_ref_i.size,:u_ref_i.size], 
#                                                                       Qf[:x0_i.size,:x0_i.size], MAX_ITER)
#             solve_times.append(iter_time_i)
#             i_prob = ids_.index(prob)
            
#             #Collecting solutions from different potential game sub-problems at current time step K:
#             X_dec[place_holder * n_states : (place_holder + 1) * n_states, :] = x_trj_converged_i[
#                 1, i_prob * n_states : (i_prob + 1) * n_states
#                 ].reshape(-1,1)
            
#             U_dec[place_holder * n_inputs : (place_holder + 1) * n_inputs, :] = u_trj_converged_i[
#                 0, i_prob * n_inputs : (i_prob + 1) * n_inputs
#                 ].reshape(-1,1)

#         obj_curr = float(objective(X_dec,U_dec, u_ref, xr, Q, R, Qf))
#         obj_history.append(obj_curr)
        
#         solve_times_total.append(np.mean(solve_times))   #Worst-case run time from each ADMM iteration 
#         # obj_history.append(float(objective(X_trj, U_trj, u_ref, xr, Q, R, Qf)))    
#         t += dt
        
#         x_curr = X_dec
        
#         X_full = np.r_[X_full, X_dec.reshape(1,-1)]
#         U_full = np.r_[U_full, U_dec.reshape(1,-1)]
        
#         distributed_mpc_iters += 1
        
#         if distributed_mpc_iters > 15:
#             print(f'Max iters reached; exiting MPC loops')
#             converged = False
#             break
        
#     if np.all(dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
#         converged = True
        
#     print(f'Final distance to goal is {dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)}')    
    
#     obj_trj = float(objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf))
    
#     logging.info(
#     f'{n_trial},'
#     f'{n_agents},{t},{converged},'
#     f'{obj_trj},{T},{dt},{radius},{ADMM},{np.mean(solve_times_total)},'
#     f'{dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
#     )
    
#     return X_full, U_full, obj_trj, np.mean(solve_times), obj_history

def solve_mpc_centralized(n_agents, x0, xr, T, radius, Q, R, Qf, n_trial = None):
    ADMM = False
    nx = n_agents * 6
    nu = n_agents * 3
    N = n_agents
    opti = Opti()
    Y_state = opti.variable((T+1)*nx + T*nu)
    cost = 0

    u_ref = np.array([0, 0, 0] * N).reshape(-1,1)
    
    for t in range(T):
        for idx in range(nx):
            cost += (Y_state[:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) *  \
            Q[idx,idx]* (Y_state[:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) 
        for idu in range(nu):
            cost += (Y_state[(T+1)*nx:][t*nu:(t+1)*nu][idu] - u_ref[idu]) *  \
            R[idu,idu] * (Y_state[(T+1)*nx:][t*nu:(t+1)*nu][idu] - u_ref[idu])

    for idf in range(nx):
        cost += (Y_state[:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf]) * \
        Qf[idf,idf] * (Y_state[:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf])


    coll_cost = 0
    smooth_trj_cost = 0
    solution_trj = []
    obj_hist = [np.inf]
    x_curr = x0

    X_trj = np.zeros((0, nx))
    U_trj = np.zeros((0, nu))
    X_trj = np.r_[X_trj, x0.T]
    iters = 0

    solve_times = []
    t = 0
    dt = 0.1
    
    x_dims = [6]*N
    f = util.generate_f(x_dims)
    Ad,Bd = linear_kinodynamics(0.1,N)
    while not np.all(dpilqr.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):

        for k in range(T):
            
            # k1 = f(Y_state[:(T+1)*nx][k*nx:(k+1)*nx],         Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            # k2 = f(Y_state[:(T+1)*nx][k*nx:(k+1)*nx]+dt/2*k1, Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            # k3 = f(Y_state[:(T+1)*nx][k*nx:(k+1)*nx]+dt/2*k2, Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            # k4 = f(Y_state[:(T+1)*nx][k*nx:(k+1)*nx]+dt*k3,   Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            # x_next = Y_state[:(T+1)*nx][k*nx:(k+1)*nx] + dt/6*(k1+2*k2+2*k3+k4) 

            # opti.subject_to(Y_state[:(T+1)*nx][(k+1)*nx:(k+2)*nx]==x_next) # close the gaps
            
            # opti.subject_to(Y_state[(T+1)*nx:][k*nu:(k+1)*nu] <= np.tile(np.array([np.pi/6, np.pi/6, 20]),(N,)).reshape(-1,1))
            # opti.subject_to(np.tile(np.array([-np.pi/6, -np.pi/6, 0]),(N,)).reshape(-1,1) <= Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            
            opti.subject_to(Y_state[:(T+1)*nx][(k+1)*nx:(k+2)*nx] \
                            == Ad @ Y_state[:(T+1)*nx][k*nx:(k+1)*nx] \
                                + Bd @ Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            
            opti.subject_to(Y_state[(T+1)*nx:][k*nu:(k+1)*nu] <= np.tile(np.array([3, 3, 3]),(N,)).reshape(-1,1))
            opti.subject_to(np.tile(np.array([-3, -3, -3]),(N,)).reshape(-1,1) <= Y_state[(T+1)*nx:][k*nu:(k+1)*nu])

            #Pair-wise Euclidean distance between each pair of agents
            distances = util.compute_pairwise_distance_nd_Sym(Y_state[:(T+1)*nx][k*nx:(k+1)*nx],[6,6,6], [3,3,3])
            #Collision avoidance cost
            for dist in distances:
                coll_cost += fmin(0,(dist - radius))**2 * 1200
                # coll_cost += fmin(0,(dist - radius))**2 * 400

            #Smoothing term
            for ind in range(nx):
                smooth_trj_cost += (Y_state[:(T+1)*nx][(k+1)*nx:(k+2)*nx][ind]-\
                                    Y_state[:(T+1)*nx][k*nx:(k+1)*nx][ind])**2
        
        X0 = opti.parameter(x0.shape[0],1)     
        opti.subject_to(Y_state[0:nx] == X0)
        
        cost_tot = cost + coll_cost + smooth_trj_cost
        opti.minimize(cost_tot)

        opti.solver("ipopt")
        opti.set_value(X0,x_curr)
        
        if iters > 0:
            opti.set_initial(sol_prev.value_variables())
            
        t0 = perf_counter()
        sol = opti.solve()
        sol_prev = sol
        solve_times.append(perf_counter() - t0)
        obj_hist.append(sol.value(cost_tot))
        
        ctrl = sol.value(Y_state)[(T+1)*nx:].reshape((T, nu))[0]
        x_curr = sol.value(Y_state)[:(T+1)*nx].reshape((T+1,nx))[1] #Same as above
        X_trj = np.r_[X_trj, x_curr.reshape(1,-1)]
        U_trj = np.r_[U_trj, ctrl.reshape(1,-1)]
        
        opti.subject_to()
        
        iters += 1
        t += dt
        if iters > 30:
            converged = False
            break
    
    if np.all(dpilqr.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
        converged = True
        
    MAX_ITER = None
    obj_trj = float(util.objective(X_trj.T, U_trj.T, u_ref, xr, Q, R, Qf)) 
    logging.info(
    f'{n_trial},'
    f'{n_agents},{t},{converged},'
    f'{obj_trj},{T},{dt},{radius},{ADMM},{np.mean(solve_times)}, {MAX_ITER},'
    f'{dpilqr.distance_to_goal(X_trj[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
    )
        
    return X_trj, U_trj, obj_trj, np.mean(solve_times), obj_hist
    

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
        "i_trial, n_agents, t, converged, obj_trj,T,dt,radius,\
         ADMM ,t_solve_step, MAX_ITER, dist_to_goal"
    )
    
    
def multi_agent_run(trial, 
                    n_states,
                    n_inputs,
                    n_agents,
                    T,
                    radius,
                    Q,
                    R,
                    Qf):
    """simulation comparing the centralized and decentralized solvers"""
    
    if n_agents == 3:
        x0,xr = util.setup_3_quads()

    elif n_agents==4:
        x0, xr=util.setup_4_quads()

    elif n_agents == 5:
        x0,xr = util.setup_5_quads()

    elif n_agents==6:
        x0, xr=util.setup_6_quads()

    elif n_agents==7:
        x0, xr= util.setup_7_quads()

    elif n_agents==8:
        x0, xr= util.setup_8_quads()

    elif n_agents==9:
        x0, xr= util.setup_9_quads()

    elif n_agents == 10:
        x0,xr = util.setup_10_quads()

    ids = [100 + i for i in range(n_agents)]
    x_dims = [n_states] * n_agents
    n_dims = [3] * n_agents
    Q = np.diag([5., 5., 5., 1., 1., 1.]*n_agents)
    Qf = Q*500
    R = 0.1*np.eye(n_agents*n_inputs)
    
    admm_iter = 1
    
    model = QuadcopterDynamics6D
    dynamics = MultiDynamicalModel([model(0.1, id_) for id_ in ids])
    goal_costs = [
        ReferenceCost(xr_i, Q.copy(), R.copy(), Qf.copy(), id_)
        for xr_i, id_ in zip(split_agents_gen(xr, x_dims), ids)
    ]
    prox_cost = ProximityCost(x_dims, radius, n_dims)
    game_cost = GameCost(goal_costs, prox_cost)

    problem = ilqrProblem(dynamics, game_cost)
    
    STEP_SIZE=1
    n_d=3
    N = T
    # Xd, Ud, Jd = solve_rhc( problem,
    #                         x0,
    #                         N,
    #                         radius,
    #                         n_d = n_d,
    #                         t_kill = None,
    #                         centralized=False,
    #                         dist_converge=0.1,
    #                         step_size=STEP_SIZE,
    #                         radius = radius,
    #                         pool=None,
    #                         i_trial = trial
    #                         )
    
    X_full, U_full, obj,  avg_SolveTime, _ = solve_mpc_centralized(n_agents, 
                                                                x0,
                                                                xr, 
                                                                T, 
                                                                radius,
                                                                Q,
                                                                R,
                                                                Qf,                                                     
                                                                trial)
    
    X_full, U_full, obj,  avg_SolveTime, _ = solve_admm_mpc(n_states,
                                                n_inputs,
                                                n_agents,
                                                x0,
                                                xr,
                                                T,
                                                radius,
                                                Q,
                                                R,
                                                Qf,
                                                admm_iter,
                                                trial)
    # X_full, U_full, obj, avg_SolveTime, _ = solve_distributed_rhc(ids,
    #                                                         n_states, 
    #                                                         n_inputs, 
    #                                                         n_agents, 
    #                                                         x0, 
    #                                                         xr, 
    #                                                         T, 
    #                                                         radius,
    #                                                         Q, 
    #                                                         R, 
    #                                                         Qf,
    #                                                         admm_iter,
    #                                                         trial)

def monte_carlo_analysis():
    """Benchmark to evaluate algorithm over many random initial conditions"""

    setup_logger()

    n_trials_iter = range(30)

    n_agents_iter = [3, 4, 5, 6, 7, 8]
    # n_agents_iter = [10]

    radius = 0.5
    
    # Change the for loops into multi-processing?

    for n_agents in n_agents_iter:
        print(f"\tn_agents: {n_agents}")
        
        if n_agents >=5 and n_agents <=7:
             radius = 0.3
        
        if n_agents >= 8 :
             radius = 0.15
            
        for i_trial in n_trials_iter:
            print(f"\t\ttrial: {i_trial}")
            
            multi_agent_run(
                            i_trial,
                            n_states,
                            n_inputs,
                            n_agents,
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
    T = 8
    # T = 10
    radius = 0.5
    Q = np.diag([5., 5., 5., 1., 1., 1.]*n_agents)
    Qf = Q*500
    R = 0.1*np.eye(n_agents*n_inputs)
    
    ids = [100 + n for n in range(n_agents)] #Assigning random IDs for agents
    
    Log_Data = False
    if not Log_Data:
        # admm_iter = 15
        # admm_iter = 5
        admm_iter = 3
        x0, xr = util.paper_setup_3_quads()
        # if centralized:
        X_full, U_full, obj, avg_SolveTime, obj_history_admm = solve_admm_mpc(n_states,
                                                            n_inputs,
                                                            n_agents,
                                                            x0,
                                                            xr,
                                                            T,
                                                            radius,
                                                            Q,
                                                            R,
                                                            Qf,
                                                            admm_iter)
        print(f'The average solve time is {avg_SolveTime} seconds!')
        #Plot trajectory
        plt.figure(dpi=150)
        dpilqr.plot_solve(X_full, float(obj), xr, x_dims, True, 3)
        # plt.gca().set_zticks([0.8,1.2], minor=False)
        plt.legend(plt.gca().get_children()[1:3], ["Start Position", "Goal Position"])
        plt.savefig('ADMM_mpc.png')
        
        
        plt.figure(dpi=150)
        dpilqr.plot_pairwise_distances(X_full, [6,6,6], [3,3,3], radius)
        plt.title('Pairwise-distances from C-ADMM')
        plt.savefig('pairwise_distances(ADMM).png')
        # else:
            
        X_full, U_full, obj,  avg_SolveTime, obj_history_centralized = solve_mpc_centralized(n_agents, 
                                                                                            x0,
                                                                                            xr, 
                                                                                            T, 
                                                                                            radius,
                                                                                            Q,
                                                                                            R,
                                                                                            Qf                                                     
                                                                                            )
        print(f'The average solve time is {avg_SolveTime} seconds!')
        
        plt.figure(dpi=150)
        dpilqr.plot_solve(X_full, float(obj), xr, x_dims, True, 3)
        # plt.gca().set_zticks([0.8,1.2], minor=False)
        plt.legend(plt.gca().get_children()[1:3], ["Start Position", "Goal Position"])
        plt.savefig('centralized_mpc_baseline.png')
        
        plt.figure(dpi=150)
        dpilqr.plot_pairwise_distances(X_full, [6,6,6], [3,3,3], radius)
        plt.title('Pairwise-distances from vanilla MPC')
        plt.savefig('pairwise_distances(vanilla_mpc).png')
        
        
        plt.figure(dpi=150)
        plt.plot(obj_history_admm, 'r', label='Potential Consensus-ADMM')
        plt.plot(obj_history_centralized, 'b', label='Baseline Centralized MPC')
        plt.ylabel('Total Cost-to-go')
        plt.xlabel('Horizon')
        plt.legend(loc='best')
        plt.savefig('convergence_rate.png')
        

    if Log_Data:
        monte_carlo_analysis()
        
        
    
        
    
        
    
        

        