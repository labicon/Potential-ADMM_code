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

opts = {'error_on_fail':False}

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
       
        scaling_matrix = np.diag([1, 1, 2])
        Ad,Bd = linear_kinodynamics(0.1, N)
        
        while True:
            try:
                smooth_trj_cost = 0
                for k in range(T):
                    
                    
                    d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx] \
                                    == Ad @ states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx] \
                                        + Bd @ states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])

                    d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu] <= np.tile(np.array([3, 3, 3]),(N,)).reshape(-1,1))
                    d[f"opti_{agent_id}"].subject_to(np.tile(np.array([-3, -3, -3]),(N,)).reshape(-1,1) <= states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])
                  
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
                                    linearized_dist = cs.norm_2(scaling_matrix@(pos_prev[j*n_states:j*n_states+3]-  \
                                            pos_prev[i*n_states:i*n_states+3])) + \
                                            (pos_prev[j*n_states:j*n_states+3].reshape(1,-1)- \
                                            pos_prev[i*n_states:i*n_states+3].reshape(1,-1))/cs.norm_2(scaling_matrix@(pos_prev[j*n_states:j*n_states+3]\
                                            -pos_prev[i*n_states:i*n_states+3]))@  \
                                            (states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx][j*n_states:j*n_states+3] \
                                            -states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx][i*n_states:i*n_states+3])
            
                                    d[f"opti_{agent_id}"].subject_to(linearized_dist >=  radius)
                    
                    #Trajectory smoothing term
                    for ind in range(nx):
                        smooth_trj_cost += (states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx][ind]-\
                                            states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx][ind])**2
                    
                X0 = d[f"opti_{agent_id}"].parameter(x0.shape[0],1)
 
                d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][0:nx] == X0)
                
                cost_tot = cost +  smooth_trj_cost
                
                d[f"opti_{agent_id}"].minimize(cost_tot)
                d[f"opti_{agent_id}"].solver("osqp",opts)
                # d[f"opti_{agent_id}"].solver("qpoases",opts)  #this one is too slow
                # d[f"opti_{agent_id}"].solver("ipopt")
                
                if iter > 0:
                    d[f"opti_{agent_id}"].set_initial(sol_prev.value_variables())
                
                d[f"opti_{agent_id}"].set_value(X0,x0)
                sol = d[f"opti_{agent_id}"].solve()
      
                sol_prev = sol
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
    # for k in range(x_trj_converged.shape[0]-1):
    #     distances = util.compute_pairwise_distance_nd_Sym(x_trj_converged[k,:].reshape(-1,1), x_dims, n_dims)
    #     for pair in distances:
    #         coupling_cost +=  fmin(0,(pair - 2*radius))**2 * 1200
    
    return x_trj_converged, u_trj_converged, admm_iter_time, coupling_cost


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
        
        try:
            x_trj_converged, u_trj_converged, admm_time,coupling_cost = solve_iteration(n_states, n_inputs, n_agents, x_curr, \
                                                                 xr, T, radius, Q, R, Qf, x_trj_init, state_prev, mpc_iter, MAX_ITER)
        except RuntimeError:
            converged = False
            
        
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
        if mpc_iter > 35:
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
        
def solve_mpc_centralized(n_agents, x0, xr, T, radius, Q, R, Qf, n_trial = None):
    ADMM = False
    nx = n_agents * 6
    nu = n_agents * 3
    N = n_agents
    opti = Opti('conic')
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
    n_dims = [3]*N
    f = util.generate_f(x_dims)
    Ad,Bd = linear_kinodynamics(0.1,N)
    
    u_init = np.random.rand(3*n_agents)*0.1
    x_trj_init = np.zeros((0, nx))
    x_trj_init = np.r_[x_trj_init, x0.reshape(1,-1)]
    Ad,Bd = linear_kinodynamics(0.1,n_agents)
    x_nominal = x0
    
    scaling_matrix = np.diag([1, 1, 2])
    
    for _ in range(T):
        x_nominal = Ad@x_nominal + Bd@u_init.reshape(-1,1)
        x_trj_init = np.r_[x_trj_init, x_nominal.reshape(1,-1)]
    
    while not np.all(dpilqr.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
        coll_cost = 0
        smooth_trj_cost = 0
        for k in range(T):
            
            opti.subject_to(Y_state[:(T+1)*nx][(k+1)*nx:(k+2)*nx] \
                            == Ad @ Y_state[:(T+1)*nx][k*nx:(k+1)*nx] \
                                + Bd @ Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            
            opti.subject_to(Y_state[(T+1)*nx:][k*nu:(k+1)*nu] <= np.tile(np.array([3, 3, 3]),(N,)).reshape(-1,1))
            opti.subject_to(np.tile(np.array([-3, -3, -3]),(N,)).reshape(-1,1) <= Y_state[(T+1)*nx:][k*nu:(k+1)*nu])

            #Linearized collision constraints:
            if N > 1:
                if iters <= 0:
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
                            linearized_dist = cs.norm_2(scaling_matrix@(pos_prev[j*n_states:j*n_states+3]-  \
                                    pos_prev[i*n_states:i*n_states+3])) + \
                                    (pos_prev[j*n_states:j*n_states+3].reshape(1,-1)- \
                                    pos_prev[i*n_states:i*n_states+3].reshape(1,-1))/cs.norm_2(scaling_matrix@(pos_prev[j*n_states:j*n_states+3]\
                                    -pos_prev[i*n_states:i*n_states+3]))@  \
                                    (Y_state[:(T+1)*nx][k*nx:(k+1)*nx][j*n_states:j*n_states+3] \
                                    -Y_state[:(T+1)*nx][k*nx:(k+1)*nx][i*n_states:i*n_states+3])
        
                            opti.subject_to(linearized_dist >= radius)

            #Smoothing term
            for ind in range(nx):
                smooth_trj_cost += (Y_state[:(T+1)*nx][(k+1)*nx:(k+2)*nx][ind]-\
                                    Y_state[:(T+1)*nx][k*nx:(k+1)*nx][ind])**2
        
        X0 = opti.parameter(x0.shape[0],1)     
        opti.subject_to(Y_state[0:nx] == X0)
        
        cost_tot = cost + coll_cost/N + smooth_trj_cost
        
        opti.minimize(cost_tot)

        opti.solver('osqp',opts)
        # opti.solver('qpoases',opts)
        # opti.solver("ipopt")
        opti.set_value(X0,x_curr)
        
        if iters > 0:
            opti.set_initial(sol_prev.value_variables())
            
        t0 = perf_counter()
        try:
            sol = opti.solve()
            
        except RuntimeError:
            converged=False
            break
            
        sol_prev = sol
        solve_times.append(perf_counter() - t0)
        obj_hist.append(sol.value(cost_tot))
        
        state_prev = sol.value(Y_state)
        
        ctrl = sol.value(Y_state)[(T+1)*nx:].reshape((T, nu))[0]
        x_curr = sol.value(Y_state)[:(T+1)*nx].reshape((T+1,nx))[1] #Same as above
        X_trj = np.r_[X_trj, x_curr.reshape(1,-1)]
        U_trj = np.r_[U_trj, ctrl.reshape(1,-1)]
        
        opti.subject_to()
        
        iters += 1
        t += dt
        if iters > 35:
            converged = False
            print(f'Max MPC iters reached; exiting MPC loops.....')
            break
    
    print(f'Final distance to goal is {dpilqr.distance_to_goal(X_trj[-1].flatten(), xr.flatten(), n_agents, n_states, 3)}')
    
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

    admm_iter = 3
    
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

def monte_carlo_analysis():
    """Benchmark to evaluate algorithm over many random initial conditions"""

    setup_logger()

    n_trials_iter = range(30)

    # n_agents_iter = [3, 4, 5, 6, 7, 8]
    n_agents_iter = [3, 5, 10]
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
    
    Log_Data = True
    if not Log_Data:
        # admm_iter = 15
        # admm_iter = 5
        admm_iter = 5
        x0, xr = util.paper_setup_3_quads(True)

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
        plt.savefig('ADMM_mpc(QP).png')
        
        
        plt.figure(dpi=150)
        dpilqr.plot_pairwise_distances(X_full, [6,6,6], [3,3,3], radius)
        plt.title('Pairwise-distances from C-ADMM (QP)')
        plt.savefig('pairwise_distances(ADMM_QP).png')
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
        plt.savefig('centralized_mpc_baseline(QP).png')
        
        plt.figure(dpi=150)
        dpilqr.plot_pairwise_distances(X_full, [6,6,6], [3,3,3], radius)
        plt.title('Pairwise-distances from vanilla MPC(QP)')
        plt.savefig('pairwise_distances(vanilla_mpc_QP).png')
        
        
        plt.figure(dpi=150)
        plt.plot(obj_history_admm, 'r', label='Potential Consensus-ADMM')
        plt.plot(obj_history_centralized, 'b', label='Baseline Centralized MPC')
        plt.ylabel('Total Cost-to-go')
        plt.xlabel('Horizon')
        plt.legend(loc='best')
        plt.savefig('convergence_rate_QP.png')
        

    if Log_Data:
        monte_carlo_analysis()
        
        
    
        
    
        
    
        

        