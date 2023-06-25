import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import casadi as cs
import dpilqr
import cvxpy as cp
from time import perf_counter

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


def solve_iteration(n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf):
    """Define constants"""

    nx = n_states*n_agents
    nu = n_inputs*n_agents
    N = n_agents
    Ad, Bd = linear_kinodynamics(0.1, N)

    """Creating empty dicts to hold Casadi variables"""
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
                    distances = util.compute_pairwise_distance_nd_Sym(states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx],[6,6,6], [3,3,3])
                    #Collision avoidance cost
                    for dist in distances:
                        coll_cost += fmin(0,(dist - 2*radius))**2 * 500

                    #Trajectory smoothing term
                    for ind in range(nx):
                        smooth_trj_cost += (states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx][ind]-\
                                            states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx][ind])**2
                    
                
                d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][0:nx] == x0) 
                        
                d[f"opti_{agent_id}"].minimize(cost + coll_cost/n_agents + smooth_trj_cost)
                
                d[f"opti_{agent_id}"].solver("ipopt")
                
                sol = d[f"opti_{agent_id}"].solve()
                # result[f"solution_{0}".format(agent_id)] = sol
                
                # print(f'paramete xbar has value {sol.value(xbar)}')
                # print(f'parameter u has value {sol.value(u)}')
                
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


def solve_rhc(n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf):
    nx = n_states*n_agents
    nu = n_inputs*n_agents
    
    X_full = np.zeros((0, nx))
    U_full = np.zeros((0, nu))
    
    u_ref = np.array([0, 0, 0]*n_agents)
    
    x_curr = x0
    mpc_iter = 0
    ADMM_iters = []
    obj_history = [np.inf]
    solve_times = []
    
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
        
        if (mpc_iter > 35) or (obj_history[mpc_iter] - obj_history[mpc_iter-1]) > 0:
            print('Exiting MPC loops')
            break
        
    
    return X_full, U_full, objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf), np.mean(ADMM_iters), np.mean(solve_times)
        
        
        
if __name__ == "__main__":
    
    n_states = 6
    n_inputs = 3
    n_agents = 3
    x_dims = [n_states]*n_agents
    x0, xr = util.paper_setup_3_quads()
    T = 15
    radius = 0.35
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
                                                Qf)
    
    print(f'The average solve time is {avg_SolveTime} seconds!')
    
    #Plot trajectory
    plt.figure(dpi=150)
    dpilqr.plot_solve(X_full, float(obj), xr, x_dims, True, 3)
    plt.gca().set_zticks([0.8,1.2], minor=False)
    plt.savefig('ADMM_mpc.png')
    
    #Plot pairwise distance
    plt.figure(dpi=150)
    dpilqr.plot_pairwise_distances(X_full, x_dims, [3,3,3], radius)
    plt.savefig('Pairwise_distances_ADMM.png')