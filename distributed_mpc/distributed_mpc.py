import casadi as cs
import numpy as np
from scipy.constants import g
from casadi import *
from util import *
import logging
import util
from time import perf_counter

centralized = False

def solve_rhc_distributed(
    dt,x0, xf, u_ref, N, n_agents, n_states, n_inputs, radius, ids,\
    x_min,x_max,y_min,y_max,z_min,z_max,v_min,v_max,theta_max,\
  theta_min,tau_max,tau_min,phi_max,phi_min,n_humans,unit_dist,n_dims=None,j_trial=None,equal_dist = False):

    x_dims = [n_states] * n_agents
    u_dims = [n_inputs] * n_agents
    
    p_opts = {"expand": True}
    s_opts = {"max_iter": 200, "print_level": 0}

    M = 100  # this is the maximum number of outer iterations
    
    n_x = n_agents * n_states
    # n_u = n_agents * n_inputs
    n_u = sum(u_dims)
    t = 0

    J_list = []
    J_list.append(np.inf)
    # for i in range(M) :
    loop = 0
 
    X_full = np.zeros((0, n_x))
    U_full = np.zeros((0, n_u))
    
    failed_count = 0
    converged = False
    
    if n_humans!=0:
        id_humans = ids[-n_humans:]
    else:
        id_humans = None
    
    t_solve_start = perf_counter()
    while not np.all(distance_to_goal(x0, xf, n_agents, n_states) <= 0.1) and (loop < M):

        ######################################################################
        # Determine sub problems to solve:

        # compute interaction graph at the current time step:
        if loop > 0:
            print(f"re-optimizing at {x0.T} \n")
            
        # rel_dists = util.compute_pairwise_distance(x0, x_dims, n_d=3)
        rel_dists = util.compute_pairwise_distance_nd_Sym(x0,x_dims,n_dims)
        #Pairwise distance between any agent and a human agent is the 2-D planar Euclidean distance
        #since the human agent is assumed to be a cylindrical object with constant height
        
        #How do I know if the current subproblem has a human agent?
        # ids[-n_humans:]: id(s) of the human agents
        graph = util.define_inter_graph_threshold(x0, radius, x_dims, ids, n_dims)

        # print(
            # f"current interaction graph is {graph}, the pairwise distances between each agent is {rel_dists} \n"
        # )
        # x0 is updated until convergence (treat x0 as the combined CURRENT state)

        # break up the problem into potential-game sub-problems at every outer iteration
        split_problem_states_initial = split_graph(x0.T, x_dims, graph)
        # print(split_problem_states_initial)
        split_problem_states = split_graph(xf.T, x_dims, graph)
        split_problem_inputs = split_graph(u_ref.reshape(-1, 1).T, u_dims, graph)

        # Initiate different instances of Opti() object
        # Each Opti() object corresponds to a subproblem (there is NO central node)
        # Note that when 2 agents are combined into a single problem, we have 2 copies of the same sub problem
        ########################################################################
        # Setting up the solvers:
        d = {}
        states = {}
        inputs = {}
        cost_fun_list = []

        d = {}  # dictionary holding Opti() objects (or subproblems)
        states = {}  # dictionary holding symbolic state trajectory for each sub-problem
        inputs = {}  ##dictionary holding symbolic input trajectory for each sub-problem

        for i, j in enumerate(split_problem_states_initial):
            d["opti_{0}".format(i)] = cs.Opti()
            states["X_{0}".format(i)] = d[f"opti_{i}"].variable(j.shape[1], N + 1)

        for i, j in enumerate(split_problem_inputs):
            inputs["U_{0}".format(i)] = d[f"opti_{i}"].variable(j.shape[1], N)

        # Storing objective functions for each sub-problem into a list:
        for i in range(len(split_problem_states_initial)):
            cost_fun_list.append(
                objective(
                    states[f"X_{i}"],
                    inputs[f"U_{i}"],
                    split_problem_inputs[i].reshape(
                        -1,
                    ),
                    split_problem_states[i].reshape(-1, 1),
                    np.eye(split_problem_states_initial[i].shape[1]) * 100,
                    np.eye(
                        split_problem_inputs[i]
                        .reshape(
                            -1,
                        )
                        .shape[0]
                    )
                    * 0.1,
                    np.eye(split_problem_states_initial[i].shape[1]) * 1000,
                )
            )

        min_max_input_list = generate_min_max_input(inputs, n_inputs,theta_max,
                          theta_min,tau_max,tau_min,phi_max,phi_min)
        min_max_state_list = generate_min_max_state(states, n_states,x_min,
                          x_max,y_min,y_max,z_min,z_max,v_min,v_max)
        
        # print(f'min_max_input_list has length {len(min_max_input_list)}')
        ##########################################################################
        # Solve each sub-problem in a sequential manner:
        
        objective_val = 0
        X_dec = np.zeros((1, n_x))
        U_dec = np.zeros((1, n_u))
        
        
        for (
            di,
            statesi,
            inputsi,
            costi,
            state_boundsi,
            input_boundsi,
            (prob, ids_),
            count,
        ) in zip(
            d.values(),
            states.values(),
            inputs.values(),
            cost_fun_list,
            min_max_state_list,
            min_max_input_list,
            graph.items(),
            range(len(d)),
        ):  # loop over sub-problems

            print(f'prob is {prob}, ids is {ids_}')
            # print(f'statesi has shape {statesi[:,0].shape}')

            print(f"Solving the {count}th sub-problem at iteration {loop}, t = {t} \n")

            min_states, max_states = state_boundsi
            min_inputs, max_inputs = input_boundsi

            di.minimize(costi)

            n_states_local = statesi.shape[0]  # each subproblem has different number of states
            # print(f'n_states_local:{n_states_local}')
            n_inputs_local = inputsi.shape[0]
            x_dims_local = [int(n_states)] * int(n_states_local / n_states)
        
            print(f"current sub-problem has state dimension : {x_dims_local} \n")
            if n_humans !=0:
                if any(item in ids_ for item in id_humans):

                    human_count = sum(item in ids_ for item in id_humans) #Number of humans in the current sub-problem
                    drones_count = len(x_dims_local)-human_count
                    #print(f'{human_count} human agents are detected in the sub problem')
                    n_dims_local = [3]*drones_count
                    n_dims_local+= [2]*human_count
                    print(f'n_dims is {n_dims_local}')
                    f = generate_f_human_drone(x_dims_local,human_count)
            else:
                f = generate_f(x_dims_local)
                drones_count = len(x_dims_local)
                n_dims_local = [3]*drones_count
                human_count = None
                
            print(f'number of drones in the current subproblem is {drones_count}\n')
            print(f'number of humans in the current subproblem is {human_count}\n')
            #Only impose constraints on the drones
            
            for k in range(N):  # loop over control intervals
                # Runge-Kutta 4 integration

                k1 = f(statesi[:, k], inputsi[:, k])
                k2 = f(statesi[:, k] + dt / 2 * k1, inputsi[:, k])
                k3 = f(statesi[:, k] + dt / 2 * k2, inputsi[:, k])
                k4 = f(statesi[:, k] + dt * k3, inputsi[:, k])
                x_next = statesi[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                di.subject_to(statesi[:, k + 1] == x_next)  # close the gaps
            
                # di.subject_to(inputsi[: ,k] <= max_inputs.T)
                # di.subject_to(min_inputs.T <= inputsi[:, k])
                
                if drones_count !=0:
                    # print(f'inputsi has shape{inputsi[0:drones_count*n_inputs,k].shape},max_inputs has     shape{max_inputs.shape}\n')
                    di.subject_to(inputsi[0:drones_count*n_inputs,k] <= max_inputs[: , 0:drones_count*n_inputs].T)
                    di.subject_to(min_inputs[: , 0:drones_count*n_inputs].T <= inputsi[0:drones_count*n_inputs,k])
                    
                    # di.subject_to(inputsi[:,k] <= max_inputs.T)
                    # di.subject_to(min_inputs.T <= inputsi[:,k])

            for k in range(N + 1):
                if drones_count !=0:
                    # print(f'statesi has shape{statesi[0:drones_count*n_states,k].shape}
                    # ,max_states has shape{max_states.shape}\n')
                    di.subject_to(statesi[0:drones_count*n_states, k] <= max_states[:, 0:drones_count*n_states].T)
                    di.subject_to(min_states[:, 0:drones_count*n_states].T <= statesi[0:drones_count*n_states, k])
                    
                    # di.subject_to(statesi[:, k] <= max_states.T)
                    # di.subject_to(min_states.T <= statesi[:, k])
     

                # Collision avoidance over the current prediction horizon: 
                # (only if the current sub-problem contains more than 1 agent):

                #Below is a hard-coded example for a heterogenous formation control task:
                if len(x_dims_local) != 1:
                    # if equal_dist == True:
                    #     # if [ind == 100 or 101 or 102 or 103 for ind in ids_]:
                    #     #     # distances = compute_distances_square(statesi[:,k])
                    #     #     # for n in distances:
                    #     #     #     di.subject_to(n == unit_dist)
                                                
                    #     # distances = compute_pairwise_distance_nd_Sym(statesi[:, k], x_dims_local, n_dims_local)
                    #     # for n in distances:
                    #     #     di.subject_to (n >= radius)
                    #     #     else:
                    #     if len(ids_) >2 :
                    #         distances = compute_pairwise_distance_nd_Sym(statesi[:, k], x_dims_local, n_dims_local)
                    #         for n in distances:
                    #             di.subject_to(n >= radius)
                    #     else:
                    #         pass
            
                    # else:
                    distances = compute_pairwise_distance_nd_Sym(
                        statesi[:, k], x_dims_local, n_dims_local
                    )
                    for n in distances:
                        di.subject_to (n >= radius)

            # equality constraints for initial condition:
            di.subject_to(
                statesi[:, 0] == split_problem_states_initial[count].reshape(-1, 1)
            )

            di.solver("ipopt", p_opts, s_opts)
            
            t0 = perf_counter()
            try:
                sol = di.solve()
                tf = perf_counter()
                print(f'solve time for the current sub-problem is {tf-t0} \n')

            except RuntimeError:
                t_solve = None
                print('Current problem is infeasible')
                failed_count +=1
                # return X_full, U_full, t, J_list, failed_count, converged
                break
          
            objective_val += sol.value(costi)
            print(
                f"objective value for the {count}th subproblem at iteration {loop} is {sol.value(costi)} \n"
            )
            # print(sol.value(statesi).shape)
            x0_local = sol.value(statesi)[:, 1]

            u_sol_local = sol.value(inputsi)[:, 0]

            i_prob = ids_.index(prob)

            # X_dec[0,:] = x0.reshape(1,-1)

            X_dec[:, count * n_states : (count + 1) * n_states] = x0_local[
                i_prob * n_states : (i_prob + 1) * n_states
            ]
            U_dec[:, count * n_inputs : (count + 1) * n_inputs] = u_sol_local[
                i_prob * n_inputs : (i_prob + 1) * n_inputs
            ]

        x0 = X_dec.reshape(-1, 1)
        print(f"current collected solution is {x0.T} \n")

        # print(x0)
        J_list.append(
            objective_val
        )  # collect aggregate objective function from all sub-problems after each control horizon is over
        print(
            f"current combined objective value is {objective_val}##########################\n"
        )
        # Store the trajectory

        X_full = np.r_[X_full, X_dec.reshape(1, -1)]
        # print(X_full.shape)
        # x0 = X_full[loop,:].reshape(-1,1)

        U_full = np.r_[U_full, U_dec.reshape(1, -1)]

        t += dt
        loop += 1
     
        if np.all(distance_to_goal(x0, xf, n_agents, n_states) <= 0.1):
            converged = True
            print(f"Terminated! at loop = {loop}, converged is {converged} \n")
            t_solve_end = perf_counter()
            t_solve = t_solve_end-t_solve_start
            
            logging.info(
                f'{j_trial},'
                f'{n_agents},{t},{failed_count},{converged},'
                f'{objective_val},{N},{dt},"{ids}",{radius},{centralized},{t_solve},'
            )
            break
        
        

    return X_full, U_full, t, J_list, failed_count, converged