import casadi as cs
import numpy as np
from scipy.constants import g

from util import (
    compute_pairwise_distance,
    compute_pairwise_distance_Sym,
    define_inter_graph_threshold,
    distance_to_goal,
    split_graph,
)

theta_max = np.pi / 6
phi_max = np.pi / 6

v_max = 3
v_min = -3

theta_min = -np.pi / 6
phi_min = -np.pi / 6

tau_max = 15
tau_min = 0

x_min = -5
x_max = 5

y_min = -5
y_max = 5

z_min = 0
z_max = 3.0


def generate_f(x_dims_local):
    
    # NOTE: Assume homogeneity of agents.
    n_agents = len(x_dims_local)
    n_states = x_dims_local[0]
    n_controls = 3
    
    def f(x, u):
        x_dot = cs.MX.zeros(x.numel())
        for i_agent in range(n_agents):
            i_xstart = i_agent * n_states
            i_ustart = i_agent * n_controls
            x_dot[i_xstart:i_xstart + n_states] = cs.vertcat(
                x[i_xstart + 3: i_xstart + 6],
                g*cs.tan(u[i_ustart]), -g*cs.tan(u[i_ustart+1]), u[i_ustart+2] - g
                )
            
        return x_dot
    
    return f


def objective(X, U, u_ref, xf, Q, R, Qf):
    total_stage_cost = 0
    for j in range(X.shape[1] - 1):
        for i in range(X.shape[0]):
            total_stage_cost += (X[i, j] - xf[i]) * Q[i, i] * (X[i, j] - xf[i])

    for j in range(U.shape[1]):
        for i in range(U.shape[0]):
            total_stage_cost += (U[i, j] - u_ref[i]) * R[i, i] * (U[i, j] - u_ref[i])

    # Quadratic terminal cost:
    total_terminal_cost = 0

    for i in range(X.shape[0]):
        total_terminal_cost += (X[i, -1] - xf[i]) * Qf[i, i] * (X[i, -1] - xf[i])

    return total_stage_cost + total_terminal_cost


def generate_min_max_input(inputs_dict, n_inputs):

    theta_max = np.pi / 6
    phi_max = np.pi / 6

    # v_max = 3
    # v_min = -3

    theta_min = -np.pi / 6
    phi_min = -np.pi / 6

    tau_max = 15
    tau_min = 0

    n_agents = [u.shape[0] // n_inputs for u in inputs_dict.values()]

    u_min = np.array([[theta_min, phi_min, tau_min]])
    u_max = np.array([[theta_max, phi_max, tau_max]])

    return [
        (np.tile(u_min, n_agents_i), np.tile(u_max, n_agents_i))
        for n_agents_i in n_agents
    ]


def generate_min_max_state(states_dict, n_states):

    x_min = -5
    x_max = 5

    y_min = -5
    y_max = 5

    z_min = 0
    z_max = 3.0

    n_agents = [x.shape[0] // n_states for x in states_dict.values()]
    x_min = np.array([[x_min, y_min, z_min, v_min, v_min, v_min]])
    x_max = np.array([[x_max, y_max, z_max, v_max, v_max, v_max]])

    return [
        (np.tile(x_min, n_agents_i), np.tile(x_max, n_agents_i))
        for n_agents_i in n_agents
    ]


def solve_rhc_distributed(
    x0, xf, u_ref, N, Q, R, Qf, n_agents, n_states, n_inputs, radius, ids
):

    x_dims = [n_states] * n_agents
    u_dims = [n_inputs] * n_agents

    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "print_level": 0}

    M = 100  # this is the entire fixed horizon

    n_x = n_agents * n_states
    n_u = n_agents * n_inputs
    t = 0

    J_list = []
    J_list.append(np.inf)
    # for i in range(M) :
    loop = 0
    dt = 0.1

    X_full = np.zeros((0, n_x))
    U_full = np.zeros((0, n_u))

    while np.any(distance_to_goal(x0, xf, n_agents, n_states) > 0.1) and (loop < M):
        # print(f'dis to goal is {distance_to_goal(x0,xf,n_agents,n_states)}')
        ######################################################################
        # Determine sub problems to solve:

        # compute interaction graph at the current time step:
        if loop > 0:
            print(f"re-optimizing at {x0.T}")

        rel_dists = compute_pairwise_distance(x0, x_dims, n_d=3)

        graph = define_inter_graph_threshold(x0, radius, x_dims, ids)

        print(
            f"current interaction graph is {graph}, the pairwise distances between each agent is {rel_dists}"
        )
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

        min_max_input_list = generate_min_max_input(inputs, n_inputs)
        min_max_state_list = generate_min_max_state(states, n_states)

        ##########################################################################
        # Solve each sub-problem in a sequential manner:
        # TODO: parallel computation for better speed?

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

            print(f"Solving the {count}th sub-problem at iteration {loop}, t = {t}")

            min_states, max_states = state_boundsi
            min_inputs, max_inputs = input_boundsi

            di.minimize(costi)

            n_states_local = statesi.shape[
                0
            ]  # each subproblem has different number of states
            # print(f'n_states_local:{n_states_local}')
            n_inputs_local = inputsi.shape[0]
            x_dims_local = [int(n_states)] * int(n_states_local / n_states)

            print(f"current sub-problem has state dimension : {x_dims_local}")
            # u_dims_local =  [int(n_inputs_local/(n_inputs_local/n_inputs))]*int(n_inputs_local/n_inputs)
            # i.e, [6,6] if the current sub-problem has 2 agents combined, or [6,6,6] if 3 agents are combined

            f = generate_f(x_dims_local)

            for k in range(N):  # loop over control intervals
                # Runge-Kutta 4 integration

                k1 = f(statesi[:, k], inputsi[:, k])
                k2 = f(statesi[:, k] + dt / 2 * k1, inputsi[:, k])
                k3 = f(statesi[:, k] + dt / 2 * k2, inputsi[:, k])
                k4 = f(statesi[:, k] + dt * k3, inputsi[:, k])
                x_next = statesi[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                di.subject_to(statesi[:, k + 1] == x_next)  # close the gaps

                di.subject_to(inputsi[:, k] <= max_inputs.T)
                di.subject_to(min_inputs.T <= inputsi[:, k])

            for k in range(N + 1):

                di.subject_to(statesi[:, k] <= max_states.T)
                di.subject_to(min_states.T <= statesi[:, k])

                # DBG
                # distances, d_test = compute_pairwise_distance_Sym(statesi[:,k], x_dims_local)
                # print(distances, d_test)

                # collision avoidance over control horizon (only if the current sub-problem contains the states of more than 1 agent):

                if len(x_dims_local) != 1:
                    distances = compute_pairwise_distance_Sym(
                        statesi[:, k], x_dims_local
                    )
                    for n in distances:
                        di.subject_to(n >= radius)

            # equality constraints for initial condition:
            di.subject_to(
                statesi[:, 0] == split_problem_states_initial[count].reshape(-1, 1)
            )

            di.solver("ipopt", p_opts, s_opts)

            sol = di.solve()

            objective_val += sol.value(costi)
            print(
                f"objective value for the {count}th subproblem at iteration {loop} is {sol.value(costi)}"
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

        # PROBLEM RIGHT HERE!!! somehow I get a bunch of unexpected zeros at this step (in the current solution x0)
        x0 = X_dec.reshape(-1, 1)
        print(f"current collected solution is {x0.T}#")

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

        if abs(J_list[loop] - J_list[loop - 1]) <= 1:
            print(f"Terminated! at loop = {loop}")
            break

        # if loop == 5:
        #     break

    return X_full, U_full, t


def solve_rhc(x0, xf, u_ref, N, Qf):
    # N is the shifting prediction horizon

    p_opts = {"expand": True}
    s_opts = {"max_iter": 100, "print_level": 0}

    opti = cs.Opti()
    M = 100  # this is the entire fixed horizon

    n_x = x0.size
    n_u = u_ref.size

    X_full = np.zeros((0, n_x))
    U_full = np.zeros((0, n_u))

    t = 0

    J_list = []
    J_list.append(np.inf)
    # for i in range(M) :
    i = 0

    f = lambda x, u: cs.cs.vertcat(
        x[3], x[4], x[5], g * cs.tan(u[0]), -g * cs.tan(u[1]), u[2] - g
    )  # dx/dt = f(x,u)

    dt = 0.05

    while (np.linalg.norm(x0[0:3] - xf[0:3]) > 0.1) and (i < M):

        X = opti.variable(6, N + 1)
        U = opti.variable(3, N)

        cost_fun = objective(X, U, u_ref, xf, Q, R, Qf)
        opti.minimize(cost_fun)

        for k in range(N):  # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = f(X[:, k], U[:, k])
            k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
            k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
            k4 = f(X[:, k] + dt * k3, U[:, k])
            x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

        opti.subject_to(X[2, :] <= 3.0)  # altitude p_z is limited
        opti.subject_to(0.4 <= X[2, :])

        opti.subject_to(X[0, :] <= 3)  # p_x is limited
        opti.subject_to(-3 <= X[0, :])

        opti.subject_to(X[1, :] <= 3)  # p_y is limited
        opti.subject_to(-3 <= X[1, :])

        opti.subject_to(X[3, :] <= 5)  # Vx is limited
        opti.subject_to(0 <= X[3, :])

        opti.subject_to(X[4, :] <= 5)  # Vy is limited
        opti.subject_to(0 <= X[4, :])

        opti.subject_to(X[5, :] <= 5)  # Vz is limited
        opti.subject_to(0 <= X[5, :])

        opti.subject_to(U[0, :] <= np.pi / 6)  # theta is limited
        opti.subject_to(-np.pi / 6 <= U[0, :])

        opti.subject_to(U[1, :] <= np.pi / 6)  # phi is limited
        opti.subject_to(-np.pi / 6 <= U[1, :])

        opti.subject_to(U[2, :] <= 20)  # tau is limited
        opti.subject_to(0 <= U[2, :])  # minimum force keeps the drone at hover

        # equality constraints for initial condition:
        opti.subject_to(X[:, 0] == x0)

        opti.solver("ipopt", p_opts, s_opts)

        sol = opti.solve()
        x0 = sol.value(X)[:, 1]
        u_sol = sol.value(U)[:, 0]
        J_list.append(sol.value(cost_fun))

        # Store the trajectory

        X_full = np.r_[X_full, x0.reshape(1, -1)]
        U_full = np.r_[U_full, u_sol.reshape(1, -1)]

        t += dt
        i += 1

        if abs(J_list[i] - J_list[i - 1]) <= 1:
            print(f"Terminated! at i = {i}")
            break

    return X_full, U_full, t
