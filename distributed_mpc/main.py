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

max_input_base = np.array([[theta_max], [phi_max], [tau_max]])
min_input_base = np.array([[theta_min], [phi_min], [tau_min]])
max_state_base = np.array([[x_max], [y_max], [z_max], [v_max],[v_max], [v_max]])
min_state_base = np.array([[x_min], [y_min], [z_min], [v_min],[v_min], [v_min]])

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
