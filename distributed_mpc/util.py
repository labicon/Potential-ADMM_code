import numpy as np
import itertools
from casadi import *
import casadi as cs
import decentralized as dec
from decentralized import random_setup

def paper_setup_3_quads(random = False):
    
    x0 = np.array([[0.5, 1.5, 1, 0, 0, 0,
                    2.5, 1.5, 1, 0, 0, 0,
                    1.5, 1.3, 1, 0, 0, 0]], 
                     dtype=float).T
    xf = np.array([[2.5, 1.5, 1, 0, 0, 0, 
                    0.5, 1.5, 1, 0, 0, 0, 
                    1.5, 2.2, 1, 0, 0, 0]]).T
    if random == True:
        x0[dec.pos_mask([6]*3, 3)] += 0.05*np.random.randn(9, 1)
        xf[dec.pos_mask([6]*3, 3)] += 0.05*np.random.randn(9, 1)
    return x0, xf

def setup_3_quads():
    
    x0,xf = random_setup(3,6,n_d=3,energy=3,var=3.0)
    
    for i in range(2,len(x0),6):
        if x0[i] <= 0.5:
            x0[i] = 1.2 + np.random.rand(1,)

        if xf[i] <= 0.5:
            xf[i] = 1.1 + np.random.rand(1,)*0.5

    return x0, xf

 
def setup_4_quads():
    
    x0,xf = random_setup(4,6,n_d=3,energy=4,var=3.0)
    
    for i in range(2,len(x0),6):
        if x0[i] <= 0.5:
            x0[i] = 1.2 + np.random.rand(1,)

        if xf[i] <= 0.5:
            xf[i] = 1.1 + np.random.rand(1,)*0.5

    return x0, xf

def paper_setup_5_quads(random = False):
    x0 = np.array([[-0.182, -0.545,  1.161,  0.   ,  0.   ,  0.   ,  1.335,  1.484,
         0.5  ,  0.   ,  0.   ,  0.   , -0.97 , -0.831,  2.295,  0.   ,
         0.   ,  0.   , -1.144, -1.193,  1.7  ,  0.   ,  0.   ,  0.   ,
         0.961,  1.085,  0.88 ,  0.   ,  0.   ,  0.   ]]).T
    
    xf =  np.array([[-1.751,  0.674, 1.193,  0.   ,  0.   ,  0.   ,  1.769,  0.102,
         2.998,  0.   ,  0.   ,  0.   , -1.452, -0.02 , 1.11 ,  0.   ,
         0.   ,  0.   ,  0.34 , -0.993, 0.93,  0.   ,  0.   ,  0.   ,
         1.094,  0.237,  1.5,  0.   ,  0.   ,  0.   ]]).T
    if random == True:
        x0[dec.pos_mask([6]*5, 3)] += 0.05*np.random.randn(15, 1)
        xf[dec.pos_mask([6]*5, 3)] += 0.05*np.random.randn(15, 1)
    
    return x0,xf


def setup_5_quads():

    x0,xf = random_setup(5,6,n_d=3,energy=5,var=3.0)
    
    for i in range(2,len(x0),6):
        if x0[i] <= 0.5:
            x0[i] = 1.2 + np.random.rand(1,)

        if xf[i] <= 0.5:
            xf[i] = 1.1 + np.random.rand(1,)*0.5
    
    return x0,xf

def setup_6_quads():

    x0,xf = random_setup(6,6,n_d=3,energy=5,var=2.5)
    
    for i in range(2,len(x0),6):
        if x0[i] <= 0.5:
            x0[i] = 1.2 + np.random.rand(1,)

        if xf[i] <= 0.5:
            xf[i] = 1.1 + np.random.rand(1,)*0.5

    return x0, xf

def setup_7_quads():

    x0,xf = random_setup(7,6,n_d=3,energy=7,var=3.0)
    
    for i in range(2,len(x0),6):
        if x0[i] <= 0.5:
            x0[i] = 1.2 + np.random.rand(1,)

        if xf[i] <= 0.5:
            xf[i] = 1.1 + np.random.rand(1,)*0.5

    return x0, xf

def setup_8_quads():

    x0,xf = random_setup(8,6,n_d=3,energy=8,var=3.0)
    
    for i in range(2,len(x0),6):
        if x0[i] <= 0.5:
            x0[i] = 1.2 + np.random.rand(1,)

        if xf[i] <= 0.5:
            xf[i] = 1.1 + np.random.rand(1,)*0.5

    return x0, xf

def setup_9_quads():
    x0,xf = random_setup(9,6,n_d=3,energy=8,var=3.0)
    
    for i in range(2,len(x0),6):
        if x0[i] <= 0.5:
            x0[i] = 1.2 + np.random.rand(1,)

        if xf[i] <= 0.5:
            xf[i] = 1.1 + np.random.rand(1,)*0.5

    return x0, xf

def setup_10_quads():
    
    x0,xf = random_setup(10,6,n_d=3,energy=8,var=3.0)
    
    for i in range(2,len(x0),6):
        if x0[i] <= 0.5:
            x0[i] = 1.2 + np.random.rand(1,)

        if xf[i] <= 0.5:
            xf[i] = 1.1 + np.random.rand(1,)*0.5

    return x0, xf

    
    
def paper_setup_10_quads(random = False):
    
    x0 = np.array([[ 0.357,  0.799,  1.504,  0.   ,  0.   ,  0.   ,  2.172,  2.283,
         1.436,  0.   ,  0.   ,  0.   , -0.085,  0.577,  2.01,  0.   ,
         0.   ,  0.   ,  0.378,  0.254,  1.66,  0.   ,  0.   ,  0.   ,
         0.184,  0.344,  2.139,  0.   ,  0.   ,  0.   ,  2.094,  2.089,
         1.304,  0.   ,  0.   ,  0.   , -2.219, -3.09 ,  1.487,  0.   ,
         0.   ,  0.   , -0.6  , -0.406,  1.319,  0.   ,  0.   ,  0.   ,
        -2.059, -3.279,  1.48 ,  0.   ,  0.   ,  0.   , -0.222,  0.43 ,
         2.325,  0.   ,  0.   ,  0.   ]]).T
    xf = np.array([[ 1.115,  1.749,  2.1,  0.   ,  0.   ,  0.   ,  0.653,  1.288,
         1.529,  0.   ,  0.   ,  0.   , -1.373, -0.243,  1.9,  0.   ,
         0.   ,  0.   , -1.314,  0.229,  1.771,  0.   ,  0.   ,  0.   ,
         1.735,  1.558,  2.081,  0.   ,  0.   ,  0.   ,  0.852, -1.583,
         2.264,  0.   ,  0.   ,  0.   ,  0.387, -1.5  ,  1.685,  0.   ,
         0.   ,  0.   , -1.26 , -0.697,  1.85,  0.   ,  0.   ,  0.   ,
        -0.213, -0.138,  1.545,  0.   ,  0.   ,  0.   , -0.582, -0.663,
         1.69,  0.   ,  0.   ,  0.   ]]).T
    
    if random == True:
        
        x0[dec.pos_mask([6]*10, 3)] += 0.01*np.random.randn(30, 1)
        xf[dec.pos_mask([6]*10, 3)] += 0.01*np.random.randn(30, 1)

    return x0,xf

def paper_setup_15_quads():
    
    x0,xf = random_setup(15,6,n_d=3,energy=15,var=15/2)
    
    for i in range(2,len(x0),6):
        if x0[i] <= 0.5:
            x0[i] = 1.2 + np.random.rand(1,)

        if xf[i] <= 0.5:
            xf[i] = 1.1 + np.random.rand(1,)*0.5
        
    return x0,xf

def paper_setup_20_quads():
    
    x0,xf = random_setup(20,6,n_d=3,energy=55,var=20/2)
    
    for i in range(2,len(x0),6):
        if x0[i] <= 0.5:
            x0[i] = 1.2 + np.random.rand(1,)

        if xf[i] <= 0.5:
            xf[i] = 1.4 + np.random.rand(1,)*0.1
        
    return x0,xf


def generate_f(x_dims_local):
    g = 9.8
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

def generate_f_human_drone(x_dims_local,n_human):
    g = 9.8
    # NOTE: Assume homogeneity of agents.
    n_agents = len(x_dims_local)
    n_states = x_dims_local[0]
    n_controls = 3
    
    def f(x, u):
        x_dot = cs.MX.zeros(x.numel())
        for i_agent in range(0,n_agents-n_human):
            i_xstart = i_agent * n_states
            i_ustart = i_agent * n_controls
            x_dot[i_xstart:i_xstart + n_states] = cs.vertcat(
                x[i_xstart + 3: i_xstart + 6],
                g*cs.tan(u[i_ustart]), -g*cs.tan(u[i_ustart+1]), u[i_ustart+2] - g
                )
        # count = 0
        for j_agent in range(n_agents-n_human,n_agents):
            j_xstart = j_agent * n_states
            j_ustart = j_agent * n_controls #human agent has 3 control var.
       
            x_dot[j_xstart:j_xstart + n_states] = cs.vertcat(
                x[j_xstart + 3]*cs.cos(u[j_ustart]), x[j_xstart+3]*cs.sin(u[j_ustart]),0,
                u[j_ustart+1], 0 , 0
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

def generate_min_max_input(inputs_dict, n_inputs,theta_max,
                          theta_min,tau_max,tau_min,phi_max,phi_min,human_count = None):

    n_agents = [u.shape[0] // n_inputs for u in inputs_dict.values()]

    u_min = np.array([[theta_min, phi_min, tau_min]])
    u_max = np.array([[theta_max, phi_max, tau_max]])
    
    if human_count:
        return [
        (np.tile(u_min, n_agents_i), np.tile(u_max, n_agents_i))
        for n_agents_i in (n_agents-human_count)
        ]
    
    else:
        return [
            (np.tile(u_min, n_agents_i), np.tile(u_max, n_agents_i))
            for n_agents_i in n_agents
        ]


def generate_min_max_state(states_dict, n_states, x_min,
                          x_max,y_min,y_max,z_min,z_max,v_min,v_max,human_count = None):

    n_agents = [x.shape[0] // n_states for x in states_dict.values()]
    x_min = np.array([[x_min, y_min, z_min, v_min, v_min, v_min]])
    x_max = np.array([[x_max, y_max, z_max, v_max, v_max, v_max]])
    
    if human_count:
        return [
        (np.tile(x_min, n_agents_i), np.tile(x_max, n_agents_i))
        for n_agents_i in (n_agents-human_count)
        ]
    
    else:
        
        return [
            (np.tile(x_min, n_agents_i), np.tile(x_max, n_agents_i))
            for n_agents_i in n_agents
        ]


def distance_to_goal(x,xf,n_agents,n_states):
    n_d = 3 
    return np.linalg.norm((x - xf).reshape(n_agents, n_states)[:, :n_d], axis=1)


def split_agents(Z, z_dims):
    """Partition a cartesian product state or control for individual agents"""
    return np.split(np.atleast_2d(Z), np.cumsum(z_dims[:-1]), axis=1)


def split_agents_gen(z, z_dims):
    """Generator version of ``split_agents``"""
    dim = z_dims[0]
    for i in range(len(z_dims)):
        yield z[i * dim : (i + 1) * dim]


def split_graph(Z, z_dims, graph):
    """Split up the state or control by grouping their ID's according to the graph"""
    assert len(set(z_dims)) == 1

    # Create a mapping from the graph to indicies.
    mapping = {id_: i for i, id_ in enumerate(list(graph))}

    n_z = z_dims[0]
    z_split = []
    for ids in graph.values():
        inds = [mapping[id_] for id_ in ids]
        z_split.append(
            np.concatenate([Z[:, i * n_z : (i + 1) * n_z] for i in inds], axis=1)
        )

    return z_split


def define_inter_graph_threshold(X, radius, x_dims, ids, n_dims=None):
    """Compute the interaction graph based on a simple thresholded distance
    for each pair of agents sampled over the trajectory
    """

    planning_radii = 2 * radius
    
    if n_dims:
        rel_dists = np.array([compute_pairwise_distance_nd_Sym(X, x_dims, n_dims)])
    else:    
        rel_dists = compute_pairwise_distance(X, x_dims)
    
    print(f'determining interaction graph with the following pair-wise distance : {rel_dists}')
    # N = X.shape[0]
    # n_samples = 10
    # sample_step = max(N // n_samples, 1)
    # sample_slice = slice(0, N + 1, sample_step)

    # Put each pair of agents within each others' graphs if they are within
    # some threshold distance from each other.
    graph = {id_: [id_] for id_ in ids}
    # print(graph)
    pair_inds = np.array(list(itertools.combinations(ids, 2)))
    for i, pair in enumerate(pair_inds):
        if np.any(rel_dists[:,i] < planning_radii):
            graph[pair[0]].append(pair[1])
            graph[pair[1]].append(pair[0])

    graph = {agent_id: sorted(prob_ids) for agent_id, prob_ids in graph.items()}
    return graph



def compute_pairwise_distance_Sym(X, x_dims, n_d=3):
    """Compute the distance between each pair of agents"""
    assert len(set(x_dims)) == 1
    eps = 1e-3
    n_agents = len(x_dims)
    n_states = x_dims[0]

    if n_agents == 1:
        raise ValueError("Can't compute pairwise distance for one agent.")  
    
    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    
    X_agent = reshape(X,(n_agents, n_states))
    distances = []
    
    if n_agents == 2:
        dX=X_agent[0,0:3]-X_agent[1,0:3]
        distances.append(sqrt(dX[0]**2+dX[1]**2+dX[2]**2 + eps))
    
    if n_agents > 2 and n_agents <=5:
        
        dX = X_agent[:n_d, pair_inds[:, 0]] - X_agent[:n_d, pair_inds[:, 1]]
        for j in range(dX.shape[1]):
            distances.append(sqrt(dX[0,j]**2+dX[1,j]**2+dX[2,j]**2 + eps))
        
        
    if n_agents > 5:
    
        dX = X_agent.T[:n_d, pair_inds[:, 0]] - X_agent.T[:n_d, pair_inds[:, 1]]
        for j in range(dX.shape[1]):
            distances.append(sqrt(dX[0,j]**2+dX[1,j]**2+dX[2,j]**2 + eps))

            
    return distances #this is a list of symbolic pariwise distances

def compute_pairwise_distance_nd_Sym(X, x_dims, n_dims):
    """Analog to the above whenever some agents only use distance in the x-y plane"""
    CYLINDER_RADIUS = 0.2

    n_states = x_dims[0]
    n_agents = len(x_dims)
    distances = []
    eps = 1e-3

    for i, n_dim_i in zip(range(n_agents), n_dims):
        for j, n_dim_j in zip(range(i + 1, n_agents), n_dims[i + 1 :]):
            n_dim = min(n_dim_i, n_dim_j)
            # print(i,j)
            Xi = X[i * n_states : i * n_states + n_dim, :]
            Xj = X[j * n_states : j * n_states + n_dim, :]
            dX = Xi-Xj
            # print(dX.shape)
            if n_dim == 3:
                distances.append(sqrt(dX[0,:]**2+dX[1,:]**2+dX[2,:]**2+eps))
            else:
                distances.append(sqrt(dX[0,:]**2+dX[1,:]**2 + eps)+CYLINDER_RADIUS)
    
    return distances





def compute_pairwise_distance(X, x_dims, n_d=3):
    """Compute the distance between each pair of agents"""
    assert len(set(x_dims)) == 1

    n_agents = len(x_dims)
    n_states = x_dims[0]

    if n_agents == 1:
        raise ValueError("Can't compute pairwise distance for one agent.")

    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    X_agent = X.reshape(-1, n_agents, n_states).swapaxes(0, 2)
    dX = X_agent[:n_d, pair_inds[:, 0]] - X_agent[:n_d, pair_inds[:, 1]]
    return np.linalg.norm(dX, axis=0).T