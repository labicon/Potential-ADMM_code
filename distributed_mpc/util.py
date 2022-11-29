import numpy as np
import itertools
from casadi import *

def paper_setup_3_quads():
    x0 = np.array([[0.5, 1.5, 1, 0, 0, 0,
                    2.5, 1.5, 1, 0, 0, 0,
                    1.5, 1.3, 1, 0, 0, 0]], 
                     dtype=float).T
    xf = np.array([[2.5, 1.5, 1, 0, 0, 0, 
                    0.5, 1.5, 1, 0, 0, 0, 
                    1.5, 2.2, 1, 0, 0, 0]]).T
    # x0[dec.pos_mask([6]*3, 3)] += 0.01*np.random.randn(9, 1)
    # xf[dec.pos_mask([6]*3, 3)] += 0.01*np.random.randn(9, 1)
    return x0, xf


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
eps = 1e-3
def compute_pairwise_distance_Sym(X, x_dims, n_d=3):
    """Compute the distance between each pair of agents"""
    assert len(set(x_dims)) == 1

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
        
    else:
        dX = X_agent[:n_d, pair_inds[:, 0]] - X_agent[:n_d, pair_inds[:, 1]]
        for j in range(dX.shape[1]):
            distances.append(sqrt(dX[0,j]**2+dX[1,j]**2+dX[2,j]**2 + eps))
            
    return distances #this is a list of symbolic pariwise distances


def define_inter_graph_threshold(X, radius, x_dims, ids):
    """Compute the interaction graph based on a simple thresholded distance
    for each pair of agents sampled over the trajectory
    """

    planning_radii = 2 * radius
    rel_dists = compute_pairwise_distance_Sym(X, x_dims) #this is a list

    N = X.shape[1]
    n_samples = 10
    # sample_step = max(N // n_samples, 1)
    # sample_slice = slice(0, N + 1, sample_step)

    # Put each pair of agents within each others' graphs if they are within
    # some threshold distance from each other.
    graph = {id_: [id_] for id_ in ids}
    pair_inds = np.array(list(itertools.combinations(ids, 2)))
    for i, pair in enumerate(pair_inds):
        if np.any(rel_dists[i] < planning_radii):
            graph[pair[0]].append(pair[1])
            graph[pair[1]].append(pair[0])

    graph = {agent_id: sorted(prob_ids) for agent_id, prob_ids in graph.items()}
    return graph
