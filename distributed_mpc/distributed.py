import logging
from time import perf_counter as pc
import itertools
import logging
import multiprocessing as mp
import numpy as np
from problem import *
from .util import *


def solve_decentralized(problem, x_dims,u_dims, X, U, radius, t_kill=None, pool=None, verbose=True, **kwargs):
    """Solve the problem via decentralization into subproblems"""

    #x_dims, u_dims are lists
    
    N = U.shape[0]
    n_states = x_dims[0]
    n_controls = u_dims[0]
    n_agents = len(x_dims)
    ids = problem.ids
    solve_info = {}

    # Compute interaction graph based on relative distances.
    graph = define_inter_graph_threshold(X, radius, x_dims, ids)
    if verbose:
        print("=" * 80 + f"\nInteraction Graph: {graph}")

    # Split up the initial state and control for each subproblem.
    x0_split = split_graph(X[np.newaxis, 0], x_dims, graph)
    U_split = split_graph(U, u_dims, graph)

    X_dec = np.zeros((N + 1, n_agents * n_states))
    U_dec = np.zeros((N, n_agents * n_controls))

    # Solve all problems in one process, keeping results for each agent in *_dec.

    if not pool:
        for i, (subproblem, x0i, Ui, id_) in enumerate(
            zip(problem.split(graph), x0_split, U_split, ids)
        ):

            t0 = pc()
            Xi_agent, Ui_agent, id_ = solve_subproblem(
                (subproblem, x0i, Ui, id_, False), **kwargs
            )
            Δt = pc() - t0

            if verbose:
                print(f"Problem {id_}: {graph[id_]}\nTook {Δt} seconds\n")

            X_dec[:, i * n_states: (i + 1) * n_states] = Xi_agent
            U_dec[:, i * n_controls: (i + 1) * n_controls] = Ui_agent

            solve_info[id_] = (Δt, graph[id_])

    # Solve in separate processes using imap.
    else:
        # Package up arguments for the subproblem solver.
        args = zip(problem.split(graph), x0_split, U_split, ids, [verbose] * len(graph))

        t0 = pc()
        for i, (Xi_agent, Ui_agent, id_) in enumerate(
            pool.imap_unordered(solve_subproblem, args)
        ):

            Δt = pc() - t0
            if verbose:
                print(f"Problem {id_}: {graph[id_]}\nTook {Δt} seconds")
            X_dec[:, i * n_states: (i + 1) * n_states] = Xi_agent
            U_dec[:, i * n_controls: (i + 1) * n_controls] = Ui_agent

            # NOTE: This cannot be compared to the single-processed version due to
            # multi-processing overhead.
            solve_info[id_] = (Δt, graph[id_])

    # Evaluate the cost of this combined trajectory.
    full_solver = ilqrSolver(problem, N)
    _, J_full = full_solver._rollout(X[0], U_dec)

    return X_dec, U_dec, J_full, solve_info