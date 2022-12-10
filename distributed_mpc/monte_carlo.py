#!/usr/bin/env python

"""Benchmark of the performance of centralized vs. decentralized potential iLQR

We conduct two primary analyses in this script, namely:
1. Allow unlimited solve time and stop after the solver converges or diverges.
2. Cap the solve time based on a "real-time" constraint.

The objective in 1 is to contrast solve times, whereas in 2 we contrast trajectory
quality in a real-time application of the algorithm. For both cases, we utilize
uniform random initial positions with stationary agents.

"""

import logging
from pathlib import Path
import multiprocessing as mp
from os import getpid
from time import strftime

import numpy as np

from util import *
from decentralized import random_setup
from distributed_mpc import *
from centralized_mpc import *

#Define simulation parameters:
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

u_ref_base = np.array([0,0,g])

max_input_base = np.array([[theta_max], [phi_max], [tau_max]])
min_input_base = np.array([[theta_min], [phi_min], [tau_min]])
max_state_base = np.array([[x_max], [y_max], [z_max], [v_max],[v_max], [v_max]])
min_state_base = np.array([[x_min], [y_min], [z_min], [v_min],[v_min], [v_min]])

radius = 0.5
N = 15
n_states = 6
n_inputs = 3


def multi_agent_run(n_agents,dt, N, radius, energy=15):
    """Single simulation comparing the centralized and decentralized solvers"""
    
    if n_agents == 3:
        x0,xf = paper_setup_3_quads()
        
    elif n_agents == 5:
        x0,xf = paper_setup_5_quads()
        
    elif n_agents == 10:
        x0,xf = paper_setup_10_quads()
    
    ids = [100 + i for i in range(n_agents)]
    
    Q = np.eye(n_states*n_agents) * 100
    R = np.eye(n_inputs*n_agents)*0.01

    Qf = 1000.0 * np.eye(Q.shape[0])

   
    # Solve the problem centralized.
    print("\t\t\tcentralized")
    max_input = np.tile(max_input_base,n_agents)
    min_input = np.tile(min_input_base,n_agents)
    max_state = np.tile(max_state_base,n_agents)
    min_state = np.tile(min_state_base,n_agents)
    u_ref = np.tile(u_ref_base,n_agents)
    Xc, Uc, tc , J_fc= solve_rhc(x0,xf,u_ref,
                           N,Q,R,Qf,n_agents,
                           n_states,n_inputs,
                           radius,max_input,
                           min_input,max_state,
                           min_state)

    # Solve the problem decentralized.
    print("\t\t\tdistributed")

    Xd, Ud, td, J_fd = solve_rhc_distributed(
            x0, xf, u_ref, N,  
            n_agents, n_states, n_inputs, radius, ids,
            x_min,x_max,y_min,y_max,z_min,z_max,v_min,
            v_max,theta_max,
            theta_min,tau_max,
            tau_min,phi_max,phi_min
                            )


def setup_logger():
    
    LOG_PATH = Path(__file__).parent.parent / "logs"
    LOG_FILE = LOG_PATH / strftime(
        "dec-mc-_%m-%d-%y_%H.%M.%S_{getpid()}.csv"
    )
    if not LOG_PATH.is_dir():
        LOG_PATH.mkdir()
    print(f"Logging results to {LOG_FILE}")
    logging.basicConfig(filename=LOG_FILE, format="%(message)s", level=logging.INFO)
    logging.info(
        "n_agents,trial,centralized,J,dt,t,ids"
        "dist_left"
    )


def monte_carlo_analysis():
    """Benchmark to evaluate algorithm over many random initial conditions"""

    setup_logger()

    n_trials_iter = range(3)
    n_agents_iter = [3, 5, 10]

    dt = 0.1
    N = 15
    ENERGY = 15.0
    radius = 0.5

    # Change the for loops into multi-processing?

    for n_agents in n_agents_iter:
        print(f"\tn_agents: {n_agents}")
        if n_agents >5:
            radius = 0.25
        for i_trial in n_trials_iter:
            print(f"\t\ttrial: {i_trial}")
            
            multi_agent_run(
                n_agents, dt, N, radius, energy=15.0
            )
    
def main():
    monte_carlo_analysis()


if __name__ == "__main__":
    main()


 