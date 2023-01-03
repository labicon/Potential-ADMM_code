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
import os
from time import strftime

import numpy as np

from util import *
import util
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
z_max = 3.5

u_ref_base = np.array([0,0,g])

max_input_base = np.array([[theta_max], [phi_max], [tau_max]])
min_input_base = np.array([[theta_min], [phi_min], [tau_min]])
max_state_base = np.array([[x_max], [y_max], [z_max], [v_max],[v_max], [v_max]])
min_state_base = np.array([[x_min], [y_min], [z_min], [v_min],[v_min], [v_min]])

radius = 0.5
N = 15
n_states = 6
n_inputs = 3


def multi_agent_run(trial, n_agents,dt, N, radius, centralized = False):
    """Single simulation comparing the centralized and decentralized solvers"""
    
    if n_agents == 3:
        x0,xf = util.paper_setup_3_quads(True)
        n_dims = [3]*3
    elif n_agents==4:
        x0, xf=util.paper_setup_4_quads(True)
        n_dims=[3]*4
    elif n_agents == 5:
        x0,xf = util.paper_setup_5_quads(True)
        n_dims = [3]*5
    elif n_agents==6:
        x0, xf=util.paper_setup_6_quads(True)
        n_dims= [3]*6
    elif n_agents==7:
        x0, xf=util.paper_setup_7_quads(True)
        n_dims= [3]*7
    elif n_agents==8:
        x0, xf= util.paper_setup_8_quads(True)
        n_dims= [3]*8
    elif n_agents==9:
        x0, xf=util.paper_setup_9_quads(True)
        x_dims= [3]*9
    elif n_agents == 10:
        x0,xf = util.paper_setup_10_quads(True)
        n_dims = [3]*10
    elif n_agents == 15:
        x0,xf = util.paper_setup_15_quads()
        n_dims = [3]*15
    elif n_agents == 20:
        x0,xf = util.paper_setup_20_quads()
        n_dims = [3]*20
        
    
    ids = [100 + i for i in range(n_agents)]
    
    Q = np.eye(n_states*n_agents) * 100
    R = np.eye(n_inputs*n_agents) * 0.01

    Qf = 1000.0 * np.eye(Q.shape[0])
    
    
    max_input = np.tile(max_input_base,n_agents)
    min_input = np.tile(min_input_base,n_agents)
    max_state = np.tile(max_state_base,n_agents)
    min_state = np.tile(min_state_base,n_agents)

    u_ref = np.tile(u_ref_base,n_agents)
   
    # Solve the problem centralized.
    if centralized == True:
        print("\t\t\tcentralized")
        

        Xc, Uc, tc , J_c, failed_count, converged = solve_rhc(trial,x0,xf,u_ref,
                               N,Q,R,Qf,n_agents,
                               n_states,n_inputs,
                               radius,max_input,
                               min_input,max_state,
                               min_state)

        
    else:
        print("\t\t\tdistributed")

        Xd, Ud, td, J_d , failed_count, converged = solve_rhc_distributed(
                trial,x0, xf, u_ref, N,  
                n_agents, n_states, n_inputs, radius, ids,
                x_min,x_max,y_min,y_max,z_min,z_max,v_min,
                v_max,theta_max,
                theta_min,tau_max,
                tau_min,phi_max,phi_min,0,n_dims
                                )

def setup_logger(centralized=False):
    
    if centralized == True:
        
        LOG_PATH = Path(__file__).parent.parent / "logs"
        LOG_FILE = LOG_PATH / strftime(
            "cen-mpc-_%m-%d-%y_%H.%M.%S_{getpid()}.csv"
        )
        if not LOG_PATH.is_dir():
            LOG_PATH.mkdir()
        print(f"Logging results to {LOG_FILE}")
        logging.basicConfig(filename=LOG_FILE, format="%(message)s", level=logging.INFO)
        logging.info(
            "i_trial,n_agents,t,failed_count,converged,objective_val,N,dt"
        )
        
    else:
    
        LOG_PATH = Path(__file__).parent.parent / "logs"
        LOG_FILE = LOG_PATH / strftime(
            "dec-mpc-_%m-%d-%y_%H.%M.%S_{getpid()}.csv"
        )
        if not LOG_PATH.is_dir():
            LOG_PATH.mkdir()
        print(f"Logging results to {LOG_FILE}")
        logging.basicConfig(filename=LOG_FILE, format="%(message)s", level=logging.INFO)
        logging.info(
            "i_trial,n_agents,t,failed_count,converged,objective_val,N,dt,ids,radius"
        )

def monte_carlo_analysis():
    """Benchmark to evaluate algorithm over many random initial conditions"""

    setup_logger()

    n_trials_iter = range(30)
    n_agents_iter =[10,15]
    # n_agents_iter = [3, 5, 10]
    # n_agents_iter = [10, 15, 20] 

    dt = 0.1
    N = 10
    radius = 0.5
    
    # Change the for loops into multi-processing?

    for n_agents in n_agents_iter:
        print(f"\tn_agents: {n_agents}")
        if n_agents >=5 and n_agents <=10:
            radius = 0.2
            
        if n_agents > 10:
            radius = 0.1
            
        for i_trial in n_trials_iter:
            print(f"\t\ttrial: {i_trial}")
            
            multi_agent_run(
                i_trial, n_agents, dt, N, radius, \
                 centralized=False)
          
    
def main():
    
    monte_carlo_analysis()


if __name__ == "__main__":
    main()


 