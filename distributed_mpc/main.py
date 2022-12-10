import casadi as cs
import numpy as np
from scipy.constants import g
from util import *
from centralized_mpc import solve_rhc
from distributed_mpc import solve_rhc_distributed

""" 
Define simulation parameters

"""

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

radius = 0.5
N = 15
n_states = 6
n_inputs = 3

centralized = True


if __name__ == "__main__" :
    
    print("Choose the number of agents (3,5,or 10):")
    n_agents = int(input())
    
    print("Choose the distributed or centralized:")
    flag = input()
    if flag == 'distributed':
        centralized = False

    if n_agents == 3:
        x0,xf = paper_setup_3_quads()
        u_ref = np.array([0,0,g,0,0,g,0,0,g])
        Q = np.diag([5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1])
        R = np.eye(n_agents*n_inputs)*0.01
        Qf = np.eye(n_agents*n_states)*1000
        max_input = np.tile(max_input_base,n_agents)
        min_input = np.tile(min_input_base,n_agents)
        max_state = np.tile(max_state_base,n_agents)
        min_state = np.tile(min_state_base,n_agents)
        
        
    elif n_agents == 5:
        x0,xf = paper_setup_5_quads()
        u_ref = np.array([0,0,g,0,0,g,0,0,g,0,0,g,0,0,g])
        Q = np.diag([5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,\
                     5,5,5,1,1,1,5,5,5,1,1,1])
        R = np.eye(n_agents*n_inputs)*0.01
        Qf = np.eye(n_agents*n_states)*1000
        max_input = np.tile(max_input_base,n_agents)
        min_input = np.tile(min_input_base,n_agents)
        max_state = np.tile(max_state_base,n_agents)
        min_state = np.tile(min_state_base,n_agents)

    elif n_agents == 10:
        x0,xf = paper_setup_10_quads()
        radius = 0.2
        u_ref = np.array([0,0,g,0,0,g,0,0,g,0,0,g,0,0,g,
                         0,0,g,0,0,g,0,0,g,0,0,g,0,0,g])
        Q = np.diag([5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,\
                     5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,\
                    5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1])
        R = np.eye(n_agents*n_inputs)*0.01
        Qf = np.eye(n_agents*n_states)*1000
        max_input = np.tile(max_input_base,n_agents)
        min_input = np.tile(min_input_base,n_agents)
        max_state = np.tile(max_state_base,n_agents)
        min_state = np.tile(min_state_base,n_agents)
        
    if centralized:
        file_name = 'centralized_sim_data'
        X_full, U_full, t, J_f = solve_rhc(x0,xf,u_ref,N,Q,R,Qf,n_agents,n_states,n_inputs,radius,
                                     max_input,min_input,max_state,min_state)
  
    if not centralized:
        file_name = 'distributed_sim_data'
        ids =  [100 + i for i in range(n_agents)]
        X_full, U_full, t, J_f = solve_rhc_distributed(
                                        x0, xf, u_ref, N, n_agents, n_states, n_inputs, radius, ids,
                                        x_min,x_max,y_min,y_max,z_min,z_max,v_min,v_max,theta_max,
                                          theta_min,tau_max,tau_min,phi_max,phi_min
                                            )
        
    np.save(file_name, X_full,U_full,t)