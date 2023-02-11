import casadi as cs
import numpy as np
from scipy.constants import g
from util import *
import util
from centralized_mpc import solve_rhc
from distributed_mpc import solve_rhc_distributed
from distributed_mpc_soft import solve_rhc_distributed_soft
""" 
Define simulation parameters

"""

theta_max = np.pi / 6
phi_max = np.pi / 6

v_max = 3
v_min = -3

theta_min = -np.pi / 6
phi_min = -np.pi / 6

tau_max = 20
tau_min = 0

x_min = -10
x_max = 10

y_min = -10
y_max = 10

z_min = 0
# z_max = 3.0
z_max = 4.0

max_input_base = np.array([[theta_max], [phi_max], [tau_max]])
min_input_base = np.array([[theta_min], [phi_min], [tau_min]])
max_state_base = np.array([[x_max], [y_max], [z_max], [v_max],[v_max], [v_max]])
min_state_base = np.array([[x_min], [y_min], [z_min], [v_min],[v_min],[v_min]])


centralized = True
soft_constraint = False
equal_dist = True

if __name__ == "__main__" :
    unit_dist = 1
    dt = 0.1
    radius = 0.35 
    N = 15
    n_states = 6
    n_inputs = 3
    print("Choose the number of agents (3, 4, 5, 8, 10 or 15):")
    n_agents = int(input())
    max_input = np.tile(max_input_base,(n_agents,1))
    min_input = np.tile(min_input_base,(n_agents,1))
    max_state = np.tile(max_state_base,(n_agents,1))
    min_state = np.tile(min_state_base,(n_agents,1))
    
    print("Choose the number of human agents (0, 1, 2, 3, or 4): ")
    n_humans = int(input())
    
    print("Choose the distributed or centralized:")
    flag = input()
    
    if flag == 'distributed':
        centralized = False
        

    if n_agents == 3:
        
        x0,xf = util.paper_setup_3_quads()
        u_ref = np.array([0,0,g,0,0,g,0,0,g])
        Q = np.diag([5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1])
        R = np.eye(n_agents*n_inputs)*0.01
        Qf = np.eye(n_agents*n_states)*1000
        n_dims = [3]*3
    
    elif n_agents == 4:
        
        if equal_dist == True:
            unit_dist = 1.0
            radius = 0.5
            x0 = np.array([[
                            2 ,0, 1, 0, 0, 0,
                            2, 1, 1, 0, 0, 0,
                            1.5, 1.5, 1.5, 0, 0, 0,
                            2.5, 2.5, 1.5, 0, 0, 0]], 
                            dtype=float).T
            xf = np.array([[
                            2+unit_dist*2 ,0+unit_dist*2, 1.5, 0, 0, 0,
                            2+unit_dist*2, 1+unit_dist*2, 1.5, 0, 0, 0,
                            3.5, 1.5, 1.5, 0, 0, 0, 
                            2.5, -1.0, 1.5, 0, 0, 0]], 
                            dtype=float).T
        else:
            x0,xf = util.setup_4_quads()
        u_ref = np.array([0,0,g,0,0,g,0,0,g,0,0,g])
                        
        Q = np.diag([5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1])
        R = np.eye(n_agents*n_inputs)*0.01
        Qf = np.eye(n_agents*n_states)*1000
        n_dims = [3]*4

    elif n_agents == 5:
        
        x0,xf = util.paper_setup_5_quads()
        u_ref = np.array([0,0,g,0,0,g,0,0,g,0,0,g,0,0,g])
        Q = np.diag([5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,\
                     5,5,5,1,1,1,5,5,5,1,1,1])
        R = np.eye(n_agents*n_inputs)*0.01
        Qf = np.eye(n_agents*n_states)*1000
        n_dims = [3]*5
        
    elif n_agents == 10:
        x0,xf = util.paper_setup_10_quads()
        radius = 0.2
        u_ref = np.array([0,0,g,0,0,g,0,0,g,0,0,g,0,0,g,
                         0,0,g,0,0,g,0,0,g,0,0,g,0,0,g])
        Q = np.diag([5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,\
                     5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,\
                    5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1])
        R = np.eye(n_agents*n_inputs)*0.01
        Qf = np.eye(n_agents*n_states)*1000
        n_dims = [3]*10
     
        
    elif n_agents == 15:
        x0,xf = util.paper_setup_15_quads()
        radius = 0.1
        u_ref = np.array([0,0,g,0,0,g,0,0,g,0,0,g,0,0,g,
                         0,0,g,0,0,g,0,0,g,0,0,g,0,0,g,
                         0,0,g,0,0,g,0,0,g,0,0,g,0,0,g])
        Q = np.diag([5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,\
                     5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,\
                    5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,
                    5,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,
                    5,5,5,1,1,1,5,5,5,1,1,1])
        R = np.eye(n_agents*n_inputs)*0.01
        Qf = np.eye(n_agents*n_states)*1000
        n_dims = [3]*15
     
        
    if centralized:
        file_name = f'{n_agents}_drones_centralized_sim_data'
        X_full,U_full, t, J_list, failed_count, converged = solve_rhc(
                                        dt,x0,xf,u_ref,N,Q,R,Qf,n_agents,n_states,n_inputs,radius,
                                        max_input,min_input,max_state,min_state,n_humans)
  
    if not centralized:
        if soft_constraint == True:
            file_name = f'{n_agents}_drones_distributed_sim_data(soft_constraint)'
            ids =  [100 + i for i in range(n_agents)]
            X_full, U_full, t, J_f = solve_rhc_distributed_soft(
                                            x0, xf, u_ref, N, n_agents, n_states, n_inputs, radius, ids,
                                            x_min,x_max,y_min,y_max,z_min,z_max,v_min,v_max,theta_max,
                                              theta_min,tau_max,tau_min,phi_max,phi_min
                                                )
        else:
            file_name = f'{n_agents}_drones_distributed_sim_data'
            ids =  [100 + i for i in range(n_agents)]
            X_full, U_full, t, J_list, failed_count, converged = solve_rhc_distributed(
                                            dt,x0, xf, u_ref, N, n_agents, n_states, n_inputs, radius, ids,
                                            x_min,x_max,y_min,y_max,z_min,z_max,v_min,v_max,theta_max,
                                              theta_min,tau_max,tau_min,phi_max,phi_min,n_humans,unit_dist,n_dims,
                                              equal_dist = True
                                                )
            
    np.save(file_name, X_full,U_full,t)