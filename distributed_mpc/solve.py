import numpy as np
from casadi import *
import do_mpc
from dynamics import *
from util import *


class SolveMPC:
    
    def __init__(self,problem,N):
        self.opti = Opti()
        self.dynamics = problem.dynamics
        self.problem = problem
        self.N = N
        self.T = opti.variable()
        self.dt = /self.N #length of a control interval
        
        self.n_states = 6
        self.n_inputs = 3
        self.n_agents = self.dynamics.n_players
             
    g = 9.81
    
    def min_obj(objective):
        
        self.dynamics.opti.minimize(objective) 
    
    # lambda x,u: vertcat(x[3],x[4],x[5],g*tan([0]),-g*tan(u[1]),u[2]-g) #dx/dt = f(x,u)
    def f(x,u):
        #TODO: determine subproblem and re-construct dynamics
        return vertcat(x[3],x[4],x[5],g*tan([0]),-g*tan(u[1]),u[2]-g) #dx/dt = f(x,u)
        
    def solve(dynamics):
        
        for k in range(self.N): #loop over control intervals
            # Runge-Kutta 4 integration
            k1 = f(dynamics.X[:,k],         dynamics.U[:,k])
            k2 = f(dynamics.X[:,k]+self.dt/2*k1, dynamics.U[:,k])
            k3 = f(dynamics.X[:,k]+self.dt/2*k2, dynamics.U[:,k])
            k4 = f(dynamics.X[:,k]+self.dt*k3,   dynamics.U[:,k])
            x_next = dynamics.X[:,k] + self.dt/6*(k1+2*k2+2*k3+k4) 

            opti.subject_to(dynamics.X[:,k+1]==x_next) # close the gaps

            #inequality constraints(for a single quad):
            #TODO: generalize to n quadcopter dynamics

        dynamics.opti.subject_to(dynamics.X[2,:]<=3.5) # altitude p_z is limited
        dynamics.opti.subject_to(0.6<=dynamics.X[2,:])

        dynamics.opti.subject_to(dynamics.X[0,:]<=4) # p_x is limited
        dynamics.opti.subject_to(-4<=dynamics.X[0,:])

        dynamics.opti.subject_to(dynamics.X[1,:]<=4) # p_y is limited
        dynamics.opti.subject_to(-4<=dynamics.X[1,:])

        dynamics.opti.subject_to(dynamics.U[0,:]<=np.pi/6) # theta is limited
        dynamics.opti.subject_to(-np.pi/6<=dynamics.U[0,:])

        dynamics.opti.subject_to(dynamics.U[1,:]<=np.pi/6) # phi is limited
        dynamics.opti.subject_to(-np.pi/6<=dynamics.U[1,:])

        dynamics.opti.subject_to(dynamics.U[2,:]<=20) # tau is limited
        dynamics.opti.subject_to(0<=dynamics.U[2,:])

        dynamics.opti.subject_to(dynamics.T>=0) #time must be positive

        #equality constraints:
        dynamics.opti.subject_to(dynamics.X[:,0] == x0)

        #initial values for solver
        dynamics.opti.set_initial(dynamics.T, 0)

        dynamics.opti.solver("ipopt")
        dynamics.sol = dynamics.opti.solve()

        return dynamics.sol.value(dynamics.X),dynamics.sol.value(dynamics.U)
