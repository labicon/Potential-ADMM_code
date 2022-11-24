import numpy as np
from casadi import *
import do_mpc
from dynamics import *
from util import *


class SolveMPC(problem,dynamics):
    
    def __init__(self,problem,dynamics):
        
        dt = dynamics.T/N #length of a control interval
        N = dynamics.N
        
        
        #minimize objective:
        dynamics.opti.minimize(objective)
        
        #dynamic constraints for a single (6D)quadcopter:
        g = 9.81
        f = lambda x,u: vertcat(x[3],x[4],x[5],g*tan([0]),-g*tan(u[1]),u[2]-g) #dx/dt = f(x,u)
        
        for k in range(N): #loop over control intervals
            # Runge-Kutta 4 integration
            k1 = f(dynamics.X[:,k],         dynamics.U[:,k])
            k2 = f(dynamics.X[:,k]+dt/2*k1, dynamics.U[:,k])
            k3 = f(dynamics.X[:,k]+dt/2*k2, dynamics.U[:,k])
            k4 = f(dynamics.X[:,k]+dt*k3,   dynamics.U[:,k])
            x_next = dynamics.X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 

            opti.subject_to(dynamics.X[:,k+1]==x_next) # close the gaps

            #inequality constraints:

            dynamics.opti.subject_to(dynamics.X[2,:]<=3.5) # altitude p_z is limited
            dynamics.opti.subject_to(0<=dynamics.X[2,:])

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
            
    def solve(dynamics):
        
        dynamics.opti.solver("ipopt")
        dynamics.sol = dynamics.opti.solve()

        return dynamics.sol.value(dynamics.X),dynamics.sol.value(dynamics.U)
