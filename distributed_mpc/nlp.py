import numpy as np
from casadi import *
import do_mpc
from model import *
from util import all

g = 9.81

class quadProblem(quadDynamics6D):

    def __init__(self, Q, R, Qf, u_ref, N, *args, **kwargs):
        x0, xf = paper_setup_3_quads()

        super.__init__(xf, Q, R, Qf, u_ref, N, *args, **kwargs)
        
        #control intervals
        opti = quadDynamics6D.opti
        X = quadDynamics6D.X
        U = quadDynamics6D.U
        T = quadDynamics6D.T
        
     
        #Setting objective function:
        objective = quadDynamics6D.cost(xf, Q, R, Qf, u_ref)[0] + quadDynamics6D.cost(xf, Q, R, Qf, u_ref)[1]

        #minimize objective:
        opti.minimize(objective)

        #dynamic constraints:
        g = 9.81
        f = lambda x,u: vertcat(x[3],x[4],x[5],g*tan([0]),-g*tan(u[1]),u[2]-g) #dx/dt = f(x,u)

        dt = T/N #length of a control interval
        
        for k in range(N): #loop over control intervals
            # Runge-Kutta 4 integration
            k1 = f(X[:,k],         U[:,k])
            k2 = f(X[:,k]+dt/2*k1, U[:,k])
            k3 = f(X[:,k]+dt/2*k2, U[:,k])
            k4 = f(X[:,k]+dt*k3,   U[:,k])
            x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 

            opti.subject_to(X[:,k+1]==x_next) # close the gaps

        #inequality constraints:

        opti.subject_to(X[2,:]<=3.5) # altitude p_z is limited
        opti.subject_to(0<=X[2,:])

        opti.subject_to(X[0,:]<=4) # p_x is limited
        opti.subject_to(-4<=X[0,:])

        opti.subject_to(X[1,:]<=4) # p_y is limited
        opti.subject_to(-4<=X[1,:])

        opti.subject_to(U[0,:]<=np.pi/6) # theta is limited
        opti.subject_to(-np.pi/6<=U[0,:])

        opti.subject_to(U[1,:]<=np.pi/6) # phi is limited
        opti.subject_to(-np.pi/6<=U[1,:])

        opti.subject_to(U[2,:]<=20) # tau is limited
        opti.subject_to(0<=U[2,:])

        opti.subject_to(T>=0) #time must be positive

        #equality constraints:
        opti.subject_to(X[0] == x0.T)
        
        #initial values for solver
        opti.set_initial(T, 0)

        

        
    def solve(self):
        
        #self.mySolver = "ipopt"
        #self.mySolver = "worhp"
        #self.mySolver = "sqpmethod"

        #solve NLP:
        self.opti.solver("ipopt")
        self.sol = self.opti.solve()

        return self.sol


        

        


        