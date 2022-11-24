import numpy as np
from casadi import *
import do_mpc
from dynamics import *
from util import *

g = 9.81

class quadProblem(quadDynamics6D):

    def __init__(self, x0, xf, Q, R, Qf, u_ref, N, *args, **kwargs):
    
        super().__init__(xf, Q, R, Qf, u_ref, N, *args, **kwargs)
        
        
        #Setting objective function:
        objective = quadDynamics6D.cost(self)[0] + quadDynamics6D.cost(self)[1]

        #minimize objective:
        self.opti.minimize(objective)

        # #dynamic constraints:
        # g = 9.81
        # f = lambda x,u: vertcat(x[3],x[4],x[5],g*tan([0]),-g*tan(u[1]),u[2]-g) #dx/dt = f(x,u)
        
        f = self._f

        dt = self.T/N #length of a control interval
        
        for k in range(N): #loop over control intervals
            # Runge-Kutta 4 integration
            k1 = f(self.X[:,k],         self.U[:,k])
            k2 = f(self.X[:,k]+dt/2*k1, self.U[:,k])
            k3 = f(self.X[:,k]+dt/2*k2, self.U[:,k])
            k4 = f(self.X[:,k]+dt*k3,   self.U[:,k])
            x_next = self.X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 

            self.opti.subject_to(self.X[:,k+1]==x_next) # close the gaps

        #inequality constraints:

        self.opti.subject_to(self.X[2,:]<=3.5) # altitude p_z is limited
        self.opti.subject_to(0<=self.X[2,:])

        self.opti.subject_to(self.X[0,:]<=4) # p_x is limited
        self.opti.subject_to(-4<=self.X[0,:])

        self.opti.subject_to(self.X[1,:]<=4) # p_y is limited
        self.opti.subject_to(-4<=self.X[1,:])

        self.opti.subject_to(self.U[0,:]<=np.pi/6) # theta is limited
        self.opti.subject_to(-np.pi/6<=self.U[0,:])

        self.opti.subject_to(self.U[1,:]<=np.pi/6) # phi is limited
        self.opti.subject_to(-np.pi/6<=self.U[1,:])

        self.opti.subject_to(self.U[2,:]<=20) # tau is limited
        self.opti.subject_to(0<=self.U[2,:])

        self.opti.subject_to(self.T>=0) #time must be positive

        #equality constraints:
        self.opti.subject_to(self.X[:,0] == x0)
        
        #initial values for solver
        self.opti.set_initial(self.T, 0)

    @property
    def ids(self):
        if not isinstance(self.dynamics, MultiDynamicalModel):
            raise NotImplementedError(
                "Only MultiDynamicalModel's have an 'ids' attribute"
            )
        if not self.dynamics.ids == self.game_cost.ids:
            raise ValueError(f"Dynamics and cost have inconsistent ID's: {self}")
        return self.dynamics.ids.copy()
    
    def solve(self):
        
        #self.mySolver = "ipopt"
        #self.mySolver = "worhp"
        #self.mySolver = "sqpmethod"

        #solve NLP:
        self.opti.solver("ipopt")
        self.sol = self.opti.solve()

        return self.sol.value(self.X),self.sol.value(self.U)


        

        


        