from casadi import *
import casadi as cs
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
import itertools 
from time import perf_counter
from decentralized import util
from problem import QuadProblem
from cost import TrackingCost

g = 9.8



class SolveSingleMPC(TrackingCost,QuadProblem):
    
    def __init__(self,TrackingCost,QuadProblem):
        self.X = TrackingCost.X
        self.U = TrackingCost.U
        self.xf = TrackingCost.xf
        self.u_ref = TrackingCost.u_ref
        self.Q = TrackingCost.Q
        self.R = TrackingCost.R
        self.Qf = TrackingCost.Qf
        self.opti = QuadProblem.opti
        self.objective = TrackingCost.objective()
        self.N = QuadProblem.N
        self.dt = QuadProblem.dt
        self.x0 = QuadProblem.x0
        
    def generate_f(self):
        f = lambda x,u: vertcat(x[3],x[4],x[5],g*tan(u[0]),-g*tan(u[1]),u[2]-g)
        return f
    
    def solve(self):
        for k in range(self.N): #loop over control intervals
        # Runge-Kutta 4 integration
            k1 = f(self.X[:,k],         self.U[:,k])
            k2 = f(self.X[:,k]+self.dt/2*k1, self.U[:,k])
            k3 = f(self.X[:,k]+self.dt/2*k2, self.U[:,k])
            k4 = f(self.X[:,k]+self.dt*k3,   self.U[:,k])
            x_next = self.X[:,k] + self.dt/6*(k1+2*k2+2*k3+k4) 

            self.opti.subject_to(self.X[:,k+1]==x_next) # close the gaps
            
        self.opti.minimize(self.objective) 
        self.opti.subject_to(self.X[2,:]<=3.0) # altitude p_z is limited
        self.opti.subject_to(0.4<=self.X[2,:])

        self.opti.subject_to(self.X[0,:]<=3) # p_x is limited
        self.opti.subject_to(-3<=self.X[0,:])

        self.opti.subject_to(self.X[1,:]<=3) # p_y is limited
        self.opti.subject_to(-3<=self.X[1,:])

        self.opti.subject_to(self.U[0,:]<=np.pi/6) # theta is limited
        self.opti.subject_to(-np.pi/6<=self.U[0,:])

        self.opti.subject_to(self.U[1,:]<=np.pi/6) # phi is limited
        self.opti.subject_to(-np.pi/6<=self.U[1,:])

        self.opti.subject_to(self.U[2,:]<=25) # tau is limited
        self.opti.subject_to(0<=self.U[2,:]) #minimum force 
        
        self.opti.subject_to(self.X[:,0] == self.x0)
        
        t0 = perf_counter()
  
        opti.solver('ipopt')
        sol = opti.solve()
        tf = perf_counter()

        print(f'total solve time is {tf-t0} seconds')
        
        X_trj = sol.value(X)
        U_trj = sol.value(U)
        
        return X_trj,U_trj
        
        
        