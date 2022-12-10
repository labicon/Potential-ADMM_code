from casadi import *
import casadi as cs
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
import itertools 
from time import perf_counter
from decentralized import util
from problem import QuadProblem


class TrackingCost(QuadProblem):
    def __init__(self,QuadProblem,Q,R,Qf):
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.X = QuadProblem.X
        self.U = QuadProblem.U
        self.xf = QuadProblem.xf
        self.u_ref = QuadProblem.u_ref
        
    def objective(self):
        
        total_stage_cost = 0
        for j in range(self.X.shape[1]-1):
            for i in range(self.X.shape[0]):
                total_stage_cost += (self.X[i,j]-self.xf[i])*self.Q[i, i]*(self.X[i,j]-self.xf[i])

        for j in range(self.U.shape[1]-1):
            for i in range(self.U.shape[0]):
                total_stage_cost += (self.U[i,j]-self.u_ref[i])*self.R[i, i]*(self.U[i,j]-self.u_ref[i])

        #Quadratic terminal cost:
        total_terminal_cost = 0

        for i in range(self.X.shape[0]):
            total_terminal_cost += (self.X[i,-1]-self.xf[i])*self.Qf[i, i]*(self.X[i,-1]-self.xf[i])

        return total_stage_cost + total_terminal_cost
    

