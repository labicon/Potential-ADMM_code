from casadi import *
import casadi as cs
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
import itertools 
from time import perf_counter
from decentralized import util


class QuadProblem:
    def __init__(self,n_agents,n_states,n_inputs,x0,xf,u_ref,N,dt):
        self.opti = Opti()
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.x0 = x0
        self.xf = xf
        self.u_ref = u_ref
        self.N = N
        self.dt = dt
        self.X = self.opti.variable(self.n_states,N+1)
        self.U = self.opti.variable(self.n_inputs,N)
        
    
    def __repr__(self):
        rep = 'QuadProblem(' + str(self.opti) + ')'
        print(f'The current QuadProblem has {self.n_agents} agents, dt is {self.dt}')
        return rep
    
    