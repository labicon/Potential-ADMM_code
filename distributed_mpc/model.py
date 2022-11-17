import numpy as np
from casadi import *
import do_mpc
from util import *
import abc


class DynamicalModel(abc.ABC):

    _id = 0

    def __init__(self, n_x, n_u, id=None):
        if not id:
            id = DynamicalModel._id
            DynamicalModel._id += 1

        self.n_x = n_x
        self.n_u = n_u
        self.id = id

    @staticmethod
    def f():
        """Continuous derivative of dynamics with respect to time"""
        pass



class MultiDynamicalModel(DynamicalModel):
    """Encompasses the dynamical simulation and linearization for a collection of
    DynamicalModel's
    """

    def __init__(self, submodels):
        self.submodels = submodels
        self.n_players = len(submodels)

        self.x_dims = [submodel.n_x for submodel in submodels]
        self.u_dims = [submodel.n_u for submodel in submodels]
        self.ids = [submodel.id for submodel in submodels]

        super().__init__(sum(self.x_dims), sum(self.u_dims), submodels[0].dt, -1)

    def split(self, graph):
        """Split this model into submodels dictated by the interaction graph"""
        split_dynamics = []
        for problem in graph:
            split_dynamics.append(
                MultiDynamicalModel(
                    [model for model in self.submodels if model.id in graph[problem]]
                )
            )

        return split_dynamics

    def __repr__(self):
        sub_reprs = ",\n\t".join([repr(submodel) for submodel in self.submodels])
        return f"MultiDynamicalModel(\n\t{sub_reprs}\n)"





class quadDynamics6D(DynamicalModel):
    """
    6D quadrotor dynamics:

    """
    def __init__(self, xf, Q, R, Qf, u_ref, N, *args, **kwargs):
        super().__init__(6,3,*args,**kwargs)

        self.opti = Opti()

        #decision variables:
        self.X = self.opti.variable(6, N+1)
        self.U = self.opti.variable(3, N)
        self.T = self.opti.variable() #final time

        self.xf = xf
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.u_ref = u_ref
    
    def cost(self):
        #u_ref is shape (3,1)
        
        #Quadratic running cost:
        self.total_stage_cost = 0
        for i in range(self._x.shape[0]):
            self.total_stage_cost += (self._x[i]-self.xf[i])*self.Q[i, i]*(self._x[i]-self.xf[i])

        for j in range(self._u.shape[0]):
            self.total_stage_cost += (self._u[j]-self.u_ref[j])*self.R[j, j]*(self._u[j]-self.u_ref[j])
        
        #Quadratic terminal cost:
        self.total_terminal_cost = 0
        for m in range(self._x.shape[0]):
            self.total_terminal_cost += (self._x[m]-self.xf[m])*self.Qf[m, m]*(self._x[m]-self.xf[m])
        
        return self.total_stage_cost, self.total_terminal_cost
