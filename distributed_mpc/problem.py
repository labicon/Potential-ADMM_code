import numpy as np
from casadi import *
import do_mpc
from dynamics import *
from util import *

g = 9.81


class quadProblem:
    """Centralized MPC problem that combines all states and all costs"""

    def __init__(self, dynamics):
        self.dynamics = dynamics #Mluti-dynamical model
        
        
    @property
    def ids(self):
        if not isinstance(self.dynamics, MultiDynamicalModel):
            raise NotImplementedError(
                "Only MultiDynamicalModel's have an 'ids' attribute"
            )
     
        return self.dynamics.ids.copy()
    
    def split(self, graph):

        split_dynamics = self.dynamics.split(graph)
        split_costs = self.game_cost.split(graph)

        return [
            quadProblem(dynamics)
            for dynamics in zip(split_dynamics)
        ]
    
    
    def extract(self, X, U, id_):
        """Extract the state and controls for a particular agent id_ from the
        concatenated problem state/controls
        """

        if id_ not in self.ids:
            raise IndexError(f"Index {id_} not in ids: {self.ids}.")

        # NOTE: Assume uniform dynamical models.
        ext_ind = self.ids.index(id_)

        x_dim = self.dynamics.x_dims[0]
        u_dim = self.dynamics.u_dims[0]
        Xi = X[:, ext_ind * x_dim : (ext_ind + 1) * x_dim]
        Ui = U[:, ext_ind * u_dim : (ext_ind + 1) * u_dim]

        return Xi, Ui
    
    def __repr__(self):
        return f"MPCProblem(\n\t{self.dynamics})"

        

        


        