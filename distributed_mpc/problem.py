import numpy as np
from casadi import *
import do_mpc
from dynamics import *
from util import *

g = 9.81


class quadProblem:

    def __init__(self, dynamics):
        self.dynamics = dynamics #a list of dynamical models
        

    def split(self, graph):

        split_dynamics = self.dynamics.split(graph)
        split_costs = self.game_cost.split(graph)

        return [
            quadProblem(dynamics)
            for dynamics in zip(split_dynamics)
        ]

    @property
    def ids(self):
        if not isinstance(self.dynamics, MultiDynamicalModel):
            raise NotImplementedError(
                "Only MultiDynamicalModel's have an 'ids' attribute"
            )
     
        return self.dynamics.ids.copy()
    
    def __repr__(self):
        return f"ilqrProblem(\n\t{self.dynamics},\n\t{self.game_cost}\n)"

        

        


        