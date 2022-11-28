import numpy as np
from casadi import *
import do_mpc
from dynamics import *
from util import *
import networkx as nx

g = 9.81


class quadProblem:
    """Centralized MPC problem that combines all states and all costs"""

    def __init__(self, dynamics):
        self.dynamics = dynamics #Mluti-dynamical model
        
        self.generate_graph(graph_type=graph_type, params=graph_params)
        
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
    
    
    def generate_graph(self, graph_type='expander', params=None):
        '''Generate connected connectivity graph according to the params.'''

        if graph_type == 'expander':
            G = nx.paley_graph(self.n_agent).to_undirected()
        elif graph_type == 'grid':
            G = nx.grid_2d_graph(*params)
        elif graph_type == 'cycle':
            G = nx.cycle_graph(self.n_agent)
        elif graph_type == 'path':
            G = nx.path_graph(self.n_agent)
        elif graph_type == 'star':
            G = nx.star_graph(self.n_agent - 1)
        elif graph_type == 'er':
            if params < 2 / (self.n_agent - 1):
                log.fatal("Need higher probability to create a connected E-R graph!")
            G = None
            while G is None or nx.is_connected(G) is False:
                G = nx.erdos_renyi_graph(self.n_agent, params)
        else:
            log.fatal('Graph type %s not supported' % graph_type)

        self.n_edges = G.number_of_edges()
        self.G = G

    def plot_graph(self):
        '''Plot the generated connectivity graph.'''

        plt.figure()
        nx.draw(self.G)
    
    def __repr__(self):
        return f"MPCProblem(\n\t{self.dynamics})"

        

        


        