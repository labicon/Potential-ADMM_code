import numpy as np
import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import dpilqr as dec
from dpilqr.util import split_agents_gen, uniform_block_diag
import abc

def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator

"""
Instantiate an abstract class such that each class inherited from DynamicalModel
must have the same abstractmethod :

"""
class DynamicalModel(abc.ABC):
    _id = 0
    def __init__(self, n_x, n_u, dt, id=None):
        if not id:
            id = DynamicalModel._id
            DynamicalModel.id +=1

        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
        self.id = id
        
    def __call__(self, x, u):

        return discretize(self.f,self.dt)

    @staticmethod
    def f():

        pass

    @abc.abstractmethod
    def linearize():

        pass

    @classmethod
    def _reset_ids(cls):

        cls._id =0

    def __repr__(self):
        return f"{type(self).__name__}(n_x: {self.n_x}, n_u: {self.n_u}, id: {self.id})"


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


    def f(self, s, u):
        """Derivative of the current combined states and controls"""
        xn = np.zeros_like(x)
        nx = self.x_dims[0]
        nu = self.u_dims[0]
        for i, model in enumerate(self.submodels):
            xn[i * nx : (i + 1) * nx] = model.f(
                setattr[i * nx : (i + 1) * nx], u[i * nu : (i + 1) * nu]
            )
        return xn

    def __call__(self, s, u):
        """Zero-order hold to integrate continuous dynamics f"""

        # return forward_euler_integration(self.f, x, u, self.dt)
        # return rk4_integration(self.f, x, u, self.dt, self.dt)
        xn = np.zeros_like(s)
        nx = self.x_dims[0]
        nu = self.u_dims[0]
        for i, model in enumerate(self.submodels):
            xn[i * nx : (i + 1) * nx] = model.__call__(
                s[i * nx : (i + 1) * nx], u[i * nu : (i + 1) * nu]
            )
        return xn

    
    def linearize(self, x, u):
        sub_linearizations = [
            submodel.linearize(xi.flatten(), ui.flatten())
            for submodel, xi, ui in zip(
                self.submodels,
                split_agents_gen(x, self.x_dims),
                split_agents_gen(u, self.u_dims),
            )
        ]
 
        sub_As = [AB[0] for AB in sub_linearizations]
        sub_Bs = [AB[1] for AB in sub_linearizations]

        return uniform_block_diag(*sub_As), uniform_block_diag(*sub_Bs)



def QuadDynamics6D(s,u):
    g = 9.81
    x, y, z, vx, vy, vz = s
    theta, phi, tau = u
    ds = jnp.array([

        vx,
        vy,
        vz,
        g*jnp.tan(theta),
        -g*jnp.tan(phi),
        tau-g

    ])

    return ds

# def cartpole(s, u):
#     """Compute the cart-pole state derivative."""
#     mp = 1.     # pendulum mass
#     mc = 4.     # cart mass
#     ℓ = 1.      # pendulum length
#     g = 9.81    # gravitational acceleration

#     x, θ, dx, dθ = s
#     sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
#     h = mc + mp*(sinθ**2)
#     ds = jnp.array([
#         dx,
#         dθ,
#         (mp*sinθ*(ℓ*(dθ**2) + g*cosθ) + u[0]) / h,
#         -((mc + mp)*g*sinθ + mp*ℓ*(dθ**2)*sinθ*cosθ + u[0]*cosθ) / (h*ℓ)
#     ])
#     return ds

