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


class QuadrotorDynamics: #this is a "simulator"
    def __init__(self, dt):
        # Define constants
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.m = 0.5  # Mass of quadrotor (kg)
        self.l = 0.2  # Length of arm (m)
        self.Jx = 0.004  # Moment of inertia around x-axis (kg*m^2)
        self.Jy = 0.004  # Moment of inertia around y-axis (kg*m^2)
        self.Jz = 0.008  # Moment of inertia around z-axis (kg*m^2)
        self.kF = 6.11e-8  # Force constant (N/(rad/s)^2)
        self.kM = 1.5e-9  # Moment constant (N*m/(rad/s)^2)

        # Initialize state variables
        self.x = np.zeros((13, 1))  # State vector [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, t]
        self.dt = dt  # Timestep

    def update(self, u):
        # Unpack state vector and input vector
        x = self.x
        dt = self.dt
        m = self.m
        l = self.l
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz
        kF = self.kF
        kM = self.kM
        g = self.g
        u1, u2, u3, u4 = u

        # Compute rotation matrix and its derivative
        phi = x[6, 0]
        theta = x[7, 0]
        psi = x[8, 0]
        R = np.array([[np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi),
                       np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)],
                      [np.sin(psi)*np.cos(theta), np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi),
                       np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)],
                      [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)]])
        Rdot = np.array([[np.sin(psi)*np.sin(theta)*np.sin(phi) - np.cos(psi)*np.cos(phi), -np.sin(psi)*np.cos(theta)*np.sin(phi) - np.cos(psi)*np.sin(phi), -np.cos(psi)*np.sin(theta)],
                         [np.cos(psi)*np.cos(phi) - np.sin(psi)*np.sin(theta)*np.sin(phi), -np.cos(psi)*np.cos(theta)*np.sin(phi) - np.sin(psi)*np.sin(phi), np.sin(psi)*np.sin(theta)],
                         [0, -np.cos(theta)*np.sin(phi), -np.cos(theta)*np.cos(phi)]]) @ np.array([[x[9, 0]], [x[10, 0]], [x[11, 0]]])

        # Compute forces and moments
        F1 = kF * u1
        F2 = kF * u2
        F3 = kF * u3
        F4 = kF * u4

        F = np.array([0, 0, F1+F2+F3+F4])
        M = np.array([(l*kF)*(u1-u3), (l*kF)*(u2-u4), kM*(u1-u2+u3-u4)])

        # Compute acceleration and angular acceleration
        a = (1/m)*R.dot(F) - np.array([0, 0, g])
        omega = np.array([[x[9, 0]], [x[10, 0]], [x[11, 0]]])
        alpha = np.linalg.inv(np.diag([Jx, Jy, Jz])).dot(M - np.cross(omega, np.diag([Jx, Jy, Jz]).dot(omega), axis=0))

        # Compute new state vector
        xdot = np.zeros((13, 1))
        xdot[0:3] = x[3:6]
        xdot[3:6] = a
        xdot[6:9] = Rdot
        xdot[9:12] = alpha
        xdot[12] = 1

        xnew = x + xdot*dt
        self.x = xnew

        # Output position and orientation
        """
        Rz(psi) = [[cos(psi), -sin(psi), 0],
            [sin(psi),  cos(psi), 0],
            [       0,         0, 1]]

        Ry(theta) = [[ cos(theta), 0, sin(theta)],
                    [          0, 1,          0],
                    [-sin(theta), 0, cos(theta)]]

        Rx(phi) = [[1,          0,           0],
                [0, cos(phi), -sin(phi)],
                [0, sin(phi),  cos(phi)]]

        R = Rz(psi) @ Ry(theta) @ Rx(phi)

        R is computed in the body frame

        """

        pos = xnew[0:3]
        R = np.array([[np.cos(xnew[8, 0])*np.cos(xnew[7, 0]), np.cos(xnew[8, 0])*np.sin(xnew[7, 0])*np.sin(xnew[6, 0]) - np.sin(xnew[8, 0])*np.cos(xnew[6, 0]), np.cos(xnew[8, 0])*np.sin(xnew[7, 0])*np.cos(xnew[6, 0]) + np.sin(xnew[8, 0])*np.sin(xnew[6, 0])],
                      [np.sin(xnew[8, 0])*np.cos(xnew[7, 0]), np.sin(xnew[8, 0])*np.sin(xnew[7, 0])*np.sin(xnew[6, 0]) + np.cos(xnew[8, 0])*np.cos(xnew[6, 0]), np.sin(xnew[8, 0])*np.sin(xnew[7, 0])*np.cos(xnew[6, 0]) - np.cos(xnew[8, 0])*np.sin(xnew[6, 0])],
                      [-np.sin(xnew[7, 0]), np.cos(xnew[7, 0])*np.sin(xnew[6, 0]), np.cos(xnew[7, 0])*np.cos(xnew[6, 0])]])
        return pos, R



