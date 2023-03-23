import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
from functools import partial
import dccp
import matplotlib.pyplot as plt
import dpilqr as dec

class MultiQuadSCP:
    def __init__(self, num_quads, num_waypoints, dt, kF, kM, g, x_init, x_end):
        self.num_quads = num_quads
        self.num_waypoints = num_waypoints
        self.dt = dt
        self.kF = kF
        self.kM = kM
        self.g = g
        self.x_init = x_init
        self.x_desired = x_end
        self.u_prev = np.zeros((num_waypoints, self.num_quads*4))
        # self.u_desired = np.zeros((num_quads, num_waypoints, 4))
        self.u_desired = np.zeros((num_waypoints,self.num_quads*4))
        self.v_max = 2*np.ones((num_quads, 3))
        self.w_max = 2*np.ones((num_quads, 3))
        self.quad_radius = 0.15
        self.radius = 0.2
        self.omega_max = 1.0
        self.control_penalty = 0.01

    def quad_dynamics(self, x, u):
        """This function maps motor commands to actual torques and forces"""
        
        f = np.array([0, 0, self.kF*sum(u[:3])])
        tau = np.array([self.kM*(u[1]-u[3]), self.kM*(u[2]-u[0]), self.kM*(u[1]+u[3]-u[0]-u[2])])

        return f, tau
    
    def quaternion_product(self, q, r):
        p = np.zeros(4)
        p[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
        p[1] = r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]
        p[2] = r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]
        p[3] = r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0]
        return p
    
    # def euler_to_quaternion(self, euler):
    #     cy = np.cos(euler[2] * 0.5)
    #     sy = np.sin(euler[2] * 0.5)
    #     cp = np.cos(euler[1] * 0.5)
    #     sp = np.sin(euler[1] * 0.5)
    #     cr = np.cos(euler[0] * 0.5)
    #     sr = np.sin(euler[0] * 0.5)
    #     qw = cy * cp * cr + sy * sp * sr
    #     qx = cy * cp * sr - sy * sp * cr
    #     qy = sy * cp * sr + cy * sp * cr
    #     qz = sy * cp * cr - cy * sp * sr
    #     return np.array([qw, qx, qy, qz])
    
    # def rotation_matrix(self, q):
    #     """This function returns the rotation matrix from a quaternion"""
    #     q0, q1, q2, q3 = q
    #     R = np.array([[1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q0*q2+q1*q3)],
    #                 [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
    #                 [2*(q1*q3-q0*q2), 2*(q0*q1+q2*q3), 1-2*(q1**2+q2**2)]])
    #     return R
    
    def solve(self):
        
        x = cp.Variable((self.num_waypoints,self.num_quads*13))
        u = cp.Variable((self.num_waypoints,self.num_quads*4))   #motor commands PWM

        # Set initial and final conditions
        for i in range(self.num_quads):
            constraints = [
                x[0, :] == self.x_init.flatten(),
                x[-1, :] == self.x_desired.flatten()
            ]

        # Set dynamics constraints and collision avoidance constraints
        for i in range(self.num_waypoints-1):
            for j in range(self.num_quads):
                x_prev = x[i, j*13:(j+1)*13]
                u_prev = u[i, j*4:(j+1)*4]
                x_next = x[i+1, j*13:(j+1)*13]
                u_next = u[i+1, j*4:(j+1)*4]

                #compute forces and torques from motor commands
                f, tau = self.quad_dynamics(x_prev, u_prev) #forces & torques at current time step
                
                #Discretized dynamics constraints:
           
                constraints += [x_next[:3] == x_prev[:3] + x_prev[7:10]*self.dt,              #position dynamics
                                
                                # x_next[3:7] == self.quaternion_product(x_prev[3:7], \
                                #             self.euler_to_quaternion(x_prev[10:13]*self.dt)), #orientation dynamics 
                                
                                x_next[7:10] == x_prev[7:10] + f/self.kF*self.dt@np.array([0, 0, 1]) \
                                - np.array([0, 0, self.g])*self.dt]    #velocity dynamics; thrust is projected along +z
                                
                                # x_next[10:13] == x_prev[10:13] + tau @ (self.kM/self.kF) * self.dt] #angular velocity dynamics

                # # Collision avoidance constraints
                # for k in range(self.num_quads):
                #     if k!=j:
                #         constraints += [cp.norm(x_next[:3] - x[i+1, k*13:(k+1)*13][:3]) >= 2*self.quad_radius]

        # Set input constraints
        for i in range(self.num_quads):
            for j in range(self.num_waypoints):
                constraints += [u[j, i*4:(i+1)*4] >= np.zeros(4),
                                u[i, i*4:(i+1)*4] <= np.array([self.kF, self.kF, self.kM, self.kF])]
        
        # Set velocity and angular velocity limits
        for i in range(self.num_quads):
            for j in range(self.num_waypoints - 1):
                constraints += [cp.norm((x[j, i*13:(i+1)*13][:3] - x[j, i*13:(i+1)*13][:3])/self.dt) <= self.v_max[i]]
                                # cp.norm((x[j, i*13:(i+1)*13][3:6] - x[j, i*13:(i+1)*13][3:6])/self.dt) <= self.omega_max[i]]
   
        # Set cost function
        cost = 0
        for i in range(self.num_quads):
            for j in range(self.num_waypoints):
                cost += cp.norm(x[j, i*13:(i+1)*13][0:3] - self.x_desired[i, :][0:3])**2 # Distance from desired position
                cost += cp.norm(x[j, i*13:(i+1)*13][7:10])**2 # Velocity magnitude
                # cost += cp.norm(x[j, i*13:(i+1)*13][10:13])**2 # Angular velocity magnitude
                cost += self.control_penalty*cp.sum_squares(u[j, i*4:(i+1)*4]) # Control penalty

        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(verbose=True)
        # prob.solve(method='dccp')

        if prob.status != 'optimal':
            raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)
        
        else:
            
            # Extract optimal trajectories and motor commands
            x_opt = x.value
            u_opt = u.value

        return x_opt, u_opt, prob.objective.value
        
def main():
    num_quads = 2
    num_waypoints = 25

    x_end = [[2.5, 1.8, 2.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],[2.5, 1.8, 2.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    x_end = np.array(x_end)

    custom_values = [[0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1.0, 1.5,  1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    custom_x_init = np.array(custom_values)
   

    quad_scp = MultiQuadSCP(num_quads=num_quads, num_waypoints=num_waypoints, dt = 0.1,\
                             kF = 6.11e-8, kM = 1.5e-9, g = 9.81, x_init=custom_x_init, x_end=x_end)
    x_opt, u_opt, objective = quad_scp.solve()

    
if __name__ == "__main__":
    main()