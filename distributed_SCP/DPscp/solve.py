import numpy as np
import cvxpy as cp

# State vector [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, t]


class MultiQuadSCP:
    def __init__(self, num_quads, num_waypoints, dt, kF, kM, g, x_init, x_end):
        self.num_quads = num_quads
        self.num_waypoints = num_waypoints
        self.dt = dt
        self.kF = kF
        self.kM = kM
        self.g = g
        self.x_init = x_init
        self.x_desired = np.tile(np.array(x_end), (num_waypoints,self.num_quads*13))
        self.u_prev = np.zeros((num_waypoints, self.num_quads*4))
        # self.u_desired = np.zeros((num_quads, num_waypoints, 4))
        # self.u_desired = np.zeros((num_waypoints,self.num_quads*4))
        self.v_max = 2*np.ones((num_quads, 3))
        self.w_max = 2*np.ones((num_quads, 3))

    def quad_dynamics(self, x, u):
        f = np.array([0, 0, self.kF*np.sum(u[:3])])
        tau = np.array([self.kM*(u[1]-u[3]), self.kM*(u[2]-u[0]), self.kM*(u[1]+u[3]-u[0]-u[2])])
        return f, tau

    def quaternion_product(self, q, r):
        p = np.zeros(4)
        p[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
        p[1] = r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]
        p[2] = r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]
        p[3] = r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0]
        return p
    
    def euler_to_quaternion(self, euler):
        cy = np.cos(euler[2] * 0.5)
        sy = np.sin(euler[2] * 0.5)
        cp = np.cos(euler[1] * 0.5)
        sp = np.sin(euler[1] * 0.5)
        cr = np.cos(euler[0] * 0.5)
        sr = np.sin(euler[0] * 0.5)
        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr
        return np.array([qw, qx, qy, qz])

    def solve(self):
        
        x = cp.Variable((self.num_waypoints,self.num_quads*13))
        u = cp.Variable((self.num_waypoints,self.num_quads*4))   #motor commands PWM

        # Set initial and final conditions
        for i in range(self.num_quads):
            constraints = [
                x[0, :] == self.x_init,
                x[-1, :] == self.x_desired
            ]

        # Set dynamics constraints and collision avoidance constraints
        for j in range(self.num_quads):
            for i in range(self.num_waypoints-1):
                x_prev = x[i, j*13:(j+1)*13]
                u_prev = u[i, j*13:(j+1)*13]
                x_next = x[i+1, j*13:(j+1)*13]
                u_next = u[i+1, j*13:(j+1)*13]

                #Dynamics constraints
                f, tau = self.quad_dynamics(x_prev, u_prev) #forces & torques at current time step
                
                constraints += [x_next[:3] == x_prev[:3] + x_prev[7:10]*self.dt,              #position dynamics
                                
                                x_next[3:7] == self.quaternion_product(x_prev[3:7], \
                                            self.euler_to_quaternion(x_prev[10:13]*self.dt)), #orientation dynamics 
                                
                                x_next[7:10] == x_prev[7:10] + \
                                f/self.kF*self.dt*np.array([0, 0, 1]) \
                                - np.array([0, 0, self.g])*self.dt,    #velocity dynamics; thrust is projected along +z
                                
                                x_next[10:13] == x_prev[10:13] + self.kM/self.kF*self.dt*tau] #angular velocity dynamics

                # Collision avoidance constraints
                for k in range(self.num_quads):
                    if k == j:
                        continue
                    constraints += [cp.norm(x_next[:3] - x[i+1, k*3:(k+1)*3]) >= 2*self.quad_radius]

        # Set input constraints
        for i in range(self.num_quads):
            for j in range(self.num_waypoints):
                constraints += [u[i, j, :] >= np.zeros(4),
                                u[i, j, :] <= np.array([self.kF, self.kF, self.kM, self.kF])]
        
        # Set velocity and angular velocity limits
        for i in range(self.num_quads):
            for j in range(self.num_waypoints - 1):
                constraints += [cp.norm((x[i, j+1, :3] - x[i, j, :3])/self.dt) <= self.v_max[i],
                                cp.norm((x[i, j+1, 3:6] - x[i, j, 3:6])/self.dt) <= self.omega_max[i]]

        # Add collision avoidance constraints
        for i in range(self.num_quads):
            for j in range(self.num_waypoints - 1):
                for k in range(self.num_quads):
                    if k != i:
                        constraints += [cp.norm(x[i, j, :3] - x[k, j, :3]) >= 2*self.radius,
                                        cp.norm(x[i, j+1, :3] - x[k, j+1, :3]) >= 2*self.radius,
                                        cp.norm(x[i, j, :2] - x[k, j, :2]) >= self.radius,
                                        cp.norm(x[i, j+1, :2] - x[k, j+1, :2]) >= self.radius]
                                        
        # Add initial and final position constraints
        for i in range(self.num_quads):
            constraints += [x[i, 0, :3] == self.waypoints[i][0],
                            x[i, self.num_waypoints-1, :3] == self.waypoints[i][-1]]
            
        # Add initial and final velocity constraints
        for i in range(self.num_quads):
            constraints += [cp.norm((x[i, 1, :3] - x[i, 0, :3])/self.dt) <= self.v_max[i],
                            cp.norm((x[i, self.num_waypoints-1, :3] - x[i, self.num_waypoints-2, :3])/self.dt) <= self.v_max[i]]
            
        # Add initial and final angular velocity constraints
        for i in range(self.num_quads):
            constraints += [cp.norm((x[i, 1, 3:6] - x[i, 0, 3:6])/self.dt) <= self.omega_max[i],
                            cp.norm((x[i, self.num_waypoints-1, 3:6] - x[i, self.num_waypoints-2, 3:6])/self.dt) <= self.omega_max[i]]
            
        # Additional Dynamics constraints 
        omega_prev = x_prev[10:13]
        omega_next = x_next[10:13]
        R_prev = self.rotation_matrix(x_prev[3:7])
        R_next = self.rotation_matrix(x_next[3:7])
        F_prev = self.kF*np.sum(u_prev)
        F_next = self.kF*np.sum(u_next)

        constraints += [cp.sum_squares(x_next[3:6] - cp.cross(omega_next, R_next.T @ self.gravity)) <= self.epsilon, #constraint on the angular velocity
                        cp.sum_squares(F_next - R_next @ np.array([0, 0, self.g])) <= self.epsilon,  #constraint on the thrust force of the quadrotor
                        cp.sum_squares(x_next[7:10]) <= self.epsilon,  #constraint on the quadrotor's linear velocity
                        cp.sum_squares(omega_next) <= self.epsilon]    #constraint on the quadrotor's angular velocity
        

        # Set cost function
        cost = 0
        for i in range(self.num_quads):
            for j in range(self.num_waypoints):
                cost += cp.norm(x[i, j, :3] - self.x_desired[i, j, :3])**2 # Distance from desired position
                cost += cp.norm(x[i, j, 3:7] - self.x_desired[i, j, 3:7])**2 # Distance from desired orientation
                cost += cp.norm(x[i, j, 7:10])**2 # Velocity magnitude
                cost += cp.norm(x[i, j, 10:13])**2 # Angular velocity magnitude
                cost += self.control_penalty*cp.sum_squares(u[i, j, :]) # Control penalty

        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS, verbose=True)
        if prob.status != cp.OPTIMAL:
            print("Optimization failed!")
            return None
        
        else:
            
            # Extract optimal trajectories and motor commands
            x_opt = x.value
            u_opt = u.value

            return x_opt, u_opt
        

def main():
    num_quads = 2
    num_waypoints = 30

    x_end = [2.5, 1.8, 2.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    custom_values = [[0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1.0, 1.5,  1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    custom_x_init = np.array(custom_values)


    quad_scp = MultiQuadSCP(num_quads=num_quads, num_waypoints=num_waypoints, dt = 0.05,\
                             kF = 6.11e-8, kM = 1.5e-9, g = 9.81, x_init=custom_x_init, x_end=x_end)
    x_trj, u_trj = quad_scp.solve()


if __name__ == "__main__":
    main()