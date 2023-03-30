import numpy as np
from scipy import sparse, optimize
from scipy.interpolate import interp1d


def getPosMat(h, K):
    # Kinematic model A,b matrices
    A = np.array([[1, 0, 0, h, 0, 0],
                  [0, 1, 0, 0, h, 0],
                  [0, 0, 1, 0, 0, h],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    b = np.vstack((h**2/2 * np.eye(3), h * np.eye(3)))

    Apos = np.zeros((3*K, 3*K)) # local inequality constraint variable
    prev_row = np.zeros((6, 3*K)) # For the first iteration of constructing matrix Ain
    idx = 0
    # Build matrix to convert acceleration to position
    for k in range(1, K+1):
        add_b = np.hstack((np.zeros((b.shape[0], (k-1)*b.shape[1])), b,
                           np.zeros((b.shape[0], (K-k)*b.shape[1]))))
        new_row = A @ prev_row + add_b
        Apos[idx:idx+3, :] = new_row[:3, :]
        prev_row = new_row
        idx += 3

    return Apos

def getPosVelMat(h, K):
    # Kinematic model A,b matrices
    A = np.array([[1, 0, 0, h, 0, 0],
                  [0, 1, 0, 0, h, 0],
                  [0, 0, 1, 0, 0, h],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    b = np.vstack((h**2/2 * np.eye(3), h * np.eye(3)))

    prev_row = np.zeros((6, 3*K))  # For the first iteration of constructing matrix Ain

    # Build matrix to convert acceleration to position
    for k in range(1, K+1):
        add_b = np.hstack((np.zeros((b.shape[0], b.shape[1]*(k-1))), b, np.zeros((b.shape[0], b.shape[1]*(K-k)))))
        new_row = np.dot(A, prev_row) + add_b
        prev_row = new_row

    Aaug = np.vstack((new_row, np.hstack((np.zeros((3, 3*(K-1))), np.eye(3))), \
                  np.hstack((np.eye(3), np.zeros((3, 3*(K-1))))))) 

    return Aaug


def getStates(po, atot, A_p, A_v, K, N):
    vo = np.array([0, 0, 0])
    po = np.squeeze(po)

    p = np.zeros((3*K, N))
    v = np.zeros((3*K, N))
    a = np.zeros((3, K, N))

    for i in range(N):
        ai = atot[3*K*(i-1):3*K*i]
        a[:, :, i] = np.reshape(ai, (3, K), order='F')
        new_p = A_p.dot(ai)
        new_v = A_v.dot(ai)
        pi = np.vstack((po[:, i], new_p + np.tile(po[:, i], (K-1, 1))))
        vi = np.vstack((vo, new_v))
        p[:, i] = np.reshape(pi, (3*K,), order='F')
        v[:, i] = np.reshape(vi, (3*K,), order='F')

    return p, v, a


def initAllSolutions(po, pf, h, K):
    po = np.squeeze(po)
    pf = np.squeeze(pf)
    N = po.shape[1]
    t = np.arange(0, K*h, h)
    p = np.zeros((po.shape[0], len(t), N))
    for i in range(N):
        diff = pf[:,i] - po[:,i]
        for k in range(len(t)):
            p[:,k,i] = po[:,i] + t[k]*diff/((K-1)*h)
    return p

def AddCollConstr(p, po, K, rmin, A, E1, E2, order):
    N = p.shape[2]
    Ain_total = np.zeros((K*N*(N-1)//2, 3*K*N))
    bin_total = np.zeros((K*N*(N-1)//2, 1))
    l = 0

    for i in range(N-1):
        pi = p[:,:,i]
        for j in range(i+1, N+1):
            pj = p[:,:,j]
            for k in range(K):
                dist = np.linalg.norm(E1 @ (pi[:,k]-pj[:,k]), ord=order)
                diff = (E2 @ (pi[:,k]-pj[:,k]))**(order-1)

                # Right side of inequality constraint (bin)
                r = dist**(order-1)*(rmin - dist) + diff @ (pi[:,k]-pj[:,k]) - diff @ (po[:,:,i].T-po[:,:,j].T)
                bin_total[l] = r

                # Construct diagonal matrix with vector difference
                diff_mat = np.hstack((np.zeros((1, 3*K*(i-1))), np.zeros((1, 3*(k-1))), diff.reshape(1,3),
                      np.zeros((1, 3*(K-k))), np.zeros((1, 3*K*(j-i-1))), np.zeros((1, 3*(k-1))),
                      -diff.reshape(1,3), np.zeros((1, 3*(K-k))), np.zeros((1, 3*K*(N-j)))))



                # Update the ineq. constraints matrix and vector
                Ain_total[l,:] = -diff_mat @ A
                l += 1

    return Ain_total, bin_total


def solveCupSCP(po, pf, h, K, N, pmin, pmax, rmin, alim, A_p, A_v, E1, E2, order):
    prev_p = initAllSolutions(po, pf, h, K)
    ub = alim * np.ones(3 * N * K)
    lb = -ub
    H = np.eye(3 * K * N)
    A = getPosMat(h, K)
    Atot = np.kron(np.eye(N), A)
    Aeq = getPosVelMat(h, K)
    Aeqtot = np.kron(np.eye(N), Aeq)
    criteria = 2
    epsilon = 1
    p_constr_h = np.zeros((3 * K, N))
    p_constr_l = np.zeros((3 * K, N))
    beq_i = np.zeros((12, N))
    for i in range(N):
        p_constr_h[:, i] = np.tile((pmax - po[:, :, i]).T, K).flatten()
        p_constr_l[:, i] = np.tile(-(pmin - po[:, :, i]).T, K).flatten()
        
        beq_i[:, i] = np.vstack(((pf[:, :, i] - po[:, :, i]).T, np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)))).reshape(-1)
   

    
    bound_h = np.reshape(p_constr_h, (-1, 1), order='F')
    bound_l = np.reshape(p_constr_l, (-1, 1), order='F')
    beqtot = np.reshape(beq_i, (-1, 1), order='F')
    k = 1
    prev_f0 = 2
    while criteria > epsilon or k <= 2:
        # Inequality Constraints
        Ain, bin = AddCollConstr(prev_p, po, K, rmin, Atot, E1, E2, order)
        Ain_total = np.vstack((Ain, Atot, -Atot))
        bin_total = np.vstack((bin.reshape(-1, 1), bound_h, bound_l))

        # Solve the QP
        atot, f0, exitflag = optimize.quadprog(sparse.csc_matrix(H), np.array([]), sparse.csc_matrix(Ain_total),
                                                bin_total, sparse.csc_matrix(Aeqtot), beqtot, lb, ub,
                                                method='interior-point')

        if atot.size == 0 or exitflag != 1:
            p = np.empty((3, K, N))
            v = np.empty((3, K, N))
            a = np.empty((3, K, N))
            success = 0
            return p, v, a, success

        success = exitflag
        p, v, a = getStates(po, atot, A_p, A_v, K, N)
        prev_p = p
        criteria = abs(prev_f0 - f0)
        prev_f0 = f0
        k += 1

    return p, v, a, success



"""Main loop"""
# Time settings and variables
T = 15  # Trajectory final time
h = 0.1  # time step duration
tk = np.arange(0, T+h, h)
K = int(T/h) + 1  # number of time steps
print(f'K is {K}')
Ts = 0.01  # period for interpolation @ 100Hz
t = np.arange(0, T+Ts, Ts)  # interpolated time vector
success = 1

# Workspace boundaries
pmin = [-3.0,-3.0,0.9]
pmax = [3.0,3.0,2.5]

# Minimum distance between vehicles in m
rmin_init = 0.6

# Variables for ellipsoid constraint
order = 2  # choose between 2 or 4 for the order of the super ellipsoid
rmin = 0.35  # X-Y protection radius for collisions
c = 2.0  # make this one for spherical constraint
E = np.diag([1,1,c])
E1 = np.linalg.inv(E)
E2 = np.linalg.inv(np.power(E, order))

# Maximum acceleration in m/s^2
alim = 1.0

N = 6  # number of vehicles

# Initial positions
po1 = [0.683, 1.443, 0.975]
po2 = [2.5, 1.484, 0.974]
po3 = [1.538, 1.332, 1.07]
po4 = [0.476, 1.006, 1.171]
po5 = [1.079, -0.6, 0.99]
po6 = [-0.5, 0, 1.2]
po = np.dstack((po1, po2, po3, po4, po5, po6))

# Final positions
pf1 = [2.412, 1.6, 0.998]
pf2 = [0.533, 1.491, 0.93]
pf3 = [1.487, 2.281, 1.123]
pf4 = [-0.612, -0.546, 0.953]
pf5 = [1.179, 0, 0.971]
pf6 = [1.1, 1.0, 1.2]
pf = np.dstack((pf1, pf2, pf3, pf4, pf5, pf6))

# Some Precomputations
p = np.zeros((3, len(t), N))
v = np.zeros((3, len(t), N))
a = np.zeros((3, len(t), N))

# Kinematic model A,b matrices
# these matrices are derived in "Trajectory Generation for Multiagent
# Point-To-Point Transitions via Distributed Model Predictive Control"
A = np.array([[1, 0, 0, h, 0, 0],
              [0, 1, 0, 0, h, 0],
              [0, 0, 1, 0, 0, h],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

b = np.vstack((np.power(h, 2)/2 * np.eye(3), h * np.eye(3)))

prev_row = np.zeros((6, 3*K))  # For the first iteration of constructing matrix Ain
A_p = np.zeros((3*(K-1), 3*K))
A_v = np.zeros((3*(K-1), 3*K))
idx = 0

print(A_p.shape)
# Build matrix to convert acceleration to position
for k in range(1, K):
    add_b = np.concatenate([np.zeros((b.shape[0], (k-1)*b.shape[1])), b, np.zeros((b.shape[0], (K-k)*b.shape[1]))], axis=1)
    print(add_b.shape)
    new_row = A @ prev_row + add_b
    print(new_row.shape)   
    A_p[idx:idx+3, :] = new_row[0:3, :]
    A_v[idx:idx+3, :] = new_row[3:6, :]
    prev_row = new_row
    idx += 3

# Solve SCP
pk, vk, ak, success = solveCupSCP(po, pf, h, K, N, pmin, pmax, rmin, alim, A_p, A_v, E1, E2, order)

# Interpolate solution with a 100Hz sampling
t = np.linspace(0, N*h, N)
p, v, a = np.zeros((3, 4, N)), np.zeros((3, 4, N)), np.zeros((3, 4, N))
for i in range(N):
    p_interp = interp1d(t, pk[:, :, i], axis=1, kind='cubic')
    v_interp = interp1d(t, vk[:, :, i], axis=1, kind='cubic')
    a_interp = interp1d(t, ak[:, :, i], axis=1, kind='cubic')
    p[:, :, i] = p_interp(tk)
    v[:, :, i] = v_interp(tk)
    a[:, :, i] = a_interp(tk)

totdist_cup = np.sum(np.sqrt(np.sum(np.diff(p, axis=2)**2, axis=0)))
print(f"The sum of trajectory length is {totdist_cup:.2f}")
print(success)
