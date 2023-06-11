from multiprocessing import Process, Pipe
import cvxpy as cp
import numpy as np

# Number of terms f_i.
N = 3
# A list of all the f_i.
T = 10  # Prediction horizon
n_states = 6
n_inputs = 3
Q_try = np.eye(6)
R_try = np.eye(3) * 0.1

n_agents = 3
nx = n_agents*n_states
nu = n_agents*n_inputs
x = cp.Variable((T + 1, nx))
u_input = cp.Variable((T, nu))
n = n_states * N

constr = []
cost_i = 0
f_list = []

for id in range(n_agents):
    for t in range(T):
        cost_i += cp.quad_form(x[t + 1, id * n_states:(id + 1) * n_states], Q_try) \
                  + cp.quad_form(u_input[t, id * n_inputs:(id + 1) * n_inputs], R_try)
    cost_i += cp.quad_form(x[-1, id * n_states:(id + 1) * n_states], Q_try * 10)
    f_list.append(cost_i)


def run_worker(f, pipe):
    rho = 1
    xbar = cp.Parameter(n, value=np.zeros(n))
    u = cp.Parameter(n, value=np.zeros(n))  # This is the scaled Lagrange multiplier
    for horizon in range(T):
        f += (rho / 2) * cp.sum_squares(x[horizon] - xbar + u)
    prox = cp.Problem(cp.Minimize(f))
    # ADMM loop.
    iter = 0
    while iter < T:
        prox.solve()
        pipe.send(x.value[iter])
        xbar.value = pipe.recv()
        u.value += x.value[iter] - xbar.value
        iter += 1

if __name__ == '__main__':
    # Setup the workers.
    pipes = []
    procs = []
    for i in range(N):
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=run_worker, args=(f_list[i], remote))]
        procs[-1].start()

    # ADMM loop.
    MAX_ITER = 50
    for i in range(MAX_ITER):
        # Gather and average xi
        xbar = sum(pipe.recv() for pipe in pipes) / N
        # Scatter xbar
        for pipe in pipes:
            pipe.send(xbar)

    # Close the pipes
    for pipe in pipes:
        pipe.close()

    # Terminate the child processes
    for p in procs:
        p.terminate()