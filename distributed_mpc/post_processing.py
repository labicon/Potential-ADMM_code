from pylab import plot, step, figure, legend, show, spy
from casadi import *
from distributed_mpc.problem import *

def postProcess(quadProblem):

    plot(quadProblem.sol.value()[0,:],label="p_x")
    plot(quadProblem.sol.value()[1,:],label="p_y")
    plot(quadProblem.sol.value()[2,:],label="p_z")

    plot(quadProblem.sol.value()[3,:],label="theta")
    plot(quadProblem.sol.value()[4,:],label="phi")
    plot(quadProblem.sol.value()[5,:],label="tau")

    legend(loc="upper left")

    figure()
    spy(quadProblem.sol.value(jacobian(quadProblem.opti.g,quadProblem.opti.x)))
    figure()
    spy(quadProblem.sol.value(hessian(quadProblem.opti.f+dot(quadProblem.opti.lam_g,quadProblem.opti.g),quadProblem.opti.x)[0]))

    show()