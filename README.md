# A potential-game approach to distributed trajectory optimization using MPC and SCP

## How to use
This repository is based on the [Casadi](https://web.casadi.org/) pacakge. The goal is to achieve efficient online trajectory optimization leveraging both distributed MPC and distributed sequential convex programming. 


## Objective
Different from conventional distributed MPC algorithms, we propose to solve the potential game of a subproblem which consists of agents in the same neighborhood. By leveraging the potential game, a neighborhood of multi-agent planning problem can be solved as a single-agent optimization problem. Instead of treating each agent as its own subproblem, our distributed MPC algorithm groups multiple agents into a cluster and solves its potential function. By doing so, we eliminate the need for a central node. The communication overhead is low because agents only exchange position information once before they solve their next optimization problem (in a receding horizon). For a more in-depth explanation on game-theoretic trajectory planning, please refer to the paper [Potential iLQR: A Potential-Minimizing Controller for Planning Multi-Agent Interactive Trajectories](https://arxiv.org/abs/2107.04926).
