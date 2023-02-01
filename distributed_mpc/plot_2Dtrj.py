#!/usr/bin/env python
​
"""Rendering of the trials used in hardware experiments for the video presentation."""
​
​
import matplotlib.pyplot as plt
import numpy as np
​
​
trajs = [
    (
        np.array(
            [[0.5, 1.5, 1, 0, 0, 0,
                    2.5, 1.5, 1, 0, 0, 0,
                    1.5, 1.3, 1, 0, 0, 0,
                    -0.5 ,0.5, 1, 0, 0, 0,
                   1.1, 1.1, 1, 0, 0, 0]]
                ),
        np.array(
            [
                [[2.5, 1.5, 1, 0, 0, 0, 
                    0.5, 1.5, 1, 0, 0, 0, 
                    1.5, 2.2, 1, 0, 0, 0,
                   -0.5, 1.5, 1, 0, 0, 0,
               -1.1, 1.1, 1, 0, 0, 0]]
            ]
        ),
    ),
    (
        np.array(
            [
                [0.449, 1.695, 1.063],
                [2.462, 1.621, 0.963],
                [1.555, 1.237, 0.953],
                [0.328, 1.043, 0.947],
                [1.179, -0.503, 0.971],
            ]
        ),
        np.array(
            [
                [2.619, 1.6, 1.026],
                [0.463, 1.487, 1.033],
                [1.323, 2.129, 1.216],
                [-0.523, -0.647, 0.896],
                [0.658, 0.977, 1.133],
            ]
        ),
    ),
    (
        np.array(
            [
                [0.374, 1.497, 0.855],
                [2.636, 1.399, 1.151],
                [1.496, 1.245, 1.065],
                [0.591, 1.239, 1.126],
                [1.124, -0.56, 1.063],
            ]
        ),
        np.array(
            [
                [2.649, 1.399, 1.034],
                [0.502, 1.671, 1.054],
                [1.607, 2.172, 1.09],
                [-0.606, -0.711, 0.909],
                [0.791, 0.862, 0.928],
            ]
        ),
    ),
]
​
​
def set_bounds(xydata, ax=None, zoom=0.1):
    """Set the axis on plt.gca() by some margin beyond the data, default 10% margin
​
    Reference:
    https://github.com/zjwilliams20/pocketknives/blob/main/pocketknives/python/graphics.py
​
    """
​
    xydata = np.atleast_2d(xydata)
​
    if not ax:
        ax = plt.gca()
​
    xmarg = xydata[:, 0].ptp() * zoom
    ymarg = xydata[:, 1].ptp() * zoom
    ax.set(
        xlim=(xydata[:, 0].min() - xmarg, xydata[:, 0].max() + xmarg),
        ylim=(xydata[:, 1].min() - ymarg, xydata[:, 1].max() + ymarg),
    )
​
​
def render_trials():
​
    fig = plt.figure()
    plt.style.use("dark_background")
​
    n_agents, n_states = trajs[0][0].shape
​
    t = 2
    plt.clf()
    for i, (x0, xf) in enumerate(trajs[t : t + 1]):
        ax = fig.add_subplot(1, 1, i + 1)
        ax.set_aspect("equal")
        # ax.set_facecolor("k")
        # ax.yaxis.set_tick_params(labelcolor="w", color="w")
        # ax.xaxis.set_tick_params(
        X = np.dstack(
            [x0.reshape(n_agents, n_states), xf.reshape(n_agents, n_states)]
        ).swapaxes(1, 2)
        for i, Xi in enumerate(X):
            plt.annotate(
                "",
                Xi[1, :2],
                Xi[0, :2],
                arrowprops=dict(facecolor=plt.cm.Dark2.colors[i]),
            )
        set_bounds(X.reshape(-1, n_states), zoom=0.1)
        plt.draw()
    plt.savefig(f"trial{t+1}.png", dpi=1000.0)
​
​
if __name__ == "__main__":
    render_trials()