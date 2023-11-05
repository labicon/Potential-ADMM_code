def plot_solve(X, J, x_goal, x_dims=None, color_agents=False, n_d=2, ax=None):
    """Plot the resultant trajectory on plt.gcf()"""

    if n_d not in (2, 3):
        raise ValueError()

    if not x_dims:
        x_dims = [X.shape[1]]

    if not ax:
        if n_d == 2:
            ax = plt.gca()
        else:
            ax = plt.gcf().add_subplot(projection="3d")

    N = X.shape[0]
    n = np.arange(N)
    cm = plt.cm.Set2

    X_split = split_agents(X, x_dims)
    x_goal_split = split_agents(x_goal.reshape(1, -1), x_dims)

    for i, (Xi, xg) in enumerate(zip(X_split, x_goal_split)):
        c = n
        if n_d == 2:
            if color_agents:
                c = cm.colors[i]
                ax.plot(Xi[:, 0], Xi[:, 1], c=c, lw=2)
            else:
                ax.plot(Xi[:, 0], Xi[:, 1], c='gray', alpha=0.5, lw=2)
            # ax.plot(Xi[0, 0], Xi[0, 1], 'bo', markersize=6, label="$x_0$")
            # ax.plot(xg[0, 0], xg[0, 1], 'rx', markersize=6, label="$x_f$")
        else:
            if color_agents:
                c = cm.colors[i]
            ax.plot(Xi[:, 0], Xi[:, 1], Xi[:, 2], c=c, lw=2)
            ax.plot(Xi[0, 0], Xi[0, 1], Xi[0, 2], color=c, marker='s', markersize=6.5, \
                    markerfacecolor='none', markeredgewidth=1, label="$x_0$")
            # ax.plot(xg[0, 0], xg[0, 1], xg[0, 2], color=c, marker='^', markersize=6, \
            #         markerfacecolor='none', markeredgewidth=1, label="$x_f$")
            ax.plot(Xi[-1, 0], Xi[-1, 1], Xi[-1, 2], color=c, marker='^', markersize=6.5, \
                    markerfacecolor='none', markeredgewidth=1, label="$x_f$")

    # ax.set_xlabel('X [m]', fontdict=palatino_font)
    # ax.set_ylabel('Y [m]', fontdict=palatino_font)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    if n_d == 3:
        ax.set_zlabel('Z [m]', fontsize=12, fontweight='bold')
        ax.zaxis.set_major_formatter(FuncFormatter(format_z_ticks))
        ax.zaxis.set_ticks([0.8, 1.2])
        
    # Set X and Y axis ticks to only show 1 and 2
    ax.xaxis.set_ticks([1, 2])
    ax.yaxis.set_ticks([1.2, 2.2])

    # ax.set_title(f"Final Cost: {J:.3g}", fontsize=14, fontweight='bold')
    # ax.legend(fontsize=12)
    # Remove gridlines
    ax.grid(False)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    
    # ax.view_init(elev=20, azim=-60)
    
    # Customize tick parameters
    ax.tick_params(labelsize=10)
    
    # # Set fewer ticks on the axes
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    # if n_d == 3:
        # ax.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    plt.tight_layout()
