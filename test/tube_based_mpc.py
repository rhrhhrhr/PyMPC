import numpy as np
import matplotlib.pyplot as plt
import pympc.poly as mp
import pympc.mpc as mm

if __name__ == '__main__':
    # Tube based MPC = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # Initialization - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    N = 9
    x_dim = 2
    u_dim = 1

    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0.5], [1]])
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[0.01]])

    A_x = np.array([[0, 1]])
    b_x = np.array([2])
    A_u = np.array([[1], [-1]])
    b_u = np.array([1, 1])
    A_w = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b_w = np.array([0.1, 0.1, 0.1, 0.1])

    x_set = mp.Polyhedron(A_x, b_x)
    u_set = mp.Polyhedron(A_u, b_u)
    w_set = mp.Polyhedron(A_w, b_w)

    # 各种集合的计算量较大，可能会花费较长时间
    t_mpc = mm.TubeBasedMPC(A, B, Q, R, N, x_set, u_set, w_set)

    disturbance_invariant_set = t_mpc.disturbance_invariant_set
    terminal_set = t_mpc.terminal_set
    terminal_set_plus_di = terminal_set + disturbance_invariant_set
    feasible_set_bar = t_mpc.feasible_set
    feasible_set = feasible_set_bar + disturbance_invariant_set

    # Simulation computation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Simulation time
    T = 9

    # Initial state
    x_ini = np.array([-5, -2])

    # Record the real state and input during every iteration
    x_t_mpc = np.zeros((t_mpc.state_dim, T + 1))
    u_t_mpc = np.zeros((t_mpc.input_dim, T))
    x_t_mpc[:, 0] = x_ini

    # Record the predicted trajectory during every iteration
    x_t_mpc_pred = np.zeros((T, x_dim, N + 1))

    for k in range(T):
        w = np.random.uniform(-0.1, 0.1, x_dim)

        u_t_mpc[:, k] = t_mpc(x_t_mpc[:, k])
        x_t_mpc[:, k + 1] = A @ x_t_mpc[:, k] + B @ u_t_mpc[:, k] + w

        x_t_mpc_pred[k] = t_mpc.state_prediction_series

    # Test for disturbance invariant set
    T_test = 20
    x_test = np.zeros((x_dim, T_test + 1))

    A_k = A - B @ t_mpc.k

    for k in range(T_test):
        w_test = np.random.uniform(-0.1, 0.1, x_dim)
        x_test[:, k + 1] = A_k @ x_test[:, k] + w_test

    # Results plot - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # State trajectory plot
    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_title('State trajectory of tube based MPC')
    ax1.grid(True)
    ax1.set_xlim([-8, 4])
    ax1.set_ylim([-3, 3])

    terminal_set.plot(ax1, color='k')
    terminal_set_plus_di.plot(ax1, color='k')
    x_set.plot(ax1, color='k')

    ax1.annotate('State bound', xy=(-6, 2), xytext=(-6, 2.5), arrowprops=dict(arrowstyle='-|>'))
    ax1.annotate('Terminal set', xy=(1.1, -0.3), xytext=(1.1, 0.5), arrowprops=dict(arrowstyle='-|>'))
    ax1.annotate('Terminal set + Disturbance invariant set', xy=(0.5, -0.9), xytext=(-4, -2.5),
                 arrowprops=dict(arrowstyle='-|>'))

    tube_text_pos = x_t_mpc_pred[3, :, 0] + np.array([0.25, -0.25])
    ax1.annotate('Tube', xy=tube_text_pos, xytext=tube_text_pos - 0.2 * tube_text_pos,
                 arrowprops=dict(arrowstyle='-|>'))

    for k in range(T):
        tube = disturbance_invariant_set + x_t_mpc_pred[k, :, 0]
        tube.plot(ax1, color='k')

    for k in range(T):
        l, u = (0, 1) if k == 0 else (k - 1, k + 1)

        line_1, = ax1.plot(x_t_mpc[0, l:u], x_t_mpc[1, l:u], color='b', marker='*', label='MPC real trajectory')
        line_2, = ax1.plot(x_t_mpc_pred[l:u, 0, 0].reshape(-1), x_t_mpc_pred[l:u, 1, 0].reshape(-1), color='g',
                           marker='s', label='MPC nominal trajectory')
        line_3, = ax1.plot(x_t_mpc_pred[k][0, :], x_t_mpc_pred[k][1, :], color='r', marker='^',
                           label='MPC predicted trajectory')

        if k == 0:
            ax1.legend()

        plt.pause(1)

        line_3.remove()

    plt.show()

    # Control input plot
    fig2, ax2 = plt.subplots(1, 1)
    ax2.set_title('Inputs of the tube based MPC')

    iterations = np.arange(T)

    ax2.plot(iterations, u_t_mpc[0, :], label='Tube based MPC input sequence')

    ax2.step(iterations, np.ones(T) * 1, 'y--', label='Input bounds')
    ax2.step(iterations, np.ones(T) * -1, 'y--')

    ax2.legend(loc='upper right')

    # Feasible set of the initial state for MPC
    fig3, ax3 = plt.subplots(1, 1)
    ax3.set_title('Feasible set for the initial state of tube based MPC')
    ax3.grid(True)

    ax3.annotate('Initial state', xy=x_ini, xytext=x_ini + 0.3 * np.abs(x_ini), arrowprops=dict(arrowstyle='-|>'))
    ax3.annotate('Feasible set for initial state\nin controller', xy=(10, -4), xytext=(-20, -6),
                 arrowprops=dict(arrowstyle='-|>'))
    ax3.annotate('Feasible set for initial state', xy=(20, -5.8), xytext=(0, -8), arrowprops=dict(arrowstyle='-|>'))
    feasible_set_bar.plot(ax3, color='r')
    feasible_set.plot(ax3, color='b')

    ax3.plot(x_ini[0], x_ini[1], color='r', marker='*')

    plt.show()

    # State trajectory of the test for disturbance invariant set
    fig4, ax4 = plt.subplots(1, 1)
    ax4.set_title('State trajectory of the test for disturbance invariant set')
    ax4.grid(True)

    disturbance_invariant_set.plot(ax4, x_lim=[-0.3, 0.3], y_lim=[-0.3, 0.3], color='k')

    for k in range(T_test):
        l, u = (0, 1) if k == 0 else (k - 1, k + 1)
        ax4.plot(x_test[0, l:u], x_test[1, l:u], color='r', marker='*', label='Test state trajectory')

        if k == 0:
            pass

        plt.pause(1)

    plt.show()
