import numpy as np
import matplotlib.pyplot as plt
import pympc.poly as mp
import pympc.mpc as mm

if __name__ == '__main__':
    # Comparison of LQR, linear MPC = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
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

    x_set = mp.Polyhedron(A_x, b_x)
    u_set = mp.Polyhedron(A_u, b_u)

    lqr = mm.LQR(A, B, Q, R)
    mpc = mm.MPC(A, B, Q, R, N, x_set, u_set)

    terminal_set = mpc.terminal_set
    feasible_set = mpc.feasible_set

    # Simulation computation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    T = 20

    x_ini = np.array([20, -4])

    # The input constraints are compulsory
    x_lqr = np.zeros((mpc.state_dim, T + 1))
    u_lqr_nom = np.zeros((mpc.input_dim, T))
    u_lqr_real = np.zeros((mpc.input_dim, T))
    x_lqr[:, 0] = x_ini

    # Without consideration of input constraints
    # x_lqr = np.zeros((mpc_controller.state_dim, T + 1))
    # u_lqr = np.zeros((mpc_controller.input_dim, T))
    # x_lqr[:, 0] = x_ini

    x_mpc = np.zeros((mpc.state_dim, T + 1))
    u_mpc = np.zeros((mpc.input_dim, T))
    x_mpc[:, 0] = x_ini

    x_mpc_pred = np.zeros((T, x_dim, N + 1))

    for k in range(T):
        w = np.random.uniform(-0.1, 0.1, x_dim)

        # When the input constraints are compulsory, use code as below
        u_lqr_nom[:, k] = lqr(x_lqr[:, k])
        u_lqr_real[:, k] = np.clip(u_lqr_nom[:, k], -1, 1)
        x_lqr[:, k + 1] = A @ x_lqr[:, k] + B @ u_lqr_real[:, k] + w

        # Without consideration of input constraints
        # u_lqr[:, k] = lqr_controller(x_lqr[:, k])
        # x_lqr[:, k + 1] = A @ x_lqr[:, k] + B @ u_lqr[:, k] + w

        u_mpc[:, k] = mpc(x_mpc[:, k])
        x_mpc[:, k + 1] = A @ x_mpc[:, k] + B @ u_mpc[:, k] + w

        x_mpc_pred[k] = mpc.state_prediction_series

    # Results plot - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # State trajectory plot
    fig1, ax1 = plt.subplots(1, 1)
    ax1.grid(True)
    ax1.set_xlim([-10, 25])
    ax1.set_ylim([-10, 5])
    ax1.set_title('State trajectories of LQR and MPC')
    ax1.annotate('State bound', xy=(2.5, 2), xytext=(2.5, 3), arrowprops=dict(arrowstyle='-|>'))
    ax1.annotate('Terminal set', xy=(1, -0.5), xytext=(5, -2), arrowprops=dict(arrowstyle='-|>'))

    terminal_set.plot(ax1, color='k')
    x_set.plot(ax1, color='k')

    for k in range(T):
        l, u = (0, 1) if k == 0 else (k - 1, k + 1)

        line_1, = ax1.plot(x_lqr[0, l:u], x_lqr[1, l:u], color='g', marker='o', label='LQR real trajectory')

        line_2, = ax1.plot(x_mpc[0, l:u], x_mpc[1, l:u], color='b', marker='*', label='MPC real trajectory')
        line_3, = ax1.plot(x_mpc_pred[k][0, :], x_mpc_pred[k][1, :], color='r', marker='^',
                           label='MPC predicted trajectory')

        if k == 0:
            ax1.legend()

        plt.pause(0.5)

        line_3.remove()

    plt.show()

    # Control input plot
    fig2, ax2 = plt.subplots(1, 1)
    ax2.set_title('Inputs of LQR and MPC')
    ax2.set_xlim([0, T - 1])

    iterations = np.arange(T)

    # The input constraints are compulsory
    ax2.step(iterations, u_lqr_nom[0, :], label='LQR nominal input sequence')
    ax2.step(iterations, u_lqr_real[0, :], label='LQR real input sequence')

    # Without consideration of input constraints
    # ax2.step(iterations, u_lqr[0, :], label='LQR input sequence')

    ax2.step(iterations, u_mpc[0, :], label='MPC input sequence')

    ax2.step(iterations, np.ones(T) * 1, 'y-.', label='Input bounds')
    ax2.step(iterations, np.ones(T) * -1, 'y-.')

    ax2.legend(loc='upper right')

    # Feasible set of the initial state for MPC
    fig3, ax3 = plt.subplots(1, 1)
    ax3.grid(True)

    ax3.set_title('The feasible set of the initial state of MPC')

    ax3.plot(x_ini[0], x_ini[1], color='r', marker='*')
    ax3.annotate('Initial state', xy=x_ini, xytext=x_ini + 0.3 * np.abs(x_ini), arrowprops=dict(arrowstyle='-|>'))
    feasible_set.plot(ax3)

    plt.show()
