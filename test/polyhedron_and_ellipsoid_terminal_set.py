import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pympc.set as ms
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

    x_set = ms.Polyhedron(A_x, b_x)
    u_set = ms.Polyhedron(A_u, b_u)

    mpc_poly = mm.MPC(A, B, Q, R, N, x_set, u_set, terminal_set_type='polyhedron')
    mpc_elli = mm.MPC(A, B, Q, R, N, x_set, u_set, terminal_set_type='ellipsoid', solver=cp.ECOS)

    poly_terminal_set = mpc_poly.terminal_set
    elli_terminal_set = mpc_elli.terminal_set

    # Simulation computation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    T = 15

    x_ini = np.array([20, -4])
    # x_ini = np.array([32, -6])

    # The input constraints are compulsory
    x_mpc_poly = np.zeros((mpc_poly.state_dim, T + 1))
    u_mpc_poly = np.zeros((mpc_poly.input_dim, T))
    x_mpc_poly[:, 0] = x_ini

    x_mpc_elli = np.zeros((mpc_elli.state_dim, T + 1))
    u_mpc_elli = np.zeros((mpc_elli.input_dim, T))
    x_mpc_elli[:, 0] = x_ini

    x_mpc_poly_pred = np.zeros((T, x_dim, N + 1))
    x_mpc_elli_pred = np.zeros((T, x_dim, N + 1))

    for k in range(T):
        w = np.random.uniform(-0.1, 0.1, x_dim)

        # When the input constraints are compulsory, use code as below
        u_mpc_poly[:, k] = mpc_poly(x_mpc_poly[:, k])
        x_mpc_poly[:, k + 1] = A @ x_mpc_poly[:, k] + B @ u_mpc_poly[:, k] + w

        u_mpc_elli[:, k] = mpc_elli(x_mpc_elli[:, k])
        x_mpc_elli[:, k + 1] = A @ x_mpc_elli[:, k] + B @ u_mpc_elli[:, k] + w

        x_mpc_poly_pred[k] = mpc_poly.state_prediction_series
        x_mpc_elli_pred[k] = mpc_elli.state_prediction_series

    # Results plot - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # State trajectory plot
    fig1, ax1 = plt.subplots(1, 1)
    ax1.grid(True)
    ax1.set_xlim([-5, 25])
    ax1.set_ylim([-10, 5])
    ax1.set_title('State trajectories')
    ax1.annotate('State bound', xy=(2.5, 2), xytext=(2.5, 3), arrowprops=dict(arrowstyle='-|>'))
    ax1.annotate('Polyhedron terminal set', xy=(2, -1.2), xytext=(5, -2), arrowprops=dict(arrowstyle='-|>'))
    ax1.annotate('Ellipsoid terminal set', xy=(0.2, -0.4), xytext=(5, 0), arrowprops=dict(arrowstyle='-|>'))

    poly_terminal_set.plot(ax1, color='k')
    elli_terminal_set.plot(ax1, color='k')
    x_set.plot(ax1, color='k')

    for k in range(T):
        l, u = (0, 1) if k == 0 else (k - 1, k + 1)

        line_1, = ax1.plot(x_mpc_poly[0, l:u], x_mpc_poly[1, l:u], color='b', marker='*', label='Poly real trajectory')
        line_2, = ax1.plot(x_mpc_poly_pred[k][0, :], x_mpc_poly_pred[k][1, :], color='r', marker='^',
                           label='Poly predicted trajectory')

        line_3, = ax1.plot(x_mpc_elli[0, l:u], x_mpc_elli[1, l:u], color='g', marker='x', label='Elli real trajectory')
        line_4, = ax1.plot(x_mpc_elli_pred[k][0, :], x_mpc_elli_pred[k][1, :], color='y', marker='o',
                           label='Elli predicted trajectory')

        if k == 0:
            ax1.legend()

        plt.pause(1)

        line_2.remove()
        line_4.remove()

    plt.show()

    # Control input plot
    fig2, ax2 = plt.subplots(1, 1)
    ax2.set_title('Input sequences')
    ax2.set_xlim([0, T - 1])

    iterations = np.arange(T)

    # The input constraints are compulsory
    ax2.step(iterations, u_mpc_poly[0, :], label='Polyhedron')
    ax2.step(iterations, u_mpc_elli[0, :], label='Ellipsoid')

    ax2.step(iterations, np.ones(T) * 1, 'y-.', label='Input bounds')
    ax2.step(iterations, np.ones(T) * -1, 'y-.')

    ax2.legend(loc='upper right')

    plt.show()
