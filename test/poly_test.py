import numpy as np
import matplotlib.pyplot as plt
import pympc.set as mp

if __name__ == '__main__':
    # Test for set = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    fig1, ax1 = plt.subplots(1, 1)
    ax1.grid(True)
    ax1.axis('equal')
    ax1.set_title('Translation')

    p1 = mp.Polyhedron(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), np.array([1, 0, 1, 0]))
    p2 = mp.Polyhedron(np.array([[-3, 1], [-1, -1], [1, 0]]), np.array([3, 1, 0]))
    p3 = np.array([0, 2]) + p1

    p1.plot(ax1, color='b')
    p3.plot(ax1, color='r')

    fig2, ax2 = plt.subplots(1, 1)
    ax2.grid(True)
    ax2.axis('equal')
    ax2.set_title('Minkowski sum')

    p4 = p1 + p2

    p1.plot(ax2, color='b')
    p2.plot(ax2, color='r')
    p4.plot(ax2, color='g')

    fig3, ax3 = plt.subplots(1, 1)
    ax3.grid(True)
    ax3.axis('equal')
    ax3.set_title('Pontryagin difference')

    p5 = p4 - p1
    p6 = p4 + (-p1)

    p1.plot(ax3, color='b')
    p4.plot(ax3, color='r')
    p5.plot(ax3, color='g')
    p6.plot(ax3, color='k')

    fi4, ax4 = plt.subplots(1, 1)
    ax4.grid(True)
    ax4.axis('equal')
    ax4.set_title('Coordinate transformation')

    theta = np.deg2rad(90)
    s, c = np.sin(theta), np.cos(theta)
    rot_mat = np.array([[c, -s], [s, c]])

    p7 = p2 @ rot_mat
    p8 = rot_mat @ p2

    p2.plot(ax4, color='b')
    p7.plot(ax4, color='r')
    p8.plot(ax4, color='g')

    fi5, ax5 = plt.subplots(1, 1)
    ax5.grid(True)
    ax5.axis('equal')
    ax5.set_title('$R^2$')

    R2 = mp.Rn(2)

    R2.plot(ax5)

    fi6, ax6 = plt.subplots(1, 1)
    ax6.grid(True)
    ax6.axis('equal')
    ax6.set_title('Unit cube')

    unit_cube = mp.Polyhedron(np.vstack((np.eye(2), -np.eye(2))), np.ones(2 * 2))

    unit_cube.plot(ax6)

    plt.show()
