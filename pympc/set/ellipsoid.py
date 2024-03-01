import numpy as np
import numpy.linalg as npl
import cvxpy as cp
import matplotlib.pyplot as plt


class EllipsoidException(Exception):
    pass


class Ellipsoid(object):
    def __init__(self, p: np.ndarray, alpha: int or float, center: np.ndarray = None):
        try:
            _ = npl.cholesky(p)
        except npl.LinAlgError:
            raise EllipsoidException('The matrix of ellipsoid must be positive definite!')

        self.__p = p
        self.__n_dim = p.shape[0]
        self.__alpha = alpha

        self.__center = np.zeros(self.__n_dim) if center is None else center

    def __call__(self, point: np.ndarray or cp.Expression) -> np.ndarray or cp.Expression:
        if isinstance(point, cp.Expression):
            res = cp.quad_form(point - self.__center, self.__p) - self.__alpha
        else:
            res = (point - self.__center, self.__p) @ self.__p @ (point - self.__center, self.__p) - self.__alpha

        return res

    def is_interior_point(self, point: np.ndarray) -> bool:
        return np.all((point - self.__center) @ self.__p @ (point - self.__center) - self.__alpha <= 0)

    def plot(self, ax: plt.Axes, n_points=2000, color='b'):
        if self.__n_dim != 2:
            raise EllipsoidException('Only 2D ellipsoid can be plotted!')

        axis_max = np.sqrt(self.__alpha / npl.eigvals(self.__p))
        x_max, y_max = axis_max * 1.5
        x_min, y_min = - axis_max * 1.5

        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        x_grid, y_grid = np.meshgrid(x, y)
        x_grid = x_grid - self.__center[0]
        y_grid = y_grid - self.__center[1]

        z = (x_grid ** 2 * self.__p[0, 0] +
             x_grid * y_grid * (self.__p[0, 1] + self.__p[1, 0]) +
             y_grid ** 2 * self.__p[1, 1])

        ax.contour(x_grid, y_grid, z, levels=[self.__alpha], colors=color)
