from __future__ import annotations

import numpy as np
import numpy.linalg as npl
import cvxpy as cp
from typing import overload
from numpy.typing import NDArray
from matplotlib.axes import Axes


class Ellipsoid:
    def __init__(
        self,
        p: NDArray[np.float64],
        alpha: float,
        center: NDArray[np.float64] | None = None,
    ):
        try:
            _ = npl.cholesky(p)
        except npl.LinAlgError:
            raise TypeError("Matrix p must be symmetric positive definite!")

        self._p = p
        self._n_dim = p.shape[0]
        self._alpha = alpha

        self._center = np.zeros(self._n_dim) if center is None else center

    def __str__(self) -> str:
        return (
            "====================================================================================================\n"
            "(x - center).T @ p @ (x - center) <= alpha\n"
            "====================================================================================================\n"
            f"p:\n"
            f"{self._p}\n"
            "----------------------------------------------------------------------------------------------------\n"
            f"alpha:\n"
            f"{self._alpha}\n"
            "----------------------------------------------------------------------------------------------------\n"
            f"center:\n"
            f"{self._center}\n"
            "===================================================================================================="
        )

    @overload
    def contains(self, point: NDArray[np.float64]) -> bool: ...

    @overload
    def contains(self, point: cp.Expression) -> cp.Constraint: ...

    def contains(
        self, point: NDArray[np.float64] | cp.Expression
    ) -> bool | cp.Constraint:
        if isinstance(point, np.ndarray):
            res = np.all(
                (point - self._center) @ self._p @ (point - self._center) - self._alpha
                <= 0
            )
        else:
            res = cp.quad_form(point - self._center, self._p) - self._alpha <= 0

        return res  # type: ignore

    def plot(self, ax: Axes, n_points=2000, color="b") -> None:
        if self._n_dim != 2:
            raise NotImplementedError("Plotting is only implemented for 2D ellipsoids.")

        axis_max = np.sqrt(self._alpha / npl.eigvals(self._p))
        x_max, y_max = axis_max * 1.5
        x_min, y_min = -axis_max * 1.5

        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        x_grid, y_grid = np.meshgrid(x, y)
        x_grid = x_grid - self._center[0]
        y_grid = y_grid - self._center[1]

        z = (
            x_grid**2 * self._p[0, 0]
            + x_grid * y_grid * (self._p[0, 1] + self._p[1, 0])
            + y_grid**2 * self._p[1, 1]
        )

        ax.contour(x_grid, y_grid, z, levels=[self._alpha], colors=color)

    @property
    def p(self) -> np.ndarray:
        return self._p

    @property
    def n_dim(self) -> int:
        return self._n_dim

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def center(self) -> NDArray[np.float64]:
        return self._center

    def __add__(self, other: Ellipsoid | NDArray[np.float64]) -> Ellipsoid:
        if isinstance(other, Ellipsoid):
            raise NotImplementedError(
                "Minkowski sum of two ellipsoids is not implemented."
            )
        else:
            return self.__class__(self._p, self._alpha, self._center + other)

    def __sub__(self, other: Ellipsoid | NDArray[np.float64]) -> Ellipsoid:
        if isinstance(other, Ellipsoid):
            raise NotImplementedError(
                "Pontryagin difference of two ellipsoids is not implemented."
            )
        else:
            return self.__add__(-other)

    def map_inv(self, mat: NDArray[np.float64]) -> Ellipsoid:
        return self.__class__(mat.T @ self._p @ mat, self._alpha, self._center)

    def map(self, mat: NDArray[np.float64]) -> Ellipsoid:
        try:
            inv_mat = npl.inv(mat).astype(np.float64)
            res = self.map_inv(inv_mat)
        except npl.LinAlgError:
            res = NotImplemented

        return res

    # 多面体的放缩
    def __mul__(self, other: float) -> Ellipsoid:
        return self.__class__(self._p, self._alpha * other, self._center)

    def __and__(self, other: Ellipsoid) -> Ellipsoid:
        raise NotImplementedError("Intersection of two ellipsoids is not implemented.")

    def __eq__(self, other: Ellipsoid) -> bool:
        if not isinstance(other, Ellipsoid):
            return False

        centers_equal = np.array_equal(self._center, other._center)
        # check proportionality: self._p / other._p == self._alpha / other._alpha
        # rewrite as self._p * other._alpha == other._p * self._alpha and use allclose for numerics
        matrices_proportional = np.allclose(
            self._p * other._alpha, other._p * self._alpha
        )

        return bool(centers_equal and matrices_proportional)
