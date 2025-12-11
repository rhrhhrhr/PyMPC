from __future__ import annotations

import numpy as np
import numpy.linalg as npl
import cvxpy as cp
from typing import overload
from numpy.typing import NDArray
from matplotlib.axes import Axes
from .support_fn import support_fn
from .ellipsoid import Ellipsoid


class InscribedEllipsoidException(Exception):
    def __init__(self):
        super().__init__(
            "Cannot find a maximum inscribed ellipsoid since the center is not in the polyhedron!"
        )


class Polyhedron:
    # 用线性不等式组 A @ x <= b 来表示一个多边形
    # l_mat: A
    # r_vec: b

    def __init__(self, l_mat: np.ndarray, r_vec: np.ndarray):
        if l_mat.shape[0] != r_vec.shape[0]:
            raise ValueError(
                "The number of rows of l_mat must equal to the size of r_vec!"
            )

        self._l_mat = l_mat
        self._r_vec = r_vec

    @property
    def n_faces(self) -> int:
        return self._l_mat.shape[0]

    @property
    def n_dim(self) -> int:
        return self._l_mat.shape[1]

    @property
    def l_mat(self) -> NDArray[np.float64]:
        return self._l_mat

    @property
    def r_vec(self) -> NDArray[np.float64]:
        return self._r_vec

    # 打印该多面体的信息
    def __str__(self) -> str:
        return (
            "====================================================================================================\n"
            "left matrix @ x <= right vector\n"
            "====================================================================================================\n"
            f"left matrix:\n"
            f"{self._l_mat}\n"
            "----------------------------------------------------------------------------------------------------\n"
            f"right vector:\n"
            f"{self._r_vec}\n"
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
            return np.all(self._l_mat @ point - self._r_vec <= 0).astype(bool)
        elif isinstance(point, cp.Expression):
            return self._l_mat @ point - self._r_vec <= 0
        else:
            raise TypeError("Point must be either a numpy array or a cvxpy expression.")

    # 多边形绘图会有误差，因为采用等高线来画的，增加采样点的个数可以提高画图精度
    def plot(
        self,
        ax: Axes,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        default_bound=100,
        n_points=2000,
        color="b",
    ) -> None:
        if self.n_dim != 2:
            raise NotImplementedError("Plotting is only implemented for 2D polyhedra.")
        if x_lim is None:
            x_min = -support_fn(np.array([-1, 0]), self)
            x_max = support_fn(np.array([1, 0]), self)
            x_lim = self.get_grid_lim(x_min, x_max, default_bound)
        if y_lim is None:
            y_min = -support_fn(np.array([0, -1]), self)
            y_max = support_fn(np.array([0, 1]), self)
            y_lim = self.get_grid_lim(y_min, y_max, default_bound)

        x = np.linspace(x_lim[0], x_lim[1], int(n_points))
        y = np.linspace(y_lim[0], y_lim[1], int(n_points))
        x_grid, y_grid = np.meshgrid(x, y)

        z = np.sum(
            np.maximum(
                self._l_mat[:, 0, np.newaxis, np.newaxis] * x_grid
                + self._l_mat[:, 1, np.newaxis, np.newaxis] * y_grid
                - self._r_vec[:, np.newaxis, np.newaxis],
                0,
            ),
            axis=0,
        )

        ax.contour(x_grid, y_grid, z, levels=0, colors=color)

    # 绘制一个多面体时需要知道大概范围
    @staticmethod
    def get_grid_lim(
        val_min: float, val_max: float, default_bound: float
    ) -> tuple[float, float]:
        if val_min == -float("inf") and val_max == float("inf"):
            lim = (-default_bound, default_bound)
        elif val_min == -float("inf") and val_max != float("inf"):
            bound = abs(val_max) * 1.2
            lim = (-bound, bound)
        elif val_min != -float("inf") and val_max == float("inf"):
            bound = abs(val_min) * 1.2
            lim = (-bound, bound)
        else:
            margin = 0.1 * (val_max - val_min)
            lim = (val_min - margin, val_max + margin)

        return lim

    def subset_eq(self, other: Polyhedron) -> bool:
        return all(
            [
                support_fn(other._l_mat[i, :], self) <= other._r_vec[i]
                for i in range(other.n_faces)
            ]
        )

    # 将不等式组右侧向量归一化，防止系数过大，影响支撑函数（线性规划）求解
    def normalization(self) -> None:
        r_vec_abs = np.where(self._r_vec == 0, 1, np.abs(self._r_vec))
        self._l_mat = self._l_mat / r_vec_abs[:, np.newaxis]
        self._r_vec = self._r_vec / r_vec_abs

    # 去除某个面
    def remove_face(self, face: int | list[int]) -> Polyhedron:
        l_mat = np.delete(self._l_mat, face, 0)
        r_vec = np.delete(self._r_vec, face, 0)

        return self.__class__(l_mat, r_vec)

    # 去除冗余项
    def remove_redundancy(self) -> None:
        # 一条边不可能冗余
        if self.n_faces < 2:
            return

        for i in range(self.n_faces - 1, 0, -1):
            without_row_i = self.remove_face(i)
            s_i = self._r_vec[i] - support_fn(self._l_mat[i, :], without_row_i)

            if s_i >= 0:
                self._l_mat = without_row_i._l_mat
                self._r_vec = without_row_i._r_vec

    # 傅里叶-莫茨金消元法，这里从最后一个元素开始倒着消除，因此使用该方法前应该把需要保留的元素放在最前面
    # 相当于给一个变量左乘矩阵
    # [1 0 0 ... 0 0 ... 0]
    # [0 1 0 ... 0 0 ... 0]
    # [. . . ... . . ... .]
    # [0 0 0 ... 1 0 ... 0]
    #              | --- |
    #                 |
    #                 V
    #              减少的维度
    def reduce_dimension(self, n_dim: int) -> None:
        if n_dim <= 0:
            return

        for _ in range(n_dim):
            pos_a = np.empty((0, self.n_dim - 1))
            pos_b = np.empty(0)
            neg_a = np.empty((0, self.n_dim - 1))
            neg_b = np.empty(0)

            new_a = np.empty((0, self.n_dim - 1))
            new_b = np.empty(0)

            for i in range(self.n_faces):
                a_i_last = self._l_mat[i, -1]
                a_i_others = self._l_mat[i, :-1]
                b_i = self._r_vec[i]

                if a_i_last > 0:
                    pos_a = np.vstack((pos_a, a_i_others / a_i_last))
                    pos_b = np.hstack((pos_b, b_i / a_i_last))
                elif a_i_last == 0:
                    new_a = np.vstack((new_a, a_i_others))
                    new_b = np.hstack((new_b, b_i))
                else:
                    neg_a = np.vstack((neg_a, -a_i_others / a_i_last))
                    neg_b = np.hstack((neg_b, -b_i / a_i_last))

            for i in range(pos_a.shape[0]):
                for j in range(neg_a.shape[0]):
                    new_a = np.vstack((new_a, pos_a[i, :] + neg_a[j, :]))
                    new_b = np.hstack((new_b, pos_b[i] + neg_b[j]))

            self._l_mat = new_a
            self._r_vec = new_b
            self.remove_redundancy()

    # 相当于给一个变量左乘矩阵
    # [1 0 0 ... 0]
    # [0 1 0 ... 0]
    # [. . . ... .]
    # [0 0 0 ... 1]
    # [0 0 0 ... 0] ---
    # [. . . ... .]   | ---> 增加的维度
    # [0 0 0 ... 0] ---
    def extend_dimension(self, n_dim: int) -> None:
        if n_dim <= 0:
            return

        zero_1 = np.zeros((self.n_faces, n_dim))
        zero_2 = np.zeros((n_dim, self.n_dim))
        zero_3 = np.zeros(2 * n_dim)
        eye = np.eye(n_dim)
        self._l_mat = np.block([[self._l_mat, zero_1], [zero_2, eye], [zero_2, -eye]])
        self._r_vec = np.block([self._r_vec, zero_3])

    # 闵可夫斯基和（或平移）
    def __add__(self, other: Polyhedron | NDArray[np.float64]) -> Polyhedron:
        if isinstance(other, Polyhedron):
            h_self_other = np.array(
                [support_fn(self._l_mat[i, :], other) for i in range(self.n_faces)]
            )
            h_other_self = np.array(
                [support_fn(other._l_mat[i, :], self) for i in range(other.n_faces)]
            )

            res_l_mat = np.vstack((self._l_mat, other._l_mat))
            res_r_vec = np.hstack(
                (self._r_vec + h_self_other, other._r_vec + h_other_self)
            )

            res = self.__class__(res_l_mat, res_r_vec)
            res.remove_redundancy()

        elif isinstance(other, np.ndarray):
            res = self.__class__(self._l_mat, self._r_vec + self._l_mat @ other)

        else:
            raise TypeError("The operand must be either a Polyhedron or a numpy array.")

        res.normalization()

        return res

    def __neg__(self) -> Polyhedron:
        return self.__class__(-self._l_mat, self._r_vec)

    # 特别的，这里指庞特里亚金差，即闵可夫斯基和的逆运算
    # 即若 p2 = p1 + p3，则 p3 = p2 - p1，只有当输入为一个点（数组）时该运算等价于 (-p1) + p2
    def __sub__(self, other: Polyhedron | NDArray[np.float64]) -> Polyhedron:
        if isinstance(other, Polyhedron):
            h_self_other = np.array(
                [support_fn(self._l_mat[i, :], other) for i in range(self.n_faces)]
            )

            res = self.__class__(self._l_mat, self._r_vec - h_self_other)
            res.remove_redundancy()

        elif isinstance(other, np.ndarray):
            res = self.__class__(self._l_mat, self._r_vec - self._l_mat @ other)

        else:
            raise TypeError("The operand must be either a Polyhedron or a numpy array.")

        res.normalization()

        return res

    # 多面体坐标变换，map的逆变换
    def map_inv(self, mat: NDArray[np.float64]) -> Polyhedron:
        return self.__class__(self._l_mat @ mat, self._r_vec)

    # 多面体坐标变换，Poly_new = Poly @ mat 意味着 Poly_new 是将 Poly 中的所有点通过 mat 映射后的区域，这一定义是为了方便计算不变集
    def map(self, mat: NDArray[np.float64]) -> Polyhedron:
        u, s, vt = npl.svd(mat)
        rank = s.shape[0]

        res = self.__class__(self._l_mat @ vt.T, self._r_vec)
        res.reduce_dimension(mat.shape[1] - rank)
        res._l_mat = res._l_mat @ np.diag(1 / s)
        res.extend_dimension(mat.shape[0] - rank)
        res._l_mat = res._l_mat @ u.T

        return res

    def __matmul__(self, other: NDArray[np.float64]) -> Polyhedron:
        return self.map(other)

    def __mul__(self, other: float) -> Polyhedron:
        return self.__class__(self._l_mat / other, self._r_vec)

    def __and__(self, other: Polyhedron) -> Polyhedron:
        res = self.__class__(
            np.vstack((self._l_mat, other._l_mat)),
            np.hstack((self._r_vec, other._r_vec)),
        )
        res.remove_redundancy()

        return res

    def __eq__(self, other: Polyhedron) -> bool:
        return self.subset_eq(other) and other.subset_eq(self)

    def get_max_ellipsoid(
        self, p: NDArray[np.float64], center: NDArray[np.float64] | None = None
    ) -> Ellipsoid:
        ellipsoid_center = np.zeros(self.n_dim) if center is None else center

        if not self.contains(ellipsoid_center):
            raise InscribedEllipsoidException

        p_bar = npl.cholesky(p)
        r_vec_bar = self._r_vec + self._l_mat @ ellipsoid_center
        l_mat_bar = self._l_mat @ npl.inv(p_bar).T
        min_center_to_edge_distance = np.min(
            np.abs(r_vec_bar) / npl.norm(l_mat_bar, ord=2, axis=1)
        )

        return Ellipsoid(p, min_center_to_edge_distance**2, center)


def unit_cube(dim: int, side_length: float):
    eye = np.eye(dim)
    return Polyhedron(np.vstack((eye, -eye)), (side_length / 2) * np.ones(2 * dim))
