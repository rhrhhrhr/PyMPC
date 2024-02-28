import numpy as np
import numpy.linalg as npl
import cvxpy as cp
import matplotlib.pyplot as plt
from typing import List


class PolyException(Exception):
    pass


class Polyhedron(object):
    # 用线性不等式组 A @ x <= b 来表示一个多边形
    # n_edges: 边的个数
    # n_dim: 多面体维度
    # l_mat: A
    # r_vec: b

    def __init__(self, l_mat: np.ndarray, r_vec: np.ndarray):
        if l_mat.ndim != 2:
            raise PolyException('The left matrix must be a 2D array!')
        if r_vec.ndim != 1:
            raise PolyException('The right vector must be a 1D array!')
        if l_mat.shape[0] != r_vec.shape[0]:
            raise PolyException('The row number of the left matrix must match the dimension of the right vector!')

        self.__n_edges, self.__n_dim = l_mat.shape
        self.__l_mat = l_mat
        self.__r_vec = r_vec

    @property
    def n_edges(self) -> int:
        return self.__n_edges

    @property
    def n_dim(self) -> int:
        return self.__n_dim

    @property
    def l_mat(self) -> np.ndarray:
        return self.__l_mat

    @property
    def r_vec(self) -> np.ndarray:
        return self.__r_vec

    def __copy__(self, other: 'Polyhedron') -> None:
        self.__n_edges = other.__n_edges
        self.__n_dim = other.__n_dim
        self.__l_mat = other.__l_mat
        self.__r_vec = other.__r_vec

    def __str__(self) -> str:
        return ('====================================================================================================\n'
                'left matrix @ x <= right vec\n'
                '====================================================================================================\n'
                f'left matrix:\n'
                f'{self.__l_mat}\n'
                '----------------------------------------------------------------------------------------------------\n'
                f'right vector:\n'
                f'{self.__r_vec}\n'
                '====================================================================================================')

    # 为优化求解器的约束作接口
    def __call__(self, point: np.ndarray or cp.Variable or cp.Parameter) -> np.ndarray or cp.Variable or cp.Parameter:
        return self.__l_mat @ point - self.__r_vec

    # 判断该点是否为内点
    def is_interior_point(self, point: np.ndarray) -> bool:
        return np.all(self.__l_mat @ point - self.__r_vec <= 0)

    # 判断此多边形是否被包含于另一个多边形
    def belongs_to(self, other: 'Polyhedron') -> bool:
        res = True

        for i in range(other.__n_edges):
            res = res and (support_fun(other.__l_mat[i, :], self) <= other.__r_vec[i])

        return res

    def equals_to(self, other: 'Polyhedron') -> bool:
        return self.belongs_to(other) and other.belongs_to(self)

    # 将不等式组右侧向量归一化，防止系数过大，影响支撑函数（线性规划）求解
    def normalization(self) -> None:
        r_vec_abs = np.where(self.__r_vec == 0, 1, np.abs(self.__r_vec))
        self.__l_mat = self.__l_mat / r_vec_abs[:, np.newaxis]
        self.__r_vec = self.__r_vec / r_vec_abs

    # 去除某个边
    def remove_edge(self, edge: int or list) -> 'Polyhedron':
        l_mat = np.delete(self.__l_mat, edge, 0)
        r_vec = np.delete(self.__r_vec, edge, 0)

        return self.__class__(l_mat, r_vec)

    # 去除冗余项
    def remove_redundant_term(self) -> None:
        if self.__n_edges >= 2:
            i = 0
            while i < self.__n_edges:
                without_row_i = self.remove_edge(i)
                s_i = self.__r_vec[i] - support_fun(self.__l_mat[i, :], without_row_i)

                if s_i >= 0:
                    self.__copy__(without_row_i)
                else:
                    i = i + 1

    # 傅里叶-莫茨金消元法，这里从最后一个元素开始倒着消除，因此使用该方法前应该把需要保留的元素放在最前面
    def fourier_motzkin_elimination(self, n_dim: int) -> None:
        for _ in range(n_dim):
            pos_a = np.empty((0, self.__n_dim - 1))
            pos_b = np.empty(0)
            neg_a = np.empty((0, self.__n_dim - 1))
            neg_b = np.empty(0)

            new_a = np.empty((0, self.__n_dim - 1))
            new_b = np.empty(0)

            for i in range(self.__n_edges):
                a_i_last = self.__l_mat[i, -1]
                a_i_others = self.__l_mat[i, :-1]
                b_i = self.__r_vec[i]

                if a_i_last > 0:
                    pos_a = np.vstack((pos_a, a_i_others / a_i_last))
                    pos_b = np.hstack((pos_b, b_i / a_i_last))
                elif a_i_last == 0:
                    new_a = np.vstack((new_a, a_i_others))
                    new_b = np.hstack((new_b, b_i))
                else:
                    neg_a = np.vstack((neg_a, -a_i_others / a_i_last))
                    neg_b = np.hstack((neg_b, -b_i / a_i_last))

            for i in range(len(pos_a)):
                for j in range(len(neg_a)):
                    new_a = np.vstack((new_a, pos_a[i, :] + neg_a[j, :]))
                    new_b = np.hstack((new_b, pos_b[i] + neg_b[j]))

            self.__init__(new_a, new_b)
            self.remove_redundant_term()

    def extend_dimensions(self, n_dim: int) -> None:
        if n_dim < 0:
            raise PolyException('The extended dimension must be a positive integer!')
        elif n_dim > 0:
            zero_1 = np.zeros((self.__n_edges, n_dim))
            zero_2 = np.zeros((n_dim, self.__n_dim))
            zero_3 = np.zeros(2 * n_dim)
            eye = np.eye(n_dim)
            self.__n_edges = self.__n_edges + 2 * n_dim
            self.__n_dim = self.__n_dim + n_dim
            self.__l_mat = np.block([[self.__l_mat, zero_1], [zero_2, eye], [zero_2, -eye]])
            self.__r_vec = np.block([self.__r_vec, zero_3])

    # 闵可夫斯基和（或平移）
    def __add__(self, other: 'Polyhedron' or np.ndarray) -> 'Polyhedron':
        if isinstance(other, Polyhedron):
            h_self_other = np.zeros(self.__n_edges)
            h_other_self = np.zeros(other.__n_edges)

            for i in range(self.__n_edges):
                h_self_other[i] = support_fun(self.__l_mat[i, :], other)

            for i in range(other.__n_edges):
                h_other_self[i] = support_fun(other.__l_mat[i, :], self)

            res_l_mat = np.vstack((self.__l_mat, other.__l_mat))
            res_r_vec = np.hstack((self.__r_vec + h_self_other, other.__r_vec + h_other_self))

            res = self.__class__(res_l_mat, res_r_vec)
            res.remove_redundant_term()

        else:
            if other.ndim != 1:
                raise PolyException('A polyhedron can only be added by a 1D array!')
            if other.size != self.__n_dim:
                raise PolyException('To add up a polyhedron and an array, their dimensions must match!')

            res = self.__class__(self.__l_mat, self.__r_vec + self.__l_mat @ other)

        res.normalization()

        return res

    def __neg__(self) -> 'Polyhedron':
        return self.__class__(-self.__l_mat, self.__r_vec)

    # 特别的，这里指庞特里亚金差，即闵可夫斯基和的逆运算
    # 即若 p2 = p1 + p3，则 p3 = p2 - p1，只有当输入为一个点（数组）时该运算等价于 (-p1) + p2
    def __sub__(self, other: 'Polyhedron' or np.ndarray) -> 'Polyhedron':
        if isinstance(other, Polyhedron):
            h_self_other = np.zeros(self.__n_edges)

            for i in range(self.__n_edges):
                h_self_other[i] = support_fun(self.__l_mat[i, :], other)

            res = self.__class__(self.__l_mat, self.__r_vec - h_self_other)
            res.remove_redundant_term()
            res.normalization()

        else:
            res = self + (-other)

        return res

    # 多面体坐标变换，Poly_new = Poly @ mat 意味着 Poly 是将 Poly_new 中的所有点通过 mat 映射后的区域，这一定义是为了方便计算不变集
    def __matmul__(self, other: np.ndarray) -> 'Polyhedron':
        if other.ndim != 2:
            raise PolyException('A polyhedron can be only multiplied by a 2D array!')
        if other.shape[0] != self.__n_dim:
            raise PolyException('The column number of the matrix does not match the polyhedron\'s dimension!')

        return self.__class__(self.__l_mat @ other, self.__r_vec)

    # 多面体坐标变换，Poly_new = Poly @ mat 意味着 Poly_new 是将 Poly 中的所有点通过 mat 映射后的区域，这一定义是为了方便计算不变集
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> 'Polyhedron' or NotImplemented:
        if ufunc == np.matmul:
            lhs, rhs = inputs
            row, col = lhs.shape
            u, s, v = npl.svd(lhs)
            rank = s.shape[0]

            res = self.__matmul__(v.T)
            res.fourier_motzkin_elimination(col - rank)
            res = res.__matmul__(np.diag(1 / s))
            res.extend_dimensions(row - rank)
            res = res.__matmul__(u.T)
        elif ufunc == np.add:
            lhs, rhs = inputs
            res = self.__add__(lhs)
        else:
            res = NotImplemented

        return res

    # 多面体的放缩
    def __mul__(self, other: int or float) -> 'Polyhedron':
        if other < 0:
            raise PolyException('A polyhedron can only multiply a positive number!')

        return self.__class__(self.__l_mat / other, self.__r_vec)

    def __rmul__(self, other: int or float) -> 'Polyhedron':
        return self.__mul__(other)

    def __truediv__(self, other: int or float) -> 'Polyhedron':
        return self.__mul__(1 / other)

    # 多面体取交集
    def __and__(self, other: 'Polyhedron') -> 'Polyhedron':
        res = self.__class__(np.vstack((self.__l_mat, other.__l_mat)), np.hstack((self.__r_vec, other.__r_vec)))
        res.remove_redundant_term()

        return res

    # 多边形绘图会有误差，因为采用等高线来画的，增加采样点的个数可以提高画图精度
    def plot(self, ax: plt.Axes, x_lim=None, y_lim=None, default_bound=100, n_points=2000, color='b') -> None:
        if self.__n_dim != 2:
            raise PolyException('Only 2D polyhedron can be plotted!')
        if x_lim is None:
            x_min = -support_fun(np.array([-1, 0]), self)
            x_max = support_fun(np.array([1, 0]), self)
            x_lim = self.get_grid_lim(x_min, x_max, default_bound)

        elif not isinstance(x_lim, list):
            raise PolyException('The parameter \'x_lim\' must be a list!')

        if y_lim is None:
            y_min = -support_fun(np.array([0, -1]), self)
            y_max = support_fun(np.array([0, 1]), self)
            y_lim = self.get_grid_lim(y_min, y_max, default_bound)

        elif not isinstance(y_lim, list):
            raise PolyException('The parameter \'y_lim\' must be a list!')

        x = np.linspace(*x_lim, n_points)
        y = np.linspace(*y_lim, n_points)
        x_grid, y_grid = np.meshgrid(x, y)

        z = np.sum(np.maximum(self.l_mat[:, 0, np.newaxis, np.newaxis] * x_grid
                              + self.l_mat[:, 1, np.newaxis, np.newaxis] * y_grid
                              - self.r_vec[:, np.newaxis, np.newaxis], 0), axis=0)

        ax.contour(x_grid, y_grid, z, levels=0, colors=color)

    @staticmethod
    def get_grid_lim(val_min: int or float, val_max: int or float, default_bound: int or float) -> List[int or float]:
        if val_min == -float('inf') and val_max == float('inf'):
            lim = [-default_bound, default_bound]
        elif val_min == -float('inf') and val_max != float('inf'):
            bound = abs(val_max) * 1.2
            lim = [-bound, bound]
        elif val_min != -float('inf') and val_max == float('inf'):
            bound = abs(val_min) * 1.2
            lim = [-bound, bound]
        else:
            margin = 0.1 * (val_max - val_min)
            lim = [val_min - margin, val_max + margin]

        return lim


class Rn(Polyhedron):
    def __init__(self, dim: int):
        if dim <= 0:
            raise PolyException('The dimension of Rn must be a positive integer!')

        super().__init__(np.zeros((1, dim)), np.zeros(1))

    def __str__(self):
        return f'Vector space: dimension -> {self.n_dim}.'


class UnitCube(Polyhedron):
    def __init__(self, dim: int, side_length: int or float):
        if dim <= 0:
            raise PolyException('The dimension of a cube must be a positive integer!')
        if side_length < 0:
            raise PolyException('The side length of a cube must be non-negative!')

        self.__side_length = side_length
        eye = np.eye(dim)

        super().__init__(np.vstack((eye, -eye)), (side_length / 2) * np.ones(2 * dim))

    @property
    def side_length(self) -> int or float:
        return self.__side_length

    def __str__(self) -> str:
        return f'Unit cube: dimension -> {self.n_dim}, side length -> {self.__side_length}.'


def support_fun(eta: np.ndarray, p: Polyhedron) -> int or float:
    if eta.ndim != 1:
        raise PolyException('The parameter \'eta\' in calculating the support function of a polyhedron must be 1D')
    if eta.size != p.n_dim:
        raise PolyException('The dimension of the parameter \'eta\' must match the dimension of the polyhedron!')

    var = cp.Variable(p.n_dim)
    prob = cp.Problem(cp.Maximize(eta @ var), [cp.NonPos(p.l_mat @ var - p.r_vec)])
    prob.solve(solver=cp.GLPK)

    return prob.value
