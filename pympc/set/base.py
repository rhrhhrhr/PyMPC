import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import abc


class SetError(Exception):
    pass


class SetTypeError(SetError):
    def __init__(self, name: str, set_type: str, tp: str):
        message = 'The type of the ' + name + ' of ' + set_type + ' must be ' + tp + '!'
        super().__init__(message)


class SetDimensionError(SetError):
    def __init__(self, *args: str):
        message = 'The dimensions of ' + ', '.join(args) + 'do not match!'
        super().__init__(message)


class SetCalculationError(SetError):
    def __init__(self, set_type: str, operation: str, other: str):
        message = 'A ' + set_type + ' can only be ' + operation + ' by a ' + other + '!'
        super().__init__(message)


class SetNotImplementedError(SetError):
    def __init__(self, function: str, set_type: str):
        message = 'The function ' + function + ' of ' + set_type + ' has not been implemented yet!'
        super().__init__(message)


class SetPlotError(SetError):
    def __init__(self):
        super().__init__('Only 2D set can be plotted!')


class SetBase(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def n_dim(self) -> int:
        ...

    # 判断是否为内点，同时也可以作为cvxpy求解器接口
    @abc.abstractmethod
    def contains(self, point: np.ndarray or cp.Expression) -> bool or cp.Constraint:
        ...

    # 画图（仅实现二维画图）
    @abc.abstractmethod
    def plot(self, ax: plt.Axes, n_points=2000, color='b') -> None:
        ...

    # 闵可夫斯基和（或平移）
    @abc.abstractmethod
    def __add__(self, other: 'SetBase' or np.ndarray) -> 'SetBase':
        ...

    # 特别的，这里指庞特里亚金差，即闵可夫斯基和的逆运算
    # 即若 p2 = p1 + p3，则 p3 = p2 - p1，只有当输入为一个点（数组）时该运算等价于 (-p1) + p2
    @abc.abstractmethod
    def __sub__(self, other: 'SetBase' or np.ndarray) -> 'SetBase':
        ...

    # 多面体坐标变换，Set_new = Set @ mat 意味着 Set 是将 Set_new 中的所有点通过 mat 映射后的区域，这一定义是为了方便计算不变集
    @abc.abstractmethod
    def __matmul__(self, other: np.ndarray) -> 'SetBase':
        ...

    # 多面体坐标变换，Set_new = Set @ mat 意味着 Set_new 是将 Poly 中的所有点通过 mat 映射后的区域，这一定义是为了方便计算不变集
    @abc.abstractmethod
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> 'SetBase' or NotImplemented:
        ...

    # 集合的放缩
    @abc.abstractmethod
    def __mul__(self, other: int or float) -> 'SetBase':
        ...

    def __rmul__(self, other: int or float) -> 'SetBase':
        return self.__mul__(other)

    def __truediv__(self, other: int or float) -> 'SetBase':
        return self.__mul__(1 / other)

    # 多面体取交集
    @abc.abstractmethod
    def __and__(self, other: 'SetBase') -> 'SetBase':
        ...


def support_fun(eta: np.ndarray, s: SetBase) -> int or float:
    if eta.ndim != 1:
        raise SetTypeError('input \'eta\'', 'support function', '1D array')
    if eta.size != s.n_dim:
        raise SetDimensionError('\'eta\'', '\'polyhedron\'')

    var = cp.Variable(s.n_dim)
    prob = cp.Problem(cp.Maximize(eta @ var), [s.contains(var)])
    prob.solve(solver=cp.GLPK)

    return prob.value
