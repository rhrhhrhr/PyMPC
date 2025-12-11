import numpy as np
import cvxpy as cp
from typing import overload, Protocol
from numpy.typing import NDArray


class Set(Protocol):
    # 集合的维度
    @property
    def n_dim(self) -> int: ...

    @overload
    def contains(self, point: NDArray[np.float64]) -> bool: ...

    @overload
    def contains(self, point: cp.Expression) -> cp.Constraint: ...

    # 判断是否为内点，同时也可以作为cvxpy求解器接口
    def contains(
        self, point: NDArray[np.float64] | cp.Expression
    ) -> bool | cp.Constraint: ...


def support_fn(eta: NDArray[np.float64], s: Set) -> float:
    var = cp.Variable(s.n_dim)
    prob = cp.Problem(cp.Maximize(eta @ var), [s.contains(var)])
    prob.solve(solver=cp.GLPK)

    return prob.value  # type: ignore
