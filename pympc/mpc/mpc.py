from .base import *


class MPC(MPCBase):
    def __init__(self, a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, pred_horizon: int,
                 state_set: set.Polyhedron, input_set: set.Polyhedron, terminal_set_type='polyhedron', solver=cp.OSQP):
        super().__init__(a, b, q, r, pred_horizon, terminal_set_type, solver)

        if not (self.state_dim == state_set.n_dim):
            raise MPCDimensionException('state set', 'state in controller')
        if not (self.input_dim == input_set.n_dim):
            raise MPCDimensionException('input set', 'input in controller')

        self.__state_set = state_set
        self.__input_set = input_set

    @property
    def state_set(self) -> set.Polyhedron:
        return self.__state_set

    @property
    def input_set(self) -> set.Polyhedron:
        return self.__input_set

    @property
    def initial_constraint(self):
        return self.state_ini - self.real_time_state == 0

    def __call__(self, real_time_state: np.ndarray) -> np.ndarray:
        self.real_time_state = real_time_state
        self.problem.solve(solver=self.solver)

        return self.input_ini.value
