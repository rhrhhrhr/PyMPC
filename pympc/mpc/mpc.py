from .base import *


class MPC(MPCBase):
    def __init__(self, a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, pred_horizon: int,
                 state_set: set.Polyhedron, input_set: set.Polyhedron, terminal_set_type='polyhedron', solver=cp.OSQP):
        super().__init__(a, b, q, r, pred_horizon, terminal_set_type, solver)

        if not (self.state_dim == state_set.n_dim):
            raise MPCDimensionError('state set and state in controller')
        if not (self.input_dim == input_set.n_dim):
            raise MPCDimensionError('input set and input in controller')

        self.__state_set = state_set
        self.__input_set = input_set

        self.__terminal_set = self.cal_terminal_set()

        self.__problem = self.construct_problem()

    @MPCBase.terminal_set_type.setter
    def terminal_set_type(self, value: str) -> None:
        if value not in ['zero', 'ellipsoid', 'polyhedron']:
            raise MPCTerminalSetTypeError()

        if self.terminal_set_type != value:
            MPCBase.terminal_set_type.fset(self, value)
            self.__terminal_set = self.cal_terminal_set()
            self.__problem = self.construct_problem()

    @MPCBase.pred_horizon.setter
    def pred_horizon(self, value: int) -> None:
        MPCBase.pred_horizon.fset(self, value)
        self.__problem = self.construct_problem()

    @property
    def state_set(self) -> set.Polyhedron:
        return self.__state_set

    @property
    def input_set(self) -> set.Polyhedron:
        return self.__input_set

    @property
    def terminal_set(self) -> set.Polyhedron:
        return self.__terminal_set

    @property
    def initial_constraint(self):
        return self.state_ini - self.real_time_state == 0

    @property
    def problem(self) -> cp.Problem:
        return self.__problem

    @property
    def feasible_set(self) -> set.Polyhedron:
        return self.cal_feasible_set()

    def __call__(self, real_time_state: np.ndarray) -> np.ndarray:
        self.real_time_state = real_time_state
        self.problem.solve(solver=self.solver)

        return self.input_ini.value
