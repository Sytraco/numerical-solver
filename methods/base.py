import numpy as np

class NumericalMethod:
    def __init__(self, x, dt) -> None:
        self.x = x
        self.Ts = dt

    def _set_initial_type(self, initial_condition):

        """
        Forces the initial condition type as np.ndarray.
        Parameter :
        -----------
        initial_condition : int, float, list, tuple, ndarray
        Return :
        -----------
        init : ndarray
        """
    
        if not isinstance(initial_condition, np.ndarray):

            if isinstance(initial_condition, list):
                init = np.array(initial_condition)
            else:
                init = np.array([initial_condition])
        else:
            init = initial_condition
        
        return init
    
    def init_compute(self, y0, stop_point):

        # number of points in the time axis
        self.N = self.x.size

        # set the initial condition as np.ndarray
        y0 = self._set_initial_type(initial_condition=y0)

        # get the spatial dimension of the problem, for example : 2D with (x0, y0) --> order = 2
        order = y0.shape[0]

        # set the stopping parameter of the computation loop
        loop = self.N-1 if stop_point is None else (stop_point - 1 if stop_point <= self.N - 1 else print(f"Expected size is out of bounds for index {self.x.size}"))

        # initial axis for the estimated solution
        y = np.zeros((order, self.N)) if stop_point is None else np.zeros((order, loop + 1))

        # applying the initial conditions
        y[:, 0] = y0

        return y, y0, loop, order