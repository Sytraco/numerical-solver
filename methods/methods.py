import numpy as np

from base import NumericalMethod


class RungeKutta(NumericalMethod):
    
    def __init__(self, x, dx, order):

        """
        x  : variable of the equation
        dx : step/sampling value
        order  : q order of the RKq method
        """
        
        super().__init__(x=x, dx=dx)
        self.q = order
        self.set_weightcoef()

    def set_weightcoef(self):
        ''' Set the 3 weight coeficients for different order methods of RK '''

        coefA = lambda coef: np.roll(np.diag(coef), -1)

        # order 2
        if self.q == 2:
            coef1 = [0, 1]
            self.A = coefA(coeficient=coef1)
            self.B = np.array(coef1)
            self.C = np.ones(2) / 2

        # order 3
        elif self.order == 3:
            coef1 = [0, 1/2, 1]
            self.A = coefA(coeficient=coef1)
            self.A[2, 0] = -1
            self.B = np.array([1/6, 2/3, 1/6])
            self.C = np.array([0, 1/2, 1])
        
        # order 4
        elif self.q == 4:
            coef1 = [0, 1/2, 1/2, 1]
            self.A = coefA(coeficient=coef1)
            self.B = np.array([1/6, 1/3, 1/3, 1/6])
            self.C = np.array([0, 1/2, 1/2, 1])
            
            
    # General scheme for RK order q
    def compute(self, ODE, y0, end=None):

        """
        ODE : ordinary differential equation
        y0  : initial conditions
        end : ending point of compution (default None)
        """
        
        y_rk, y0, loop, order = self.init_compute(y0=y0, stop_point=end)
        
        for xi, xn in zip(range(loop), self.x):
            
            K = np.zeros((order, self.q+1))

            for i in range(1, self.q+1):
                
                K[:, i] = ODE(
                    ydy = y_rk[:, xi] + self.dx * (self.A[i-1, i-2] * K[:, i-1]),
                    x_axis  = xn + self.C[i-1] * self.dx
                    )

            K = [np.delete(arr=K, obj=0) if K.size == K.shape[0] else np.delete(arr=K, obj=0, axis=1)][0]

            y_rk[:, xi+1] = y_rk[:, xi] + self.dx * np.sum(self.B * K, axis=1)
        
        return y_rk