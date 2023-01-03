import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error

class Reactor:
    """
    A class to train and simulate a reactor model based of BDsim
    """ 
    def ode_eval(self, tspan, par, u, x0):
        sol = solve_ivp(reactor, t_span=(tspan[0], tspan[-1]), y0=x0, \
        method='LSODA', args=(par, u,), dense_output=False)
        return sol

    def predict(self, time, par, u):
        """
        Simulate reactor states for given time and load variables
        """
        par = np.array(par)
        samples = len(time)
        tspan = time.values
        x0 = np.array([0.0031, 0.4235, 0.1432, 0.4302, 333.5500,])
        
        sol = self.ode_eval(tspan, par, u[:, 0], x0)
        x0 = sol.y[:, -1] # New SS initial guess for current conditions
        
        svR = np.array([])
        for i in range(samples-1):
            svR = np.append(svR, x0)
            sol = self.ode_eval(tspan[i:i+2], par, u[:, i], x0)
            # sol = solve_ivp(reactor, t_span=(tspan[i], tspan[i+1]), y0=x0, \
            # method='LSODA', args=(par, u[:, i],), dense_output=False)
            x0 = sol.y[:, -1]
        svR = np.append(svR, sol.y[:, -1])
        svR = svR.reshape(samples, len(x0))
        return svR
    
    def train(self, time, u, xvals, par0=0.046):
        """
        Kinetic parameter identification
        """
        def _obj(par, time, u, xvals):
            """
            Objective to minimize MSE
            """
            svR = self.predict(time, par, u)
            MSE = mean_squared_error(xvals, svR[:, 3], squared=True)
            return MSE
        solution = least_squares(
            fun=_obj,
            x0=par0,
            method='lm',
            args=(time, u, xvals),
            verbose=False,
            )
        par = solution.x
        print('Reaction rate:', par)
        return par

    def dyn_rates(self, time, u, xvals, par0=0.046):
        samples = len(time)
        tspan = time.values
        x0 = np.array([0.0031, 0.4235, 0.1432, 0.4302, 333.5500,])

        def _obj_dyn(par, tspan, u, x0, xvals):
            """
            Objective to minimize MSE
            """
            sol = self.ode_eval(tspan, par, u, x0)
            y_pred = np.append(x0[3], sol.y[3, -1])
            MSE = mean_squared_error(xvals, y_pred, squared=True)
            return MSE

        rates = np.array([])
        for i in range(1, samples):
            solution = least_squares(
            fun=_obj_dyn,
            x0=par0,
            method='lm',
            args=(tspan[i-1:i+1], u[:, i], x0, xvals[i-1:i+1]),
            verbose=False,
            )
            par = solution.x
            rates = np.append(rates, par)
            sol = self.ode_eval(tspan[i-1:i+1], par, u[:, i], x0)
            x0 = sol.y[:, -1]
        return rates

def reactor(t, x, par, u):
    """
    Reactor model based of BDsim
    """
    #       TG     M     G     E
    M = np.array([0.853, 0.032, 0.092, 0.286,])
    ro = np.array([954.0, 757.0, 1340.0, 844.0,])
    cp = np.array([2110.0, 2785.0, 2556.0, 2146.0,])
    nc = 4
    Mo = M[0,]
    Mm = M[1,]
    xo = np.array([1, 0, 0, 0,])
    xm = np.array([0, 1, 0, 0,])
    VR = 20.
    vmol = M / ro
    cpmol = cp * M
    cpmolo = cpmol[0]
    cpmolm = cpmol[1]

    dHr = -6309. # Fixar este parâmetro

    xR = x[:nc].T
    TR = x[nc]

    No = u[0]/Mo/3600 # kg/h -> mol/s
    Nm = u[1]/Mm/3600 # kg/h -> mol/s
    To = u[2]
    Tm = u[3]

    # mixture amount and properties
    nR = VR / np.sum(vmol * xR) # mol
    cpmolR = np.sum(xR * cpmol) # J/(mol K)

    r = kinetics(par)
    
    rx = np.array([-r, -3*r, r, 3*r,]).reshape(-1,) # [TG, M, G, E]

    # mass balances for the nc components
    dxRdt = (Nm*(xm - xR) + No*(xo - xR) + rx*VR) / nR
    
    # energy balance
    dTRdt = (Nm*cpmolm*(Tm - TR) + No*cpmolo*(To - TR) \
        + VR*np.sum(-dHr * r)) / (nR * cpmolR)
        
    return np.append(dxRdt, dTRdt)

def kinetics(par):
    """
    Kinetic model
    """
    return par
