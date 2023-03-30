import numpy as np
from gekko import GEKKO

class Reactor:
    """
    A class to train and simulate a reactor model based of BDsim
    """
    def predict(self, t, u, par):
        m = GEKKO(remote=False)    # create GEKKO model
        
        m.time = t.values # time points
        
        M = np.array([0.853, 0.032, 0.092, 0.286])
        ro = np.array([954.0, 757.0, 1340.0, 844.0])
        cp = np.array([2110.0, 2785.0, 2556.0, 2146.0])
        xo = np.array([1, 0, 0, 0])
        xm = np.array([0, 1, 0, 0])
        vmol = M / ro
        cpmol = cp * M
        
        # create GEKKO constants
        nc = 4
        Mo = 0.853
        Mm = 0.032
        cpmolo = cpmol[0]
        cpmolm = cpmol[1]
        VR = 20
        dHr = -6309
        
        # create GEKKO parameter
        r = m.Param(value=par)
        u0 = m.Param(value=u[0, :])
        u1 = m.Param(value=u[1, :])
        u2 = m.Param(value=u[2, :])
        u3 = m.Param(value=u[3, :])
        
        # create GEKKO variables
        x = np.array([None]*4)
        x[0] = m.Var(0.0031)
        x[1] = m.Var(0.4235)
        x[2] = m.Var(0.1432)
        x[3] = m.Var(0.429)
        T = m.Var(333.5500)
        
        rx = [-r, -3*r, r, 3*r,]

        # create GEKKO equations
        To = u2
        Tm = u3
        No = m.Intermediate(u0/Mo/3600)
        Nm =  m.Intermediate(u1/Mm/3600)
        cpmolR = m.Intermediate(np.sum(x * cpmol))
        nR = m.Intermediate(VR / np.sum(vmol * x))

        m.Equations([nR*x[i].dt() == Nm*(xm[i] - x[i]) + No*(xo[i] - x[i]) + rx[i]*VR for i in range(nc)])
        m.Equation(nR*cpmolR*T.dt() == Nm*cpmolm*(Tm - T) + No*cpmolo*(To - T) + VR*(-dHr)*r)

        # solve ODE
        m.options.SOLVER= 1
        m.options.IMODE = 7 # Sequential Simulation Mode
        m.solve(disp=False)
        
        print('Simulation completed!')
        return np.concatenate((np.array(x.tolist()), np.array(T).reshape(1, -1))).T

    def train(self, t, y, u, par0=0.046, dynamic=False):
        m = GEKKO(remote=False)    # create GEKKO model

        m.time = t # time points

        M = np.array([0.853, 0.032, 0.092, 0.286])
        ro = np.array([954.0, 757.0, 1340.0, 844.0])
        cp = np.array([2110.0, 2785.0, 2556.0, 2146.0])
        xo = np.array([1, 0, 0, 0])
        xm = np.array([0, 1, 0, 0])
        vmol = M / ro
        cpmol = cp * M

        # create GEKKO constants
        nc = 4
        Mo = 0.853
        Mm = 0.032
        cpmolo = cpmol[0]
        cpmolm = cpmol[1]
        VR = 20
        dHr = -6309

        # create GEKKO parameter
        if dynamic:
            r = m.MV(value=par0)
        else:
            r = m.FV(value=par0)
        
        u0 = m.Param(value=u[0, :])
        u1 = m.Param(value=u[1, :])
        u2 = m.Param(value=u[2, :])
        u3 = m.Param(value=u[3, :])

        # create GEKKO variables
        x = np.array([None]*4)
        x[0] = m.Var(0.0031)
        x[1] = m.Var(0.4235)
        x[2] = m.Var(0.1432)
        x[3] = m.Var(0.429)
        ymeas = m.Param(y.values)
        T = m.Var(333.5500)

        rx = [-r, -3*r, r, 3*r,]

        # create GEKKO equations
        To = u2
        Tm = u3
        No = m.Intermediate(u0/Mo/3600)
        Nm =  m.Intermediate(u1/Mm/3600)
        cpmolR = m.Intermediate(np.sum(x * cpmol))
        nR = m.Intermediate(VR / np.sum(vmol * x))

        m.Equations([nR*x[i].dt() == Nm*(xm[i] - x[i]) + No*(xo[i] - x[i]) + rx[i]*VR for i in range(nc)])
        m.Equation(nR*cpmolR*T.dt() == Nm*cpmolm*(Tm - T) + No*cpmolo*(To - T) + VR*(-dHr)*r)

        # Minimize
        m.Minimize((x[3] - ymeas)**2)

        # solve ODE
        r.STATUS = 1
        m.options.IMODE = 2 # Regression mode
        m.options.NODES = 3 # collocation nodes
        
        m.solve(disp=False)
        
        print('Parameter identification: done!')
        if dynamic == True:
            r = np.array(r)
        else:
            r = [r[0]]
        
        return r
