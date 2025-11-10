class NumericalSchemeParams(object):
    """General numerical scheme parameters.

    Attributes:
        T: (float) Length of time interval
        L: (float) Length of space interval
        nt: (int) Number of time steps, including initial condition
        nx: (int) Number of points in space, start and end count together as one point
        dt: (float) Time resolution
        dx: (float) Spatial resolution
        dtdx: (float) dt / dx
        x_start: (float) Left end of space interval, default is 0
        x_end: (float) Right end of space interval
        t_start: (float) Initial time, default is 0
        t_end: (float) Final time 
    """

    def __init__(self, T, L, nt, nx, *args, x_start=0, t_start=0, **kwargs):
        """Initialize scheme parameters.

        Args:
            T: (float) Length of time interval
            L: (float) Length of space interval
            nt: (int) Number of time steps, not including initial condition
            nx: (int) Number of points in space, start and end count together as one point
            nu: (float) Viscosity coefficient
            x_start: (float) Left end of space interval, default is 0
            t_start: (float) Initial time, default is 0
        """
        self.T = float(T)
        self.L = float(L)
        self.nt = nt
        self.nx = nx
        self.dt = self.T / nt
        self.dx = self.L / nx
        self.dtdx = self.dt / self.dx
        self.x_start = x_start
        self.t_start = t_start
        self.x_end = self.x_start + self.L
        self.t_end = self.t_start + self.T


class ViscousParams(NumericalSchemeParams):
    """Parameters for numerical schemes solving equations with a viscosity term.

    Extends the NumericalSchemeParams class.

    Additional attributes:
        nu: (float) Viscosity coefficient.
        d: (float) Dimensionless viscosity coefficient = nu * dt / dx^2.
    """

    def __init__(self, T, L, nt, nx, nu, *args, **kwargs):
        super().__init__(T, L, nt, nx, *args, **kwargs)
        self.nu = float(nu)
        self.d = self.nu * self.dt / (self.dx ** 2)


class AdvectionDiffusionParams(ViscousParams):
    """Parmeters for numerical schemes solving the linear advection-diffusion equation.

    Extends the ViscousParams class.
    
    Additional attribute:
        v: (float) Constant velocity for advection.
    """

    def __init__(self, T, L, nt, nx, nu, v, *args, **kwargs):
        super().__init__(T, L, nt, nx, nu, *args, **kwargs)
        self.v = v