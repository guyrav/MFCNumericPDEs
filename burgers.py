from itertools import product
from typing import Callable
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


class Params(object):
    """Numerical scheme parameters.

    Attributes:
        T: (float) Length of time interval
        L: (float) Length of space interval
        nt: (int) Number of time steps, including initial condition
        nx: (int) Number of points in space, start and end count together as one point
        dt: (float) Time resolution
        dx: (float) Spatial resolution
        x_start: (float) Left end of space interval, default is 0
        x_end: (float) Right end of space interval
        t_start: (float) Initial time, default is 0
        t_end: (float) Final time
        nu: (float) Viscosity coefficient
        d: (float) Dimensionless viscosity coefficient
        dtdx: (float) dt / dx
    """
    def __init__(self, T, L, nt, nx, nu, x_start=0, t_start=0):
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
        self.nu = float(nu)
        self.d = self.nu * self.dt / (self.dx ** 2)
        self.dtdx = self.dt / self.dx
        self.x_start = x_start
        self.t_start = t_start
        self.x_end = self.x_start + self.L
        self.t_end = self.t_start + self.T


def init(params: Params,
         initial_condition: Callable[[np.ndarray[float]], np.ndarray[float]],
         record_all=False):
    """Create space and time coordinates and initial state.

    If record_all is set to True, also create a table to store all states.

    Args:
        params: (Params) Numerical scheme parameters
        initial_condition: (Callable) Function that takes in array of points in space and returns initial state
        record_all: (bool) Flag indicating the need to save all states over time
    """
    x = np.linspace(params.x_start, params.x_end, params.nx + 1, endpoint=True)
    t = np.linspace(params.t_start, params.t_end, params.nt + 1, endpoint=True)
    u = initial_condition(x)

    if record_all:
        all_u = np.zeros((params.nt + 1, params.nx + 1), dtype=float)
        all_u[0, :] = u[:]
        return x, t, u, all_u
    else:
        return x, t, u


def gaussian(mu: float, sigma: float) -> Callable:
    """Return a function that applies a gaussian with mean mu and variance sigma^2."""
    def ret(x: np.ndarray[float]) -> np.ndarray[float]:
        return 1. / np.sqrt(2 * np.pi * sigma) * np.exp(- ((x - mu) / sigma)**2 / 2)
    return ret


def near_constant(u_0: float, epsilon: float, perturbation: Callable):
    """Return a function that applies mean velocity plus a perturbation scaled by epsilon."""
    def ret(x: np.ndarray[float]) -> np.ndarray[float]:
        return u_0 * (1 + epsilon * perturbation(x))
    return ret


# def initiate(x,params,n=1):
#     """
#     Parameters
#     ----------
#     x : (ndarray) Spatial grid points.
#     params : (dict) Dictionary of simulation parameters containing "L", the domain length.
#     n : (int, optional) Identifier for the type of initial condition to allow new initial conditions.
#
#     Returns
#     -------
#     velocity : (ndarray) Initial velocity u(x, t=0).
#     """
#     if n==1:
#         L=params["L"]
#         x_centered = x - (L/2)
#         velocity = np.exp(-x_centered**2/2)
#
#     return(velocity)


def FTCS(u_0: np.ndarray[float], scheme_params: Params, u_all=None):
    """Solve the 1D Burgers’ equation numerically using the FTCS scheme.

    Parameters
    ----------
    u0 : (ndarray) Initial condition (velocity at t=0).
    scheme_params : (Params) Parameters for the numerical scheme
    u_all : (ndarray, optional) Either None or an array with shape (nt+1, nx+1).
                                If not None, the full time evolution is stored in it (default None).

    Returns
    -------
    u : (ndarray) Final velocity field.
    """
    u = np.zeros(u_0.shape[0] - 1, dtype=float)
    u[:] = u_0[:-1]
    c = scheme_params.dtdx
    d = scheme_params.d
    store_u = (u_all is not None)

    if store_u:
        u_all[0] = u_0

    for n in range(scheme_params.nt):
        shifted_right = np.roll(u, -1)
        shifted_left = np.roll(u, 1)
        u -= (c * 0.5 * u * (shifted_right - shifted_left) - d * (shifted_right - 2 * u + shifted_left))

        if store_u:
            u_all[n + 1] = np.pad(u, [(0, 1)], mode='wrap')
        
    return u


def plot_solution(t: np.ndarray[float], x: np.ndarray[float], u: np.ndarray[float], params: Params):
    """
    Animate and plot the evolution of the solution over time.

    Parameters
    ----------
    t : (ndarray) Time labels.
    x : (ndarray) Space labels.
    u : (ndarray) Velocity field at all time steps.
    params : (Params) Numerical scheme parameters.
    """
    plt.plot(x, u[0], 'k', label='Initial Condition')
    plt.legend(loc='best')
    plt.ylabel('velocity')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([-0.2,1.2])
    plt.pause(0.5)

    for t_n, u_n in zip(t, u):
        plt.cla()
        plt.plot(x, u_n, 'b', label=f'Time {t_n:.2f}')
        plt.legend(loc='best')
        plt.ylabel('velocity')
        plt.xlabel('x')
        plt.ylim([-0.2, 1.2])
        plt.pause(0.05)
    
    plt.show()
    

def plot_scaled_mass(u0,u_all, dt, dx, nt, nx, T, L, nu):
    """
    Plot the normalized mass variation of the system over time.
    Computes mass as:
        M(t) = (M0 - Σ u(x, t) * dx) / M0

    Parameters
    ----------
    u0 : (ndarray) Initial velocity field.
    u_all : (ndarray) Velocity field at all time steps (from FTCS with store_u=True).
    dt, dx, nt, nx, T, L, nu : (float or int) Parameters.
    """
    mass = np.zeros(nt+1)
    time = np.linspace(0.0,T,nt+1)
    mass0 = dx*sum(u0)

    for n, u in enumerate(u_all):
        mass[n] = (mass0 - dx*sum(u))/mass0

    plt.cla()
    plt.plot(time,mass,'k',marker='o', markersize=1, label="mass evolution")
    plt.legend(loc='best')
    plt.ylabel('mass')
    plt.xlabel('t')
    plt.ylim([-0.01, 0.01])
    plt.show()


def total_variation_periodic(u: np.ndarray[float]) -> float:
    return np.sum(np.abs(u - np.roll(u, 1)))


def plot_spatial_variation(dt, dx, nt, nx, T, L, nu):
    """
    Compare and displays spatial variability evolution for different spatial resolutions and viscosity values.
    Variability is computed as V(t) = (V0 - Σ |u(x, t)- u(x-1,t)|) / V0, with V0 = Σ |u0(x, t)- u0(x-1,t)|

    Parameters
    ----------
    dt, dx, nt, nx, T, L, nu : (float or int) Parameters.

    """
    dx_values = [dt/2, dt, 1.5*dt, 2*dt, 10*dt]
    nx_values = [40*nx, 20*nx, 15*nx, 10*nx, 5*nx]
    nu_values = [nu, nu*2, nu*10]
    time = np.linspace(0.0,T,nt+1)

    for viscosity in nu_values:
        for  i in range(len(dx_values)):
            x,p = make_params(dt, dx_values[i], nt, nx_values[i], viscosity)
            u0 = initiate(x,p,1)
            u_all = FTCS (x, u0, **p, store_u=True)

            #spatial variability of initial condition
            shifted_left = np.roll(u0, 1)
            diff = abs(u0 - shifted_left)
            variability0 = sum(diff)
            variability_all = []

            for u in u_all:
                variability = total_variation_periodic(u)
                #variability_all.append(variability)
                variability_all.append((variability - variability0)/variability0)

            plt.plot(time,variability_all,marker='o', markersize=1, label=f'dx= {dx_values[i]:.2f}, nu = {viscosity:.2f}')
            plt.legend(loc='best')
            plt.title(f'Spatial absolute variation over time, for dt = {dt:.2f} ')
            plt.ylabel('Spatial cumulated differences')
            plt.xlabel('t')
            plt.xlim([0,L])
            plt.ylim([-4, 10])
            plt.pause(0.5)
    plt.show()


def plot_conditions_stability(dt, dx, nt, nx, T, L, nu):
    """
    Explore FTCS stability conditions by varying dt and nu for fixed dx values.
    Marks parameter pairs (dt, nu) that cause divergence: to the left of the curve, the numerical scheme is stable.

    Parameters
    ----------
    dt, dx, nt, nx, T, L, nu : float or int
        Base simulation parameters.
    """
    dt_values = np.linspace(0.01, 1.5, 100)
    nu_values = np.linspace(0.01, 1.5, 100)
    dx_values = [0.1, 0.3, 0.6, 0.9]
    colours = ['lightblue', 'blue', 'darkblue', 'k']

    threshold = 1.0  # divergence criterion

    for n, resolution in enumerate(dx_values):
        mark = True
        for viscosity in nu_values:
            for step in dt_values:
                x, params = make_params(step, resolution, nt, nx, viscosity)
                u0 = initiate(x, params, 1)
                u_all = FTCS(x, u0, **params, store_u=True)

                # compute initial variability
                shifted_left = np.roll(u0, 1)
                diff = abs(u0 - shifted_left)
                variability0 = np.sum(diff)

                # check variability growth over time
                diverged = False
                for u in u_all:
                    shifted_left = np.roll(u, 1)
                    diff = abs(u - shifted_left)
                    variability = (np.sum(diff) - variability0) / variability0

                    if variability > threshold or np.any(np.isnan(u)):
                        diverged = True
                        break

                # plot instability boundary point
                if diverged:
                    plt.plot(step, viscosity, color=colours[n],
                             marker='o', markersize=2,
                             label=f'dx = {resolution:.2f}' if mark else "" )
                    mark=False
                    break  # move to next viscosity

    plt.xlabel('dt (time step)')
    plt.ylabel('ν (viscosity)')
    plt.title('FTCS Stability Map — Divergent Regions')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.legend(title='Spatial step dx')
    plt.grid(True, linestyle=':')
    plt.show()


def plot_stability_heatmap(dt, dx, nt, nx, T, L, nu):
    """
    Create heatmaps showing FTCS stability (stable/unstable) regions
    for different spatial resolutions (dx values).

    Parameters
    ----------
    dt, dx, nt, nx, T, L, nu : float or int
        Base simulation parameters.
    """
    # Parameter grids
    dt_values = np.linspace(0.05, 1.0, 40)
    nu_values = np.linspace(0.05, 1.0, 40)
    dx_values = [0.1, 0.4, 0.8]

    threshold = 1.0  # variability threshold for divergence

    # Prepare figure
    fig, axes = plt.subplots(1, len(dx_values), figsize=(16, 5), sharey=True)

    for idx, resolution in enumerate(dx_values):
        stability_map = np.zeros((len(nu_values), len(dt_values)))  # 0 = stable, 1 = diverged

        for i, viscosity in enumerate(nu_values):
            for j, step in enumerate(dt_values):
                try:
                    x, params = make_params(step, resolution, nt, nx, viscosity)
                    u0 = initiate(x, params, 1)
                    u_all = FTCS(x, u0, **params, store_u=True)

                    shifted_left = np.roll(u0, 1)
                    diff = abs(u0 - shifted_left)
                    variability0 = np.sum(diff)

                    diverged = False
                    for u in u_all:
                        shifted_left = np.roll(u, 1)
                        diff = abs(u - shifted_left)
                        variability = (np.sum(diff) - variability0) / variability0
                        if variability > threshold or np.any(np.isnan(u)):
                            diverged = True
                            break

                    stability_map[i, j] = 1 if diverged else 0

                except Exception:
                    stability_map[i, j] = np.nan  # handle any numerical crash safely

        # Plot the heatmap for this dx
        ax = axes[idx]
        im = ax.imshow(stability_map, origin='lower', cmap='RdYlBu_r',
                       extent=[dt_values.min(), dt_values.max(),
                               nu_values.min(), nu_values.max()],
                       aspect='auto')
        ax.set_title(f'dx = {resolution:.2f}')
        ax.set_xlabel('dt (time step)')
        if idx == 0:
            ax.set_ylabel('ν (viscosity)')

        # Add colorbar for first plot
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('1 = Diverged, 0 = Stable')

    plt.suptitle('FTCS Stability Heatmaps for Different Spatial Steps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_linearized_stability(c, d, v_max):
    """Plot a 2D colormap of the final total variation for a near-linear Burger's equation.

    Adapted from ChatGPT.

    Args:
        c: (ndarray) Values of the Courant number (dimensionless mean velocity)
        d: (ndarray) Values of the dimensionless viscosity coefficient
        v_max: Values of the total variation of the velocity at final time
    """
    c, d, v_max = np.array(c), np.array(d), np.array(v_max)

    # grid definition
    ci = np.linspace(c.min(), c.max(), 300)
    di = np.linspace(d.min(), d.max(), 300)
    Ci, Di = np.meshgrid(ci, di)

    # interpolate to grid
    Vi = griddata((c, d), v_max, (Ci, Di), method='linear')

    # plot
    plt.figure()
    plt.pcolormesh(Ci, Di, Vi, shading='auto', cmap='viridis')
    plt.plot(ci, ci**2 / 2, color='red', linewidth=2)
    plt.plot(ci, 0.5 * np.ones_like(ci), color='red', linewidth=2)
    plt.colorbar(label='Relative Total Variation at t=1 (log scale)')  #  negative values shown as 0
    plt.xlim(ci.min(), ci.max())
    plt.ylim(di.min(), di.max())
    plt.xlabel('Courant number (u_mean * dt / dx)')
    plt.ylabel('Dimensionless viscosity coefficient (nu * dt / dx^2)')
    plt.show()


def run_single_variation_experiment(T, L, nt, u_mean, initial_condition, c, d):
    nx = int(np.ceil(c * L * nt / (L * u_mean)))
    nu = d * L**2 * nt / (T * nx**2)
    params = Params(T, L, nt, nx, nu)

    c_actual = u_mean * params.dtdx
    d_actual = params.d

    x, t, u_0 = init(params, initial_condition, record_all=False)
    v_0 = total_variation_periodic(u_0)

    u = FTCS(u_0, params)
    v = total_variation_periodic(u)
    v_relative = (v - v_0) / v_0

    return c_actual, d_actual, v_relative


def linearized_stability_experiment(T, L, nt, u_mean, initial_condition):
    c = np.linspace(0.1, 1.5, 40)
    d = np.linspace(0.1, 0.7, 40)
    # v = np.zeros((c.shape[0], d.shape[0]), dtype=float)

    # c = u_mean * dt / dx = u_mean * T * nx / (L * nt)
    # d = nu * dt / dx^2 = nu * T * nx^2 / (L^2 * nt)
    # c / T * L * nt / u_mean = nx
    # d / T * L^2 * nt / nx^2 = nu

    cs = []
    ds = []
    log_vs = []

    for i, j in product(np.arange(c.shape[0]), np.arange(d.shape[0])):
        c_actual, d_actual, v_relative = run_single_variation_experiment(T, L, nt, u_mean, initial_condition, c[i], d[j])

        cs.append(c_actual)
        ds.append(d_actual)
        log_vs.append(np.log(max(v_relative, np.exp(-2))))

    plot_linearized_stability(cs, ds, log_vs)


def reverse_step(start, end):
    def ret(x):
        result = np.zeros_like(x)
        mask = np.logical_and(x > start, x < end)
        line = 1 - (x - start) / (end - start)
        result[mask] = line[mask]
        return result
    return ret


def real_solution_tk(params: Params, k: np.ndarray, u_0: np.ndarray, u_mean: float):
    return u_0 * np.exp(- (params.nu * k**2 + 1j * u_mean * k) * params.T)


def run_single_accuracy_experiment(params: Params, u_mean: float, initial_condition: Callable):
    x, t, u_0 = init(params, initial_condition, record_all=False)

    u_0_k = np.fft.fft(u_0[:-1])  # remove redundant endpoint
    k = 2 * np.pi / params.L * np.fft.fftfreq(params.nx)
    real_u_k = real_solution_tk(params, k, u_0_k, u_mean)
    real_u = np.fft.ifft(real_u_k)

    u = FTCS(u_0, params)
    scheme_u_k = np.fft.fft(u)

    return np.sqrt(np.mean(np.abs(u - real_u) ** 2))
    # return np.sqrt(np.sum(np.abs(scheme_u_k - real_u_k) ** 2)) / params.nx


def plot_accuracy(nxs, alpha, errors):
    log_nx = np.log(nxs)
    log_errors = np.log(errors)

    plt.figure()
    plt.scatter(log_nx, log_errors, color='black')
    plt.xlabel('log(nx)')
    plt.ylabel('log(MSE)')
    plt.title(f"Error at final time T (with nt = {alpha} * nx^2)")

    p = np.polyfit(log_nx, log_errors, 1)
    plt.plot(log_nx, np.polyval(p, log_nx), '--', color='red',
             label=f"log(MSE) = {p[0]:.1f}*log(nx) + {p[1]:.1f}")
    plt.legend()
    plt.show()


def accuracy_experiment(T, L, nu, u_mean, initial_condition):
    nx_min_log = 3
    num_points = 7
    base = 2
    nxs = np.logspace(nx_min_log, nx_min_log + num_points - 1, num_points, base=base, dtype=int)
    alpha = int(np.ceil(T * nu / L**2))
    nts = alpha * nxs**2
    errors = np.zeros(num_points, dtype=float)

    for i, (nx, nt) in enumerate(zip(nxs, nts)):
        params = Params(T, L, nt, nx, nu)
        errors[i] = run_single_accuracy_experiment(params, u_mean, initial_condition)

    plot_accuracy(nxs, alpha, errors)


def main():
    # Stability experiment
    T = L = 1
    nt = 12
    # u_mean = 0.2
    # epsilon = 0.05
    # initial_condition = near_constant(u_mean, epsilon, gaussian(L/2, 1))
    # initial_condition = near_constant(u_mean, epsilon, reverse_step(1./3, 2./3))
    # linearized_stability_experiment(T, L,nt, u_mean, initial_condition)

    # Accuracy experiment
    T = L = 1
    u_mean = 0.5
    epsilon = 0.05
    nu = 0.1
    initial_condition = near_constant(u_mean, epsilon, gaussian(L/2, 1))
    accuracy_experiment(T, L, nu, u_mean, initial_condition)

    # x, t, u_0, u = init(params, initial_condition, True)
    #
    # u_final = FTCS(u_0, params, u)

    #plot_scaled_mass(u0,u_all,**p)
    # plot_solution(t, x, u, params)
    #plot_spatial_variation(**p)
    # plot_conditions_stability(**p)
    #plot_stability_heatmap(**p)


if __name__ == "__main__":
    main()

