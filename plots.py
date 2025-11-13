from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy.interpolate import griddata


def plot_evolution(t: np.ndarray[float], x: np.ndarray[float], u: np.ndarray[float], title: str):
    """Animate and plot the evolution of the solution over time.

    Args:
        t : (np.ndarray[float]) Time labels.
        x : (np.ndarray[float]) Space labels.
        u : (np.ndarray[float]) Velocity field at all time steps.
        title : (str) Plot title.
    """
    ylims = get_ylims(u[0])

    plt.plot(x, u[0], 'k', label='Initial Condition')
    plt.title(title)
    plt.legend(loc='best')
    plt.ylabel('velocity')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim(ylims)
    plt.pause(0.5)

    for t_n, u_n in zip(t, u):
        plt.cla()
        plt.plot(x, u_n, 'b', label=f'Time {t_n:.2f}')
        plt.title(title)
        plt.legend(loc='best')
        plt.ylabel('velocity')
        plt.xlabel('x')
        plt.ylim(ylims)
        plt.pause(0.05)

    plt.show()


def plot_initial_condition(x: np.ndarray[float], u: np.ndarray[float], title: str):
    """
    Plot the initial condition of the system.

    Args:
        x : (np.ndarray[float]) Space labels.
        u : (np.ndarray[float]) Velocity field at time t_start.
        title : (str) Plot title.
    """
    ylims = ylims = get_ylims(u)

    plt.plot(x, u, 'k')
    plt.title(title)
    plt.ylabel('velocity')
    plt.xlabel('x')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim(ylims)

    plt.show()


def plot_evolution_comparison(t: np.ndarray[float], x: np.ndarray[float],
                              u_a: np.ndarray[float], u_b: np.ndarray[float],
                              title: str, label_a: str, label_b: str):
    """Animate and plot the evolution of both solutions a and b over time.
    
    Args:
       t : (np.ndarray[float]) Time labels.
       x : (np.ndarray[float]) Space labels.
       u_a : (np.ndarray[float]) Velocity field a at all time steps.
       u_b : (np.ndarray[float]) Velocity field b at all time steps.
       title : (str) Plot title.
       label_a : (str) Label for solution a
       label_b : (str) Label for solution b
    """
    ylims = get_ylims(u_a[0])

    plt.plot(x, u_a[0], 'k', label='Initial Condition')
    plt.title(title)
    plt.legend(loc='best')
    plt.ylabel('velocity')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim(ylims)
    plt.pause(0.5)

    for t_n, u_a_n, u_b_n in zip(t, u_a, u_b):
        plt.cla()
        plt.plot(x, u_a_n, 'b', label=f'{label_a}, Time {t_n:.2f}')
        plt.plot(x, u_b_n, 'r', label=f'{label_b}, Time {t_n:.2f}')
        plt.title(title)
        plt.legend(loc='best')
        plt.ylabel('velocity')
        plt.xlabel('x')
        plt.ylim(ylims)
        plt.pause(3. / t.size)

    plt.show()


def plot_stability_contours(dxs: np.ndarray[float],
                            diverging_dts: List[np.ndarray[float]],
                            diverging_nus: List[np.ndarray[float]],
                            title: str):
    """Plot the stability boundaries of a scheme in terms of nu and dt, for several values of dx
    
    Args:
       dxs : (np.ndarray[float]) dx value per contour
       diverging_dts : (List[List[float]]) list of lists of dt values, one list per contour
       diverging_nus : (List[List[float]]) list of lists of nu values, one list per contour
       title : (str) Plot title
    """
    
    cmap = matplotlib.cm.get_cmap('spring')
    color_values = np.linspace(0, 1, len(dxs))

    for dx, dts, nus, color in zip(dxs, diverging_dts, diverging_nus, color_values):
        plt.scatter(dts, nus, color=cmap(color), s=8, label=fr"$\Delta x = {dx:.2f}$")

    plt.xlabel(r'$\Delta t$ (time step)', fontsize=13)
    plt.ylabel(r'$\nu$ (viscosity)', fontsize=13)
    plt.title(title, fontsize=14)
    plt.xlim([0, 1.5])
    plt.ylim([0, 1.5])

    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':')

    plt.show()


def plot_stability_contours_comparison(dx: float,
                            diverging_dts: np.ndarray[float],
                            diverging_nus: np.ndarray[float],
                            u0: np.ndarray[float],
                            title: str):
    """
    Plot the stability boundaries of a scheme and compare it to linear-advection diffusion boundaries.

    The boundaries are given in terms of dt and nu, for a given dx.
    The boundaries for the advection term of linearised equation are computed for the mean and max value of the initial condition u0.

    Args:
       dx : (float) dx value
       diverging_dts : (np.ndarray[float]) array of dt values
       diverging_nus : (np.ndarray[float]) array of nu values
       u0 : (np.ndarray[float]) array of initial state
       title : (str) Plot title
    """

    dts = np.linspace(0,max(np.max(diverging_dts),1.5),100)
    plt.scatter(diverging_dts, diverging_nus, color='k', s=4,
            label="Numerical unstability boundary")
    plt.plot(dts, ((np.max(u0)**2) * dts) / 2, color='red',
            label=r'$\Delta t = \frac{2\nu}{(\max(u_0))^2}$')
    plt.plot(dts, ((np.mean(u0)**2) * dts) / 2, color='orange',
            label=r'$\Delta t = \frac{2\nu}{(\mathrm{mean}(u_0))^2}$')
    plt.plot(dts, (dx**2) / (2 * dts), color='blue',
            label=r'$\Delta t = \frac{(\Delta x)^2}{2\nu}$')

    plt.xlabel(r'$\Delta t$ (time step)', fontsize=13)
    plt.ylabel(r'$\nu$ (viscosity)', fontsize=13)
    plt.title(title, fontsize=14)
    plt.ylim([0, 1.5])
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':')

    plt.show()


def plot_moment_evolution(t, m, n, title: str):
    """Plot the relative moment n difference of the system over time.

    Args:
        t : (np.ndarray[float]) Time labels.
        m : (np.ndarray[float]) n moment differences, relative to baseline.
    """
    plt.cla()
    plt.plot(t, m, 'k', marker='o', markersize=1, label="Numerical evolution")
    plt.legend(loc='best')
    plt.ylabel(f'relative moment {n} difference')
    plt.title(title, fontsize=14)
    plt.xlabel('t')
    plt.ylim([-0.5, 0.5])
    plt.legend(fontsize=11)
    plt.show()


def plot_linearized_stability(c, d, v, log_min, log_max, title: str):
    """Plot a 2D colormap of the final total variation for a near-linear Burger's equation.

    Adapted from ChatGPT.

    Args:
        c : (ndarray) Values of the Courant number (dimensionless mean velocity).
        d : (ndarray) Values of the dimensionless viscosity coefficient.
        v_max : Values of the total variation of the velocity at final time.
    """
    c, d, v = np.array(c), np.array(d), np.array(v)
    log_v = [np.log(min(max(v_i, np.exp(log_min)), np.exp(log_max))) for v_i in v]

    # grid definition
    ci = np.linspace(c.min(), c.max(), 300)
    di = np.linspace(d.min(), d.max(), 300)
    Ci, Di = np.meshgrid(ci, di)

    # interpolate to grid
    Vi = griddata((c, d), log_v, (Ci, Di), method='linear')

    # plot
    plt.figure()
    plt.pcolormesh(Ci, Di, Vi, shading='auto', cmap='viridis')
    plt.plot(ci, ci**2 / 2, color='red', linewidth=2, label=r'$c^2 = 2d$')
    plt.plot(ci, 0.5 * np.ones_like(ci), color='purple', linewidth=2, label=r'$2d = 1$')
    plt.title(title)
    plt.legend()
    plt.colorbar()
    plt.xlim(ci.min(), ci.max())
    plt.ylim(di.min(), di.max())
    plt.xlabel(r'Courant number, $c = u_{mean} \cdot \frac{dt}{dx}$')
    plt.ylabel(r'Dimensionless viscosity coefficient, $d = \nu \frac{dt}{dx^2}$')
    plt.show()


def plot_accuracy(dxs, errors, title: str):
    """Plot the errors vs dx on a log-log scale.

    Args:
        dxs : (np.ndarray[float]) Spatial resolution values.
        errors : (np.ndarray[float]) Error values for the solutions corresponding to the values of dx.
        title : (str) Plot title.
    """
    
    log_dx = np.log(dxs)
    log_errors = np.log(errors)

    plt.figure()
    plt.scatter(log_dx, log_errors, color='black')
    plt.xlabel('log(dx)')
    plt.ylabel('log(RMSE)')
    plt.title(title)

    p = np.polyfit(log_dx, log_errors, 1)
    plt.plot(log_dx, np.polyval(p, log_dx), '--', color='red',
             label=f"log(RMSE) = {p[0]:.1f}*log(dx) + {p[1]:.1f}")
    plt.legend(fontsize='x-large')
    plt.show()


def get_ylims(*args):
    """Compute y-axis bounds for readable plots
    
    Args:
        args : Arrays of values to be be plotted - can be one or more.
    
    Return:
        ylims : (list[float]) Min and max values for y-axis
    """
    u_min = min(arg.min() for arg in args)
    u_max = max(arg.max() for arg in args)
    u_mean = (u_min + u_max) / 2.
    alpha = 1.4
    ylims = [alpha * u_min + (1 - alpha) * u_mean, alpha * u_max + (1 - alpha) * u_mean]
    return ylims


def plot_bounds_evolution(t, u_min, u_max, title):
    """Plot the evolution of the minimal and maximal values of a solution over time.

    Mark points where the bounds have diverged compared to previous time step.
    
    Args:
       t : (np.ndarray[float]) Time labels.
       u_min : (np.ndarray[float]) Minimum value of u as a function of time
       u_max : (np.ndarray[float]) Maximum value of u as a function of time.
       title : (str) Plot title.
    """
    bad_inds_min = 1 + np.argwhere(u_min[1:] < u_min[:-1])
    bad_inds_max = 1 + np.argwhere(u_max[1:] > u_max[:-1])
    bad_ts = np.concatenate([t[bad_inds_min], t[bad_inds_max]])
    bad_us = np.concatenate([u_min[bad_inds_min], u_max[bad_inds_max]])

    plt.figure()
    plt.plot(t, u_max, color='black')
    plt.plot(t, u_min, color='black')
    plt.fill_between(t, u_min, u_max, color='blue', alpha=0.5)
    plt.scatter(bad_ts, bad_us, color='red', label='Bounds expanded')
    plt.ylim(get_ylims(u_min, u_max))
    plt.title(title)
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('u')
    plt.show()
