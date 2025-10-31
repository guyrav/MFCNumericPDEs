import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy.interpolate import griddata


def plot_evolution(t: np.ndarray[float], x: np.ndarray[float], u: np.ndarray[float], title: str):
    """
    Animate and plot the evolution of the solution over time.

    Parameters
    ----------
    t : (np.ndarray) Time labels.
    x : (np.ndarray) Space labels.
    u : (np.ndarray) Velocity field at all time steps.
    title : (str) Plot title.
    """
    u_0_min, u_0_mean, u_0_max = u[0].min(), u[0].mean(), u[0].max()
    alpha = 1.4
    ylims = [alpha * u_0_min + (1 - alpha) * u_0_mean, alpha * u_0_max + (1 - alpha) * u_0_mean]

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


def plot_evolution_comparison(t: np.ndarray[float], x: np.ndarray[float],
                              u_a: np.ndarray[float], u_b: np.ndarray[float],
                              title: str, label_a: str, label_b: str):
    u_0_min, u_0_mean, u_0_max = u_a[0].min(), u_a[0].mean(), u_a[0].max()
    alpha = 1.4
    ylims = [alpha * u_0_min + (1 - alpha) * u_0_mean, alpha * u_0_max + (1 - alpha) * u_0_mean]

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


def plot_stability_contours(dxs, diverging_dts, diverging_nus, title: str):
    cmap = matplotlib.cm.get_cmap('spring')
    color_values = np.linspace(0, 1, len(dxs))

    for dx, dts, nus, color in zip(dxs, diverging_dts, diverging_nus, color_values):
        plt.scatter(dts, nus, color=cmap(color), s=4, label=f"dx = {dx:.2f}")

    plt.legend()
    plt.title(title)
    plt.xlabel('dt')
    plt.ylabel('nu')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.show()


def plot_mass_evolution(t, m):
    """
    Plot the relative mass difference of the system over time, given by $M(t) = (M_0 - Σ_x u(t, x)) / M_0$.

    Parameters
    ----------
    t: (np.ndarray) Time labels.
    m: (np.ndarray) Relative mass differences.
    """
    plt.cla()
    plt.plot(t, m, 'k', marker='o', markersize=1, label="M(t)")
    plt.legend(loc='best')
    plt.ylabel('relative mass difference')
    plt.xlabel('t')
    plt.ylim([-0.01, 0.01])
    plt.show()


def plot_linearized_stability(c, d, v, log_min, log_max, title: str):
    """Plot a 2D colormap of the final total variation for a near-linear Burger's equation.

    Adapted from ChatGPT.

    Args:
        c: (ndarray) Values of the Courant number (dimensionless mean velocity)
        d: (ndarray) Values of the dimensionless viscosity coefficient
        v_max: Values of the total variation of the velocity at final time
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
    log_dx = np.log(dxs)
    log_errors = np.log(errors)

    plt.figure()
    plt.scatter(log_dx, log_errors, color='black')
    plt.xlabel('log(dx)')
    plt.ylabel('log(MSE)')
    plt.title(title)

    p = np.polyfit(log_dx, log_errors, 1)
    plt.plot(log_dx, np.polyval(p, log_dx), '--', color='red',
             label=f"log(MSE) = {p[0]:.1f}*log(dx) + {p[1]:.1f}")
    plt.legend()
    plt.show()
