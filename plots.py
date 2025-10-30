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


def plot_stability_contours(dxs, diverging_dts, diverging_nus, title: str):
    cmap = matplotlib.cm.get_cmap('brg')
    color_values = np.linspace(0, 1, len(dxs))

    for dx, dts, nus, color in zip(dxs, diverging_dts, diverging_nus, color_values):
        plt.scatter(dts, nus, color=cmap(color), label=f"dx = {dx:.2f}")

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


# TODO: Ask Andrea whether this is worth doing
# def plot_variation_evolution(dt, dx, nt, nx, T, L, nu):
#     """Plot the evolution of total variation for different spatial resolutions and viscosity values.
#     Variability is computed as V(t) = (V0 - Σ |u(x, t)- u(x-1,t)|) / V0, with V0 = Σ |u0(x, t)- u0(x-1,t)|
#
#     Parameters
#     ----------
#     dt, dx, nt, nx, T, L, nu : (float or int) Parameters.
#
#     """
#     dx_values = [dt/2, dt, 1.5*dt, 2*dt, 10*dt]
#     nx_values = [40*nx, 20*nx, 15*nx, 10*nx, 5*nx]
#     nu_values = [nu, nu*2, nu*10]
#     time = np.linspace(0.0,T,nt+1)
#
#     for viscosity in nu_values:
#         for  i in range(len(dx_values)):
#             x,p = make_params(dt, dx_values[i], nt, nx_values[i], viscosity)
#             u0 = initiate(x,p,1)
#             u_all = FTCS (x, u0, **p, store_u=True)
#
#             #spatial variability of initial condition
#             shifted_left = np.roll(u0, 1)
#             diff = abs(u0 - shifted_left)
#             variability0 = sum(diff)
#             variability_all = []
#
#             for u in u_all:
#                 variability = total_variation_periodic(u)
#                 #variability_all.append(variability)
#                 variability_all.append((variability - variability0)/variability0)
#
#             plt.plot(time,variability_all,marker='o', markersize=1, label=f'dx= {dx_values[i]:.2f}, nu = {viscosity:.2f}')
#             plt.legend(loc='best')
#             plt.title(f'Spatial absolute variation over time, for dt = {dt:.2f} ')
#             plt.ylabel('Spatial cumulated differences')
#             plt.xlabel('t')
#             plt.xlim([0,L])
#             plt.ylim([-4, 10])
#             plt.pause(0.5)
#     plt.show()


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
