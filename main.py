import numpy as np

from params import ViscousParams, AdvectionDiffusionParams
from plots import plot_evolution, plot_stability_contours, plot_mass_evolution, plot_linearized_stability, \
    plot_accuracy, plot_evolution_comparison, plot_bounds_evolution
from schemes import BurgersFTCS, AdvectionDiffusionFTCS, AdvectionDiffusionSpectral, BurgersLeapfrog, \
    BurgersSemiSpectral
from experiments import run_time_evolution, divergence_contour_experiment, get_relative_mass_evolution, \
    linearized_stability_experiment, accuracy_experiment, reference_solution, get_bounds
from initial_conditions import gaussian, near_constant, reverse_step, sine_wave


def run_evolution(scheme, T, L, nt, nx, nu):
    # initial_condition = lambda x: 10 * np.sqrt(2 * np.pi) * gaussian(L / 2, 1)(x)
    # initial_condition = lambda x: np.sqrt(2 * np.pi) * gaussian(L/2, 1)(x)
    # initial_condition = near_constant(0.2, 0.05, reverse_step(1. / 3, 2. / 3))
    initial_condition = sine_wave(2 * np.pi / L)
    params = ViscousParams(T, L, nt, nx, nu)
    x, t, u = run_time_evolution(scheme, params, initial_condition)
    plot_evolution(t, x, u, f"{str(scheme)}, gaussian initial condition")


def compare_evolution(scheme_a, scheme_b, T, L, nt, nx, nu, label_a, label_b):
    u_mean = 0.2
    initial_condition = near_constant(u_mean, 0.05, reverse_step(1. / 3, 2. / 3))
    params = AdvectionDiffusionParams(T, L, nt, nx, nu, u_mean)
    x, t, u_a = run_time_evolution(scheme_a, params, initial_condition)
    _, _, u_b = run_time_evolution(scheme_b, params, initial_condition)
    plot_evolution_comparison(t, x, u_a, u_b, "Gaussian initial condition", label_a, label_b)


def run_cd_linearized_stability(scheme, T, L, nt, u_mean, epsilon, perturbation):
    initial_condition = near_constant(u_mean, epsilon, perturbation)
    cs, ds, vs = linearized_stability_experiment(scheme, T, L, nt, u_mean, initial_condition,
                                                 0.1, 1.5, 40, 0.1, 0.7, 40)
    plot_linearized_stability(cs, ds, vs, log_min=-3, log_max=10,
                              title=f"{str(scheme)}, Relative Total Variation at t={T} (clipped)")


def run_stability_contours(scheme):
    nt = 120
    nx = 50
    # initial_condition = lambda params: lambda x: 5 * np.sqrt(2 * np.pi) * gaussian(params.L / 2, 1)(x)
    initial_condition = lambda params: sine_wave(2 * np.pi / params.L)
    dxs = [0.1, 0.3, 0.6, 0.9]
    diverging_dts, diverging_nus = divergence_contour_experiment(scheme, initial_condition,
                                                                 dxs, nt, nx,
                                                                 0.01, 1.0, 100,
                                                                 0.01, 1.0, 100)
    plot_stability_contours(dxs, diverging_dts, diverging_nus, "Minimal values of dt and nu for an unstable solution")


def run_accuracy(scheme):
    T = L = 1
    u_mean = 0.5
    epsilon = 0.05
    nu = 0.1
    initial_condition = near_constant(u_mean, epsilon, gaussian(L/2, 1))
    # initial_condition = near_constant(u_mean, epsilon, reverse_step(1. / 3, 2. / 3))
    dxs, errors = accuracy_experiment(scheme, T, L, nu, initial_condition)
    plot_accuracy(dxs, errors, r"Error vs. high-resolution solution at $t=1$ (with $\Delta t \sim \Delta x^2$)")


def run_mass_evolution(scheme):
    T = 20
    L = 10
    nx = 50
    nt = 120
    nu = 0.1
    _, t, history = run_time_evolution(scheme, ViscousParams(T, L, nt, nx, nu), gaussian(L / 2, 1))
    mass = get_relative_mass_evolution(history)
    plot_mass_evolution(t, mass)


def run_bounds_evolution(scheme, dt, dx, nu, epsilon):
    nt = 40
    nx = 100
    T = nt * dt
    L = nx * dx
    u_mean = 0.2
    params = AdvectionDiffusionParams(T, L, nt, nx, nu,  u_mean)
    initial_condition = near_constant(u_mean, epsilon, reverse_step(1. / 3 * L, 2. / 3 * L))
    _, t, history = run_time_evolution(scheme, params, initial_condition)
    u_min, u_max = get_bounds(history)
    plot_bounds_evolution(t, u_min, u_max, "Bounds of $u(t, x)$ over time")
    # plot_bounds_evolution(t, u_min, u_max,
    #                       f"Bounds of $u(x, t)$ with "
    #                       f"$\\frac{{\\Delta_t\\Delta_u}}{{2\\Delta_x}}={dt * epsilon / dx:.2f}$ "
    #                       f"and $\\frac{{\\nu\\Delta_t\\Delta_u}}{{\\Delta_x^2}}={nu * dt * epsilon / dx**2:.2f}$")



def main():
    # run_evolution(BurgersFTCS(), 20, 10, 120, 50, 0.1)

    # compare_evolution(BurgersFTCS(), AdvectionDiffusionSpectral(), 20, 10, 120, 50, 0.1,
    #                   "FTCS", "Spectral")

    # run_stability_contours(BurgersFTCS())

    # run_mass_evolution(BurgersSemiSpectral())

    # run_cd_linearized_stability(BurgersSemiSpectral(), 1, 1, 12, 0.2, 0.05,
    #                             reverse_step(1. / 3, 2. / 3))

    # run_accuracy(AdvectionDiffusionSpectral())

    run_bounds_evolution(BurgersLeapfrog(BurgersFTCS()), 1, 5, 0.1, 0.05)


if __name__ == "__main__":
    main()
