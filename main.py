import numpy as np

from params import ViscousParams, AdvectionDiffusionParams
from plots import plot_evolution, plot_stability_contours, plot_mass_evolution, plot_linearized_stability, plot_accuracy
from schemes import BurgersFTCS, AdvectionDiffusionFTCS, BurgersSpectral
from experiments import run_time_evolution, divergence_contour_experiment, get_relative_mass_evolution, \
    linearized_stability_experiment, accuracy_experiment
from initial_conditions import gaussian, near_constant, reverse_step


def main():
    # scheme = BurgersFTCS()
    # T = 20
    # L = 10
    # nt = 120
    # nx = 50
    # nu = 0.1
    # initial_condition = lambda x: np.sqrt(2 * np.pi) * gaussian(L/2, 1)(x)
    # params = ViscousParams(T, L, nt, nx, nu)
    # x, t, u = run_time_evolution(scheme, params, initial_condition)
    # plot_evolution(t, x, u, "Burgers equation, FTCS, gaussian initial condition")

    scheme = BurgersSpectral()
    T = 20
    L = 10
    nt = 120
    nx = 50
    nu = 0.1
    u_mean = 0.2
    epsilon = 0.05
    initial_condition = near_constant(u_mean, epsilon, gaussian(L/2, 1))
    params = AdvectionDiffusionParams(T, L, nt, nx, nu, u_mean)
    x, t, u = run_time_evolution(scheme, params, initial_condition)
    plot_evolution(t, x, u, "Burgers equation, spectral scheme, gaussian initial condition")

    # scheme = BurgersFTCS()
    # nt = 120
    # nx = 50
    # initial_condition = lambda params: lambda x: np.sqrt(2 * np.pi) * gaussian(params.L / 2, 1)(x)
    # dxs = [0.1, 0.3, 0.6, 0.9]
    # diverging_dts, diverging_nus = divergence_contour_experiment(scheme, initial_condition,
    #                                                              dxs, nt, nx,
    #                                                              0.01, 1.0, 100,
    #                                                              0.01, 1.0, 100)
    # plot_stability_contours(dxs, diverging_dts, diverging_nus, "Minimal values of dt and nu for an unstable solution")

    # scheme = BurgersFTCS()
    # T = 20
    # L = 10
    # nx = 50
    # nt = 120
    # nu = 0.1
    # _, t, history = run_time_evolution(scheme, ViscousParams(T, L, nt, nx, nu), gaussian(L / 2, 1))
    # mass = get_relative_mass_evolution(history)
    # plot_mass_evolution(t, mass)

    # # scheme = BurgersFTCS()
    # scheme = AdvectionDiffusionFTCS()
    # T = L = 1
    # nt = 12
    # u_mean = 0.2
    # epsilon = 0.05
    # # initial_condition = near_constant(u_mean, epsilon, gaussian(L/2, 1))
    # initial_condition = near_constant(u_mean, epsilon, reverse_step(1. / 3, 2. / 3))
    # cs, ds, vs = linearized_stability_experiment(scheme, T, L, nt, u_mean, initial_condition,
    #                                              0.1, 1.5, 40, 0.1, 0.7, 40)
    # plot_linearized_stability(cs, ds, vs, log_min=-3, log_max=10,
    #                           title=r"Log Relative Total Variation at $t=1$ (clipped to [-3, 10])")

    # scheme = BurgersFTCS()
    # T = L = 1
    # u_mean = 0.5
    # epsilon = 0.05
    # nu = 0.1
    # initial_condition = near_constant(u_mean, epsilon, gaussian(L/2, 1))
    # # initial_condition = near_constant(u_mean, epsilon, reverse_step(1. / 3, 2. / 3))
    # dxs, errors = accuracy_experiment(scheme, T, L, nu, initial_condition)
    # plot_accuracy(dxs, errors, r"Error vs. high-resolution solution at $t=1$ (with $\Delta t \sim \Delta x^2$)")


if __name__ == "__main__":
    main()
