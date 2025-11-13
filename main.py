import numpy as np

from params import ViscousParams, AdvectionDiffusionParams
from plots import plot_evolution, plot_stability_contours, plot_stability_contours_comparison, plot_moment_evolution, plot_linearized_stability, \
    plot_accuracy, plot_evolution_comparison, plot_bounds_evolution, plot_initial_condition
from schemes import BurgersFTCS, AdvectionDiffusionFTCS, AdvectionDiffusionSpectral, BurgersLeapfrog, \
    BurgersSemiSpectral
from experiments import run_time_evolution, divergence_contour_experiment, get_relative_moment_evolution, \
    linearized_stability_experiment, accuracy_experiment, reference_solution, get_bounds
from initial_conditions import gaussian, near_constant, reverse_step, sine_wave, step_function


def run_evolution(scheme, T, L, nt, nx, nu):
    # initial_condition = lambda x: 10 * np.sqrt(2 * np.pi) * gaussian(L / 2, 1)(x)
    initial_condition = lambda x: np.sqrt(2 * np.pi) * gaussian(L/2, 1)(x)
    # initial_condition = near_constant(0.2, 0.05, reverse_step(1. / 3, 2. / 3))
    # initial_condition = sine_wave(2 * np.pi / L)
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
    x = np.linspace(0, L, 100, dtype=float)
    u_0 = initial_condition(x)
    plot_initial_condition(x, u_0, f"Initial condition, $\\epsilon / u_{{\\text{{mean}}}} = {epsilon}$")
    plot_linearized_stability(cs, ds, vs, log_min=-3, log_max=10,
                              title=f"{str(scheme)}, Relative Total Variation at t={T} (clipped)")


def run_accuracy(scheme):
    T = L = 1
    u_mean = 0.5
    epsilon = 0.05
    nu = 0.1
    # initial_condition = near_constant(u_mean, epsilon, gaussian(L/2, 1))
    # initial_condition = near_constant(u_mean, epsilon, reverse_step(1. / 3, 2. / 3))
    initial_condition = gaussian(L/2, 1)
    dxs, errors = accuracy_experiment(scheme, T, L, nu, initial_condition)
    plot_accuracy(dxs, errors, r"Error vs. high-resolution solution at $t=1$ (with $\Delta t \sim \Delta x^2$)")


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



def stability_experiments(scheme):
    """
    Explore stability conditions of a numerical scheme.

    Different parameters defining the conditions of the simulation are hard coded.
    A list of initial states is chosen and the experiments are run for each of them.

    Plots generated:
        Plot of initial state
        Plot of stability contours of the scheme with respect to nu and dt, for different resolutions dx
        Plot of stability contours of the scheme against stability boundaries for the linear case, for a chosen dx  
    """
    T = 20
    L = 10

    dx_values = np.array([0.1, 0.3, 0.6, 0.9])

    dt_min, dt_max = 0.01, 1.5
    nu_min, nu_max = 0.01, 1.5
    n_dt = 100
    n_nu = 100

    initial_conditions = [lambda x: 4 * np.sqrt(2 * np.pi) * gaussian(L/2, 1)(x), lambda x: 2* np.sqrt(2 * np.pi) * gaussian(L/2, 1)(x),
                         near_constant(0.2, 0.5, reverse_step(1. / 3, 3. / 3)), step_function(L, a=2) ]
    #initial_conditions =[step_function(L, a=4) ]

    for f in initial_conditions:
        x = np.linspace(0,L,100)
        plot_initial_condition(x,f(x), "Initial state")
        real_dxs, diverging_dts, diverging_nus = divergence_contour_experiment(scheme, f, L, T, dx_values, 
                                                                     dt_min, dt_max, n_dt, nu_min, 
                                                                     nu_max, n_nu)

        plot_stability_contours(real_dxs, diverging_dts, diverging_nus, r'Minimal values of $\Delta t$ and $\nu$ for an unstable solution')
        plot_stability_contours_comparison(real_dxs[1], np.array(diverging_dts[1]), np.array(diverging_nus[1]), f(x), "Comparison linear and non-linear stability boundaries")



def run_moment_evolution(scheme):
    """
    Visualize mass conservation and energy decay, for a given scheme and non constant initial state.

    Plots generated:
        Plot of relative mass evolution
        Plot of relative energy evolution
    """
    T = 20
    L = 10
    nx = 30
    nt = 120
    nu = 0.2

    initial_state = step_function(L, a=2)
  
    _, t, history = run_time_evolution(scheme, ViscousParams(T, L, nt, nx, nu),  initial_state)
    mass = get_relative_moment_evolution(history,1)
    moment2 = get_relative_moment_evolution(history,2)
    energy = moment2/2

    plot_moment_evolution(t, mass,1, "Relative mass for step function initial state")
    plot_moment_evolution(t, energy,2, "Relative energy for step function initial state")






def main():
    # run_evolution(BurgersFTCS(), 20, 10, 120, 50, 0.1)

    # compare_evolution(BurgersFTCS(), AdvectionDiffusionSpectral(), 20, 10, 120, 50, 0.1,
                    #   "FTCS", "Spectral")
    

    # run_cd_linearized_stability(BurgersFTCS(), 1, 1, 12, 0.2, 1.,
    #                             reverse_step(1. / 3, 2. / 3))

    #run_accuracy(BurgersFTCS())

    # run_bounds_evolution(BurgersLeapfrog(BurgersFTCS()), 1, 5, 0.1, 0.05)



    # tests Andrea :
    # run_moment_evolution(BurgersFTCS())
    # stability_experiments(BurgersFTCS()) 


if __name__ == "__main__":
    main()
