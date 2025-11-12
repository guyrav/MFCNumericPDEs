from typing import Callable
from itertools import product
import numpy as np

from params import NumericalSchemeParams, ViscousParams, AdvectionDiffusionParams
from schemes import Scheme
from state_properties import total_mass, relative_total_mass, total_variation, relative_total_variation


def init(params: NumericalSchemeParams,
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
    u[-1] = u[0]

    if record_all:
        history = np.zeros((params.nt + 1, params.nx + 1), dtype=float)
        history[0, :] = u[:]
        return x, t, u, history
    else:
        return x, t, u


def run_time_evolution(scheme: Scheme, params: NumericalSchemeParams, initial_condition: Callable):
    """Run the numerical scheme given, on a given initial condition.

    Args:
        scheme: (Scheme) Numerical scheme
        params: (Params) Numerical scheme parameters
        initial_condition: (Callable) Function that takes in array of points in space and returns initial state

    Return:
        x : (nparray[float]) Spatial grid
        t : (nparray[float]) Temporal grid
        history : (nparray(nparray[float])) Matrix with dimensions (nt +1)*(nx + 1) storing all values 
                                            of the solution under the given scheme for each grid point 
                                            (in space and time)
    """

    x, t, u_0, history = init(params, initial_condition, record_all=True)

    scheme(params, u_0, history)

    return x, t, history


def get_relative_mass_evolution(history: np.ndarray):
    m_0 = total_mass(history[0, :-1])
    return relative_total_mass(m_0, history[:, :-1])


def get_bounds(history: np.ndarray[float]):
    return history.min(axis=1), history.max(axis=1)


def get_relative_variation_evolution(history: np.ndarray):
    v_0 = total_variation(history[0, :-1])
    return relative_total_variation(v_0, history[:, :-1])


def run_single_variation_experiment(scheme, T, L, nt, u_mean, initial_condition, c, d):
    nx = int(np.ceil(c * L * nt / (L * u_mean)))
    nu = d * L**2 * nt / (T * nx**2)
    params = AdvectionDiffusionParams(T, L, nt, nx, nu, u_mean)

    c_actual = u_mean * params.dtdx
    d_actual = params.d

    x, t, u_0 = init(params, initial_condition, record_all=False)
    v_0 = total_variation(u_0)

    u = scheme(params, u_0)
    v = relative_total_variation(v_0, u)

    return c_actual, d_actual, v


def linearized_stability_experiment(scheme, T, L, nt, u_mean, initial_condition, c_min, c_max, nc, d_min, d_max, nd):
    # c = np.linspace(0.1, 1.5, 40)
    # d = np.linspace(0.1, 0.7, 40)
    c = np.linspace(c_min, c_max, nc)
    d = np.linspace(d_min, d_max, nd)

    # c = u_mean * dt / dx = u_mean * T * nx / (L * nt)
    # d = nu * dt / dx^2 = nu * T * nx^2 / (L^2 * nt)
    # c / T * L * nt / u_mean = nx
    # d / T * L^2 * nt / nx^2 = nu

    cs = []
    ds = []
    vs = []

    for i, j in product(np.arange(c.shape[0]), np.arange(d.shape[0])):
        c_actual, d_actual, v_relative = run_single_variation_experiment(scheme, T, L, nt, u_mean, initial_condition, c[i], d[j])

        cs.append(c_actual)
        ds.append(d_actual)
        vs.append(v_relative)

    return cs, ds, vs


def run_single_accuracy_experiment(scheme: Scheme, params: NumericalSchemeParams, initial_condition: Callable, reference_solution: Callable):
    x, t, u_0 = init(params, initial_condition, record_all=False)

    u_ref = reference_solution(x)

    u = scheme(params, u_0)

    return np.sqrt(np.mean(np.abs(u - u_ref) ** 2))


def reference_solution(scheme: Scheme, T: float, L: float, nu: float, nx: int, initial_condition: Callable):
    nt = scheme.minimal_stable_nt(nx, T, L, nu)
    params = ViscousParams(T, L, nt, nx, nu)
    x, t, u_0 = init(params, initial_condition, record_all=False)

    return x, scheme(params, u_0)


def accuracy_experiment(scheme: Scheme, T, L, nu, initial_condition):
    nx_min_log = 3
    num_points = 7
    base = 2
    nxs = np.logspace(nx_min_log, nx_min_log + num_points - 1, num_points, base=base, dtype=int)
    nts = scheme.minimal_stable_nt(nxs, T, L, nu)
    dxs = []
    errors = []

    x_ref, u_ref = reference_solution(scheme, T, L, nu, 2 * nxs.max(), initial_condition)

    for nx, nt in zip(nxs, nts):
        params = ViscousParams(T, L, nt, nx, nu)
        dxs.append(params.dx)
        errors.append(run_single_accuracy_experiment(scheme, params, initial_condition,
                                                     lambda x: np.interp(x, x_ref, u_ref)))

    return dxs, errors


def divergence_contour_experiment(scheme: Scheme, initial_condition:Callable, L, T,
                                  dxs, dt_min, dt_max, dt_num, nu_min, nu_max, nu_num):
    """
    Explore FTCS stability conditions by varying dt and nu for a set of dx values.
    Marks parameter pairs (dt, nu) that cause divergence: to the left of the curve, the numerical scheme is stable.

    Args:
        scheme : (Scheme) Numerical scheme for the experiment
        initial_condition : (Callable) Function that takes in array of points in space and returns initial state
        L: (float) Length of space interval         
        T: (float) Length of time interval
        dxs : (np.ndarray[float]) array of dx values
        dt_min, dt_max : (float) Bounds for dt values
        dt_num : (int) Number of dt values to test
        nu_min, nu_max : (float) Bounds for nu values
        nu_num : (int) Number of nu values to test

    Return:
        true_dxs : (np.ndarray[float]) True array of dx values used in the experiment
        dt_diverged : (List(List[float])) List of smallest dt value causing divergence, for each dx value tested
        nu_diverged : (List(List[float])) List of smallest nu value causing divergence, for each dx value tested
    """
    dt_values = np.linspace(dt_min, dt_max, dt_num)
    nu_values = np.linspace(nu_min, nu_max, nu_num)

    nt_values = np.unique((T / dt_values).astype(int))
    nt_values = nt_values[::-1]

    nx_values = np.unique((L / dxs).astype(int))
    true_dxs = L/nx_values

    threshold = 1.0  # divergence criterion

    dt_diverged, nu_diverged = [], []
    for _ in nx_values:
        dt_diverged.append([])
        nu_diverged.append([])

    for nx, bad_dts, bad_nus in zip(nx_values, dt_diverged, nu_diverged):
        for nu in nu_values:
            for nt in nt_values:
                params = ViscousParams(T, L, nt, nx, nu)
                x, t, u_0, history = init(params, initial_condition, record_all=True)

                _ = scheme(params, u_0, history)

                v_0 = total_variation(u_0[:-1])

                v = relative_total_variation(v_0, history[1:, :-1])

                if np.any(v > threshold) or np.any(np.isnan(v)):
                    bad_dts.append(params.dt)
                    bad_nus.append(nu)
                    break

    return true_dxs, dt_diverged, nu_diverged
