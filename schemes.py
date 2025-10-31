import numpy as np

from params import NumericalSchemeParams, ViscousParams, AdvectionDiffusionParams


class Scheme(object):
    """Numerical scheme object.

    Methods:

    """
    def __call__(self, params: NumericalSchemeParams, initial_condition: np.ndarray[float], history=None):
        """Evolve the initial condition over time using the numerical scheme.

        Arguments:
            params: (Params) Parameters object.
            initial_condition: (ndarray[float]) Initial condition as a sequence of values.
            history: (ndarray[float], optional) Table of size (nt + 1) * nx or (nt + 1) * (nx + 1)#
                                                to hold all values over time in.
        """
        pass

    @staticmethod
    def minimal_stable_nt(nx, *args, **kwargs):
        pass


    def __str__(self):
        return "Unknown scheme"


class ForwardTimeScheme(Scheme):
    def _apply_step(self, params: NumericalSchemeParams, current_state: np.ndarray[float]):
        """Apply one forward time step, putting the output back in the current state."""
        pass

    def __call__(self, params: NumericalSchemeParams, initial_condition: np.ndarray[float], history=None):
        """Evolve the initial condition over time using the numerical scheme."""
        has_padding = (initial_condition.shape[0] > params.nx)  # Assumed either nx or nx + 1

        u = np.zeros(params.nx, dtype=float)
        u[:] = initial_condition[:-1] if has_padding else initial_condition[:]

        store_u = (history is not None)

        if store_u:
            history[0] = initial_condition

        for n in range(params.nt):
            self._apply_step(params, u)

            if store_u:
                history[n + 1] = np.pad(u, [(0, 1)], mode='wrap') if has_padding else u

        return np.pad(u, [(0, 1)], mode='wrap') if has_padding else u


class BurgersFTCS(ForwardTimeScheme):
    """Forward-time centered-space numerical scheme for the viscous Burgers equation.

    Methods:
        apply_step: Apply one forward time step, putting the output back in the current state.
                    Overridden from parent class.
    """
    def _apply_step(self, params: ViscousParams, current_state: np.ndarray[float]):
        c = params.dtdx
        d = params.d

        shifted_right = np.roll(current_state, -1)
        shifted_left = np.roll(current_state, 1)
        current_state -= (c * 0.5 * current_state * (shifted_right - shifted_left)
                          - d * (shifted_right - 2 * current_state + shifted_left))

    @staticmethod
    def minimal_stable_nt(nx, T: float, L: float, nu: float):
        return int(np.ceil(nu * L / (T ** 2))) * (nx ** 2)

    def __str__(self):
        return "Burgers, FTCS"


class AdvectionDiffusionFTCS(ForwardTimeScheme):
    """Forward-time centered-space numerical scheme for the linear advection-diffusion equation.

    Methods:
        apply_step: Apply one forward time step, putting the output back in the current state.
                    Overridden from parent class.
    """
    def _apply_step(self, params: AdvectionDiffusionParams, current_state: np.ndarray[float]):
        c = params.dtdx
        d = params.d
        v = params.v

        shifted_right = np.roll(current_state, -1)
        shifted_left = np.roll(current_state, 1)
        current_state -= (c * v * 0.5 * (shifted_right - shifted_left)
                          - d * (shifted_right - 2 * current_state + shifted_left))

    @staticmethod
    def minimal_stable_nt(nx, T: float, L: float, nu: float):
        return int(np.ceil(nu * L / (T ** 2))) * (nx ** 2)

    def __str__(self):
        return "Linear advection-diffusion, FTCS"


class AdvectionDiffusionSpectral(Scheme):
    """Numerical scheme for solving the linear advection-diffusion equation by Fourier analysis."""
    @staticmethod
    def _solve(params, k, u_0, t):
        return np.fft.ifft(np.fft.fft(u_0) * np.exp(-(params.nu * k**2 + 1j * params.v * k) * t))

    def __call__(self, params: AdvectionDiffusionParams, initial_condition: np.ndarray[float], history=None):
        if history is not None:
            history[0, :] = initial_condition

        has_padding = (initial_condition.shape[0] > params.nx)  # Assumed either nx or nx + 1
        if has_padding:
            initial_condition = initial_condition[:-1]

        k = 2 * np.pi * np.fft.fftfreq(params.nx, params.dx)

        if history is not None:
            k = k.reshape((1,) + k.shape)
            initial_condition = initial_condition.reshape((1,) + initial_condition.shape)
            t = np.linspace(params.t_start + params.dt, params.t_end, params.nt).reshape((params.nt, 1))
        else:
            t = params.T

        solution = self._solve(params, k, initial_condition, t)
        if has_padding:
            solution = np.pad(solution, [(0, 0)] * (k.ndim - 1) + [(0, 1)], mode='wrap')

        if history is not None:
            history[1:, :] = solution
            return solution[-1]
        else:
            return solution

    def __str__(self):
        return "Linear advection-diffusion, spectral time & space"


class Leapfrog(Scheme):
    def __init__(self, first_step_scheme: Scheme):
        self.first_step_scheme = first_step_scheme

    def _apply_first_step(self, first_step_params: NumericalSchemeParams,
                    previous_state: np.ndarray[float],
                    current_state: np.ndarray[float]):
        previous_state[:] = current_state[:]
        self.first_step_scheme(first_step_params, current_state)

    def _apply_step(self, params: NumericalSchemeParams,
                    previous_state: np.ndarray[float],
                    current_state: np.ndarray[float]):
        pass

    def _get_first_step_params(self, params: NumericalSchemeParams) -> NumericalSchemeParams:
        pass

    def __call__(self, params: NumericalSchemeParams, initial_condition: np.ndarray[float], history=None):
        """Evolve the initial condition over time using the numerical scheme."""
        has_padding = (initial_condition.shape[0] > params.nx)  # Assumed either nx or nx + 1

        u = np.zeros(params.nx, dtype=float)
        u_prev = np.zeros_like(u)
        u[:] = initial_condition[:-1] if has_padding else initial_condition[:]

        store_u = (history is not None)

        if store_u:
            history[0] = initial_condition

        if params.nt > 0:
            self._apply_first_step(self._get_first_step_params(params), u_prev, u)

            if store_u:
                history[1] = np.pad(u, [(0, 1)], mode='wrap') if has_padding else u

        for n in range(1, params.nt):
            self._apply_step(params, u_prev, u)

            if store_u:
                history[n + 1] = np.pad(u, [(0, 1)], mode='wrap') if has_padding else u

        return np.pad(u, [(0, 1)], mode='wrap') if has_padding else u


class BurgersLeapfrog(Leapfrog):
    def _get_first_step_params(self, params: ViscousParams) -> NumericalSchemeParams:
        return ViscousParams(params.dt, params.L, 1, params.nx, params.nu)

    def _apply_step(self, params: ViscousParams,
                    previous_state: np.ndarray[float],
                    current_state: np.ndarray[float]):
        c = params.dtdx
        d = params.d

        shifted_right = np.roll(current_state, -1)
        shifted_left = np.roll(current_state, 1)
        previous_shifted_right = np.roll(previous_state, -1)
        previous_shifted_left = np.roll(previous_state, 1)
        temp = current_state.copy()
        current_state[:] = previous_state - 2 * (c * 0.5 * current_state * (shifted_right - shifted_left)
                                                 - d * (previous_shifted_right - 2 * previous_state + previous_shifted_left))
        previous_state[:] = temp

    def __str__(self):
        return "Burgers, CTCS with FT diffusion"


class BurgersSemiSpectral(ForwardTimeScheme):
    def _apply_step(self, params: ViscousParams, current_state: np.ndarray[float]):
        c = params.dtdx
        d = params.d

        u_hat = np.fft.fft(current_state)
        k = 2 * np.pi * np.fft.fftfreq(params.nx, params.dx)

        u_x = np.real(np.fft.ifft(1j * k * u_hat))
        u_xx = np.real(np.fft.ifft(- k**2 * u_hat))

        current_state -= params.dt * (current_state * u_x - params.nu * u_xx)

    def __str__(self):
        return "Burgers, FT spectral in space"
