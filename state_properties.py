import numpy as np


def relative(a, b):
    return (b - a) / a


def total_mass(f: np.ndarray):
    return f.sum(axis=-1)


def relative_total_mass(m_0: float, f: np.ndarray):
    return relative(m_0, total_mass(f))


def total_variation(f: np.ndarray):
    return np.sum(np.abs(f - np.roll(f, 1, axis=-1)), axis=-1)


def relative_total_variation(v_0, f: np.ndarray):
    return relative(v_0, total_variation(f))


def energy(f: np.ndarray):
    return np.sum(f ** 2) / 2


def sum_ux_squared_spectral(f: np.ndarray, dx: float):
    f_hat = np.fft.fft(f)
    k = 2 * np.pi * np.fft.fftfreq(f.size, dx)
    return energy(1j * k * f_hat)  # Using Plancherel's theorem


def sum_ux_squared_centered(f: np.ndarray, dx: float):
    return energy(f - np.roll(f, 2, axis=-1)) / (dx ** 2)
