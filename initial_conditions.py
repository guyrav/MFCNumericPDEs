import numpy as np
from typing import Callable


def gaussian(mu: float, sigma: float) -> Callable:
    """
    Return a gaussian function with mean mu and variance sigma^2
    """
    def ret(x: np.ndarray[float]) -> np.ndarray[float]:
        return 1. / np.sqrt(2 * np.pi * sigma) * np.exp(- ((x - mu) / sigma)**2 / 2)
    return ret


def reverse_step(start, end):
    """
    Return a function that produces a descending linear ramp between `start` and `end`.
    """
    def ret(x):
        result = np.zeros_like(x)
        mask = np.logical_and(x > start, x < end)
        line = 1 - (x - start) / (end - start)
        result[mask] = line[mask]
        return result
    return ret


def near_constant(u_0: float, epsilon: float, perturbation: Callable):
    """
    Return a function consisting of a mean velocity plus a perturbation scaled by epsilon.
    """
    def ret(x: np.ndarray[float]) -> np.ndarray[float]:
        return u_0 * (1 + epsilon * perturbation(x))
    return ret


def sine_wave(k):
    """
    Return a sine wave function with wavenumber `k`.
    """
    def ret(x):
        return np.sin(k * x)
    return ret