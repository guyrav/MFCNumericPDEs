import numpy as np


def relative(a, b):
    return (b - a) / a


def moment(f: np.ndarray, n : int=1):
    h = f**n
    return h.sum(axis=-1)


def relative_moment(ref: float,f: np.ndarray, n : int=1):
    return relative(ref, moment(f,n))


def total_variation(f: np.ndarray):
    return np.sum(np.abs(f - np.roll(f, 1, axis=-1)), axis=-1)


def relative_total_variation(v_0, f: np.ndarray):
    return relative(v_0, total_variation(f))
