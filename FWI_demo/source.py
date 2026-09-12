import numpy as np

def ricker(t, f0, t0=None):
    if t0 is None:
        t0 = 1.0 / f0
    tau = np.pi * f0 * (t - t0)
    wavelet = (1.0 - 2.0 * tau**2) * np.exp(-tau**2)
    return wavelet

def gaussian(t, f0, t0=None):
    if t0 is None:
        t0 = 1.0 / f0
    tau = np.pi * f0 * (t - t0)
    wavelet = np.exp(-tau**2)
    return wavelet

def gaussian_first_derivative(t, f0, t0=None):
    if t0 is None:
        t0 = 1.0 / f0
    tau = np.pi * f0 * (t - t0)
    wavelet = (-2.0 * tau * np.exp(-tau**2))
    return wavelet

def sine_burst_hann(t, f0, n_cycles=3):
    duration = n_cycles / f0
    wavelet = np.zeros_like(t)
    mask = t <= duration
    tt = t[mask]
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * tt / duration))
    wavelet[mask] = window* np.sin(2.0 * np.pi * f0 * tt)
    return wavelet

def morlet(t, f0, t0=None, n_cycles=5):
    if t0 is None:
        t0 = 1.5 * n_cycles / f0
    sigma = n_cycles / (2.0 * np.pi * f0)
    tau = t - t0
    wavelet = np.exp(-tau**2 / (2.0 * sigma**2)) * np.cos(2.0 * np.pi * f0 * tau)
    return wavelet