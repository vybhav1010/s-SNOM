import numpy as np
from scipy.optimize import least_squares

from data_gen import n_wav, wavenumber
from optimizer import eta_to_vector, forward_function


MESH_STRIDE = 2
MAX_NFEV = 200
LOCAL_SNR_WINDOW = 5
SNR_FLOOR = 1e-4
ETA_STD_FLOOR = 1e-12

mesh_indices = np.arange(0, n_wav, MESH_STRIDE, dtype=np.int64)
# if mesh_indices[-1] != n_wav - 1:
#     mesh_indices = np.append(mesh_indices, n_wav - 1)

mesh_wavenumbers = np.asarray(wavenumber, dtype=np.float64)[mesh_indices]


def _x_log_abs_x(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    nonzero_mask = x != 0.0
    result[nonzero_mask] = x[nonzero_mask] * np.log(np.abs(x[nonzero_mask]))
    return result

def triangle_fit_im(mesh_values):
    mesh_values = np.asarray(mesh_values, dtype=np.float64)
    if mesh_values.shape != mesh_wavenumbers.shape:
        raise ValueError(
            "mesh_values must have the same shape as mesh_wavenumbers. "
            f"Expected {mesh_wavenumbers.shape}, got {mesh_values.shape}."
        )

    return np.interp(
        np.asarray(wavenumber, dtype=np.float64),
        mesh_wavenumbers,
        mesh_values,
    )


def g_func(x, y):
    return _x_log_abs_x(x + y) + _x_log_abs_x(x - y)


def local_eta_snr(eta_values, window_size=LOCAL_SNR_WINDOW):
    eta_values = np.asarray(eta_values, dtype=np.complex128)
    half_window = window_size // 2
    snr = np.empty_like(eta_values.real, dtype=np.float64)

    for index in range(len(eta_values)):
        window_start = max(0, index - half_window)
        window_stop = min(len(eta_values), index + half_window + 1)
        eta_window = eta_values[window_start:window_stop]
        mean_eta = np.mean(eta_window)
        noise_std = np.std(eta_window - mean_eta)
        snr[index] = np.abs(mean_eta) / max(noise_std, ETA_STD_FLOOR)

    return np.maximum(snr, SNR_FLOOR)


def kk(eps_var_im, mesh_values=None):
    del mesh_values

    omega = np.asarray(wavenumber, dtype=np.float64)
    eps_var_im = np.asarray(eps_var_im, dtype=np.float64)

    if eps_var_im.shape != omega.shape:
        raise ValueError(
            "eps_var_im must live on the full wavenumber grid. "
            f"Expected {omega.shape}, got {eps_var_im.shape}."
        )

    weights = np.gradient(omega)
    numerator = omega[None, :] * weights[None, :] * eps_var_im[None, :]
    denominator = (omega[None, :] ** 2) - (omega[:, None] ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = (2.0 / np.pi) * numerator / denominator

    np.fill_diagonal(kernel, 0.0)
    kernel[~np.isfinite(kernel)] = 0.0

    return np.sum(kernel, axis=1)



def mesh_forward(mesh_values, eps_initial):
    eps_initial = np.asarray(eps_initial, dtype=np.complex128)
    eps_var_im = triangle_fit_im(mesh_values)
    eps_var_real = kk(eps_var_im, mesh_values)
    eps_var = eps_var_real + 1j * eps_var_im

    return forward_function(eps_initial + eps_var)


def kk_optimize(eta, eps_initial):
    eps_initial = np.asarray(eps_initial, dtype=np.complex128)
    eta_array = np.asarray(eta)
    if eta_array.shape == (n_wav,):
        target_eta_vector = eta_to_vector(eta_array)
    else:
        target_eta_vector = eta_array.astype(np.float64, copy=False)

    target_eta_real = target_eta_vector[:n_wav]
    target_eta_imag = target_eta_vector[n_wav:]
    target_eta = target_eta_real + 1j * target_eta_imag
    target_eta_snr = local_eta_snr(target_eta)

    
    def residual(mesh_values):
        decay_factor = 1e-1

        min_div = 1e-4
        eta_residual = mesh_forward(mesh_values, eps_initial) - target_eta_vector

        mesh_eps_imag = eps_initial.imag[mesh_indices]
        mesh_divisor = np.maximum(np.abs(mesh_eps_imag), min_div)
        mesh_percent_diff_loss = np.abs(mesh_values) / mesh_divisor
        mesh_weighted_diff = mesh_percent_diff_loss / target_eta_snr[mesh_indices]

        regularization = (
            decay_factor * np.tile(mesh_weighted_diff, MESH_STRIDE * 2) / 2
        )[:(2 * n_wav)]

        return eta_residual + regularization
    


    eps_scale = float(np.max(np.abs(eps_initial)))
    if not np.isfinite(eps_scale) or eps_scale == 0.0:
        eps_scale = 1.0

    rng = np.random.default_rng(0)
    mesh_initial = rng.uniform(
        low=-0.25 * eps_scale,
        high=0.25 * eps_scale,
        size=mesh_wavenumbers.shape,
    )

    lower_bounds = np.full(mesh_wavenumbers.shape, -eps_scale, dtype=np.float64)
    upper_bounds = np.full(mesh_wavenumbers.shape, eps_scale, dtype=np.float64)

    result = least_squares(
        residual,
        mesh_initial,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        loss="soft_l1",
        x_scale=upper_bounds - lower_bounds,
        max_nfev=MAX_NFEV,
    )

    result.eps_var_im = triangle_fit_im(result.x)
    result.eps_var_real = kk(result.eps_var_im)
    result.eps_recovered = (
        eps_initial + result.eps_var_real + 1j * result.eps_var_im
    )
    result.eta_recovered = forward_function(result.eps_recovered)

    return result
