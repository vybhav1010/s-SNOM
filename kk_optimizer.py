import numpy as np
from scipy.optimize import least_squares

from data_gen import n_wav, wavenumber
from optimizer import eta_to_vector, forward_function


MESH_STRIDE = 2
LOCAL_SNR_WINDOW = 5
SNR_FLOOR = 1e-4
ETA_STD_FLOOR = 1e-12

mesh_indices = np.arange(0, n_wav, MESH_STRIDE, dtype=np.int64)
if mesh_indices[-1] != n_wav - 1:
    mesh_indices = np.append(mesh_indices, n_wav - 1)

mesh_wavenumbers = np.asarray(wavenumber, dtype=np.float64)[mesh_indices]


def _linear_continuation_weights(nodes, evaluation_point):
    """Weights for the least-squares line through three mesh nodes."""
    nodes = np.asarray(nodes, dtype=np.float64)
    design = np.column_stack((nodes, np.ones(nodes.shape)))
    return np.array([evaluation_point, 1.0], dtype=np.float64) @ np.linalg.pinv(design)


left_ghost_wavenumber = 2.0 * mesh_wavenumbers[0] - mesh_wavenumbers[1]
right_ghost_wavenumber = 2.0 * mesh_wavenumbers[-1] - mesh_wavenumbers[-2]
left_outer_wavenumber = 2.0 * left_ghost_wavenumber - mesh_wavenumbers[0]
right_outer_wavenumber = 2.0 * right_ghost_wavenumber - mesh_wavenumbers[-1]
left_continuation_weights = _linear_continuation_weights(
    mesh_wavenumbers[:3], left_ghost_wavenumber
)
right_continuation_weights = _linear_continuation_weights(
    mesh_wavenumbers[-3:], right_ghost_wavenumber
)

extended_wavenumbers = np.concatenate(
    (
        [left_outer_wavenumber, left_ghost_wavenumber],
        mesh_wavenumbers,
        [right_ghost_wavenumber, right_outer_wavenumber],
    )
)


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

    extended_values = np.concatenate(
        (
            [0.0, left_continuation_weights @ mesh_values[:3]],
            mesh_values,
            [right_continuation_weights @ mesh_values[-3:], 0.0],
        )
    )
    return np.interp(
        np.asarray(wavenumber, dtype=np.float64),
        extended_wavenumbers,
        extended_values,
        left=0.0,
        right=0.0,
    )

def g_func(x, y):
    return _x_log_abs_x(x + y) + _x_log_abs_x(x - y)


def triangle_kk_matrix():
    """Return analytic KK coefficients for the imaginary-part hat basis.

    Column ``i`` is epsilon_1,i^Delta(omega) from the triangular-basis
    expression. The recovered real correction is therefore the matrix-vector
    product of these coefficients with epsilon_2^Delta(omega_i), the fitted
    imaginary correction at each mesh node.
    """
    omega = np.asarray(wavenumber, dtype=np.float64)[:, None]
    omega_im1 = extended_wavenumbers[:-2][None, :]
    omega_i = extended_wavenumbers[1:-1][None, :]
    omega_ip1 = extended_wavenumbers[2:][None, :]

    left_spacing = omega_i - omega_im1
    right_spacing = omega_ip1 - omega_i
    coefficients = (
        g_func(omega_im1, omega) / left_spacing
        - (
            (omega_ip1 - omega_im1)
            * g_func(omega_i, omega)
            / (left_spacing * right_spacing)
        )
        + g_func(omega_ip1, omega) / right_spacing
    ) / np.pi
    continuation_map = np.zeros(
        (extended_wavenumbers.size - 2, mesh_wavenumbers.size), dtype=np.float64
    )
    continuation_map[0, :3] = left_continuation_weights
    continuation_map[1 : mesh_wavenumbers.size + 1, :] = np.eye(
        mesh_wavenumbers.size
    )
    continuation_map[-1, -3:] = right_continuation_weights
    return coefficients @ continuation_map


triangle_kk_coefficients = triangle_kk_matrix()


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


def kk(mesh_values):
    """Calculate the real correction from fitted imaginary mesh values."""
    mesh_values = np.asarray(mesh_values, dtype=np.float64)
    if mesh_values.shape != mesh_wavenumbers.shape:
        raise ValueError(
            "mesh_values must have the same shape as mesh_wavenumbers. "
            f"Expected {mesh_wavenumbers.shape}, got {mesh_values.shape}."
        )
    return triangle_kk_coefficients @ mesh_values



def mesh_forward(mesh_values, eps_initial):
    eps_initial = np.asarray(eps_initial, dtype=np.complex128)
    eps_var_im = triangle_fit_im(mesh_values)
    eps_var_real = kk(mesh_values)
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
        mesh_weighted_diff = mesh_percent_diff_loss / np.pow(target_eta_snr[mesh_indices], 2)

        regularization = decay_factor * mesh_weighted_diff

        return np.concatenate((eta_residual, regularization))
    


    eps_scale = float(np.max(np.abs(eps_initial)))
    if not np.isfinite(eps_scale) or eps_scale == 0.0:
        eps_scale = 1.0

    mesh_initial = np.zeros(mesh_wavenumbers.shape, dtype=np.float64)

    lower_bounds = np.full(mesh_wavenumbers.shape, -eps_scale, dtype=np.float64)
    upper_bounds = np.full(mesh_wavenumbers.shape, eps_scale, dtype=np.float64)

    result = least_squares(
        residual,
        mesh_initial,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        loss="linear",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        x_scale=upper_bounds - lower_bounds,
        verbose=2,
    )

    result.eps_var_im = triangle_fit_im(result.x)
    result.eps_var_real = kk(result.x)
    result.eps_recovered = (
        eps_initial + result.eps_var_real + 1j * result.eps_var_im
    )
    result.eta_recovered = forward_function(result.eps_recovered)

    return result
