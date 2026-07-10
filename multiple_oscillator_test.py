import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from time import perf_counter
from conv_network_2 import ConvNetwork, MODEL_PATH
from data_gen import (
    A,
    c_r,
    drude_wavenumber_grid,
    eps_inf_range,
    eta_to_vector,
    gamma_p_range,
    gamma_ph_range,
    harmonic,
    n_wav,
    plas_freq_range,
    theta_in,
    trans_phon_frequency_range,
    wavenumber,
    z,
)
from kk_optimizer import kk_optimize
from optimizer import (
    compute_strength_multiple_bounds,
    epsilon_from_params,
    forward_function,
    optimize_parameters,
)
from peak_fit_plots import plot_peak_fit_progress
import scipy.stats
import snompy


eps_inf = 1.6
gamma = 20
trans_phon_frequency = 876
long_phon_frequency = 886
drude_gamma = float(np.median(gamma_p_range))
drude_plasma_frequency = float(np.median(plas_freq_range))

batch_size = 16

long_phon_frequency_stop = 960
long_phon_frequency_step = 100

ERROR_WINDOW = (550.0, 800.0)
NUM_FIT_ITERATIONS = 2

def alpha_fff_ref():
    eps = 11.7
    ref_sample = snompy.sample.bulk_sample(eps_sub=eps)  # Si
    alpha_eff = snompy.fdm.eff_pol_n(sample=ref_sample, A_tip=A, n=harmonic, z_tip=z)

    r = ref_sample.refl_coef(theta_in=theta_in)
    fff_ref = (1 + c_r * r) ** 2

    return alpha_eff, fff_ref


def eta_from_eps_sub(eps_sub):
    sample = snompy.sample.bulk_sample(eps_sub=eps_sub)
    sample_r = sample.refl_coef(theta_in=theta_in)
    fff_sample = (1 + c_r * sample_r) ** 2
    alpha_ref, fff_ref = alpha_fff_ref()
    alpha_eff_sample = snompy.fdm.eff_pol_n(
        sample=sample,
        A_tip=A,
        n=harmonic,
        z_tip=z,
    )
    return (fff_sample * alpha_eff_sample) / (fff_ref * alpha_ref)


def _window_mask(wavenumber_values, window):
    window_min, window_max = window
    window_mask = (wavenumber_values >= window_min) & (wavenumber_values <= window_max)
    windowed_wavenumber = wavenumber_values[window_mask]
    return window_min, window_max, window_mask, windowed_wavenumber


def _absolute_and_percent_errors(true_values, fitted_values):
    denom_floor = 1e-12
    absolute_error = np.abs(fitted_values - true_values)
    percent_error = 100.0 * (fitted_values - true_values) / np.maximum(
        np.abs(true_values),
        denom_floor,
    )
    return absolute_error, percent_error


def plot_eta_summary(
    wavenumber_values,
    true_eta,
    lorentz_eta,
    recovered_eta,
    eta_residual_vector,
    window,
    output_filename="multiple_oscillator_eta_summary.png",
):
    window_min, window_max, window_mask, windowed_wavenumber = _window_mask(
        wavenumber_values,
        window,
    )
    lorentz_real_abs, lorentz_real_pct = _absolute_and_percent_errors(
        true_eta.real,
        lorentz_eta.real,
    )
    lorentz_imag_abs, lorentz_imag_pct = _absolute_and_percent_errors(
        true_eta.imag,
        lorentz_eta.imag,
    )
    kk_real_abs, kk_real_pct = _absolute_and_percent_errors(
        true_eta.real,
        recovered_eta.real,
    )
    kk_imag_abs, kk_imag_pct = _absolute_and_percent_errors(
        true_eta.imag,
        recovered_eta.imag,
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(wavenumber_values, true_eta.real, label="True Re(eta)", linewidth=2)
    axes[0, 0].plot(wavenumber_values, true_eta.imag, label="True Im(eta)", linewidth=2)
    axes[0, 0].plot(
        wavenumber_values,
        lorentz_eta.real,
        label="Initial Joint Re(eta)",
        linewidth=2,
        linestyle=":",
    )
    axes[0, 0].plot(
        wavenumber_values,
        lorentz_eta.imag,
        label="Initial Joint Im(eta)",
        linewidth=2,
        linestyle=":",
    )
    axes[0, 0].plot(
        wavenumber_values,
        recovered_eta.real,
        label="Recovered KK Re(eta)",
        linewidth=2,
        linestyle="--",
    )
    axes[0, 0].plot(
        wavenumber_values,
        recovered_eta.imag,
        label="Recovered KK Im(eta)",
        linewidth=2,
        linestyle="--",
    )
    axes[0, 0].set_xlabel("Wavenumber")
    axes[0, 0].set_ylabel("Eta")
    axes[0, 0].set_title("Eta Comparison")
    axes[0, 0].legend()

    axes[0, 1].plot(
        windowed_wavenumber,
        lorentz_real_abs[window_mask],
        label="Initial Joint |Re(eta) Error|",
        linewidth=2,
        linestyle=":",
    )
    axes[0, 1].plot(
        windowed_wavenumber,
        lorentz_imag_abs[window_mask],
        label="Initial Joint |Im(eta) Error|",
        linewidth=2,
        linestyle=":",
    )
    axes[0, 1].plot(
        windowed_wavenumber,
        kk_real_abs[window_mask],
        label="KK |Re(eta) Error|",
        linewidth=2,
        linestyle="--",
    )
    axes[0, 1].plot(
        windowed_wavenumber,
        kk_imag_abs[window_mask],
        label="KK |Im(eta) Error|",
        linewidth=2,
        linestyle="--",
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[0, 1].set_xlabel("Wavenumber")
    axes[0, 1].set_ylabel("Absolute Error")
    axes[0, 1].set_title(f"Eta Absolute Error ({window_min:.0f} to {window_max:.0f})")
    axes[0, 1].legend()

    axes[1, 0].plot(
        windowed_wavenumber,
        lorentz_real_pct[window_mask],
        label="Initial Joint Re(eta) % Error",
        linewidth=2,
        linestyle=":",
    )
    axes[1, 0].plot(
        windowed_wavenumber,
        lorentz_imag_pct[window_mask],
        label="Initial Joint Im(eta) % Error",
        linewidth=2,
        linestyle=":",
    )
    axes[1, 0].plot(
        windowed_wavenumber,
        kk_real_pct[window_mask],
        label="KK Re(eta) % Error",
        linewidth=2,
        linestyle="--",
    )
    axes[1, 0].plot(
        windowed_wavenumber,
        kk_imag_pct[window_mask],
        label="KK Im(eta) % Error",
        linewidth=2,
        linestyle="--",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel("Wavenumber")
    axes[1, 0].set_ylabel("Percent Error")
    axes[1, 0].set_title(f"Eta Percentage Error ({window_min:.0f} to {window_max:.0f})")
    axes[1, 0].legend()

    axes[1, 1].plot(
        wavenumber_values,
        eta_residual_vector[: len(wavenumber_values)],
        label="Residual Re(eta)",
        linewidth=2,
    )
    axes[1, 1].plot(
        wavenumber_values,
        eta_residual_vector[len(wavenumber_values) :],
        label="Residual Im(eta)",
        linewidth=2,
    )
    axes[1, 1].set_xlabel("Wavenumber")
    axes[1, 1].set_ylabel("Eta Residual")
    axes[1, 1].set_title("Eta Residual Vector After KK Refinement")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_filename, dpi=200)
    plt.close(figure)


def plot_dielectric_summary(
    wavenumber_values,
    true_eps,
    lorentz_eps,
    recovered_eps,
    window,
    output_filename="multiple_oscillator_eps_summary.png",
):
    window_min, window_max, window_mask, windowed_wavenumber = _window_mask(
        wavenumber_values,
        window,
    )
    lorentz_real_abs, lorentz_real_pct = _absolute_and_percent_errors(
        true_eps.real,
        lorentz_eps.real,
    )
    lorentz_imag_abs, lorentz_imag_pct = _absolute_and_percent_errors(
        true_eps.imag,
        lorentz_eps.imag,
    )
    kk_real_abs, kk_real_pct = _absolute_and_percent_errors(
        true_eps.real,
        recovered_eps.real,
    )
    kk_imag_abs, kk_imag_pct = _absolute_and_percent_errors(
        true_eps.imag,
        recovered_eps.imag,
    )

    figure, axes = plt.subplots(3, 1, figsize=(12, 14))

    axes[0].plot(wavenumber_values, true_eps.real, label="True Re(eps)", linewidth=2)
    axes[0].plot(wavenumber_values, true_eps.imag, label="True Im(eps)", linewidth=2)
    axes[0].plot(
        wavenumber_values,
        lorentz_eps.real,
        label="Initial Joint Re(eps)",
        linewidth=2,
        linestyle=":",
    )
    axes[0].plot(
        wavenumber_values,
        lorentz_eps.imag,
        label="Initial Joint Im(eps)",
        linewidth=2,
        linestyle=":",
    )
    axes[0].plot(
        wavenumber_values,
        recovered_eps.real,
        label="Recovered KK Re(eps)",
        linewidth=2,
        linestyle="--",
    )
    axes[0].plot(
        wavenumber_values,
        recovered_eps.imag,
        label="Recovered KK Im(eps)",
        linewidth=2,
        linestyle="--",
    )
    axes[0].set_xlabel("Wavenumber")
    axes[0].set_ylabel("Dielectric Function")
    axes[0].set_title("Multiple Oscillator Dielectric Function")
    axes[0].legend()

    axes[1].plot(
        windowed_wavenumber,
        lorentz_real_abs[window_mask],
        label="Initial Joint |Re(eps) Error|",
        linewidth=2,
        linestyle=":",
    )
    axes[1].plot(
        windowed_wavenumber,
        lorentz_imag_abs[window_mask],
        label="Initial Joint |Im(eps) Error|",
        linewidth=2,
        linestyle=":",
    )
    axes[1].plot(
        windowed_wavenumber,
        kk_real_abs[window_mask],
        label="KK |Re(eps) Error|",
        linewidth=2,
        linestyle="--",
    )
    axes[1].plot(
        windowed_wavenumber,
        kk_imag_abs[window_mask],
        label="KK |Im(eps) Error|",
        linewidth=2,
        linestyle="--",
    )
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[1].set_xlabel("Wavenumber")
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title(
        f"Dielectric Absolute Error ({window_min:.0f} to {window_max:.0f})"
    )
    axes[1].legend()

    axes[2].plot(
        windowed_wavenumber,
        lorentz_real_pct[window_mask],
        label="Initial Joint Re(eps) % Error",
        linewidth=2,
        linestyle=":",
    )
    axes[2].plot(
        windowed_wavenumber,
        lorentz_imag_pct[window_mask],
        label="Initial Joint Im(eps) % Error",
        linewidth=2,
        linestyle=":",
    )
    axes[2].plot(
        windowed_wavenumber,
        kk_real_pct[window_mask],
        label="KK Re(eps) % Error",
        linewidth=2,
        linestyle="--",
    )
    axes[2].plot(
        windowed_wavenumber,
        kk_imag_pct[window_mask],
        label="KK Im(eps) % Error",
        linewidth=2,
        linestyle="--",
    )
    axes[2].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[2].set_xlabel("Wavenumber")
    axes[2].set_ylabel("Percent Error")
    axes[2].set_title(
        f"Dielectric Percentage Error ({window_min:.0f} to {window_max:.0f})"
    )
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_filename, dpi=200)
    plt.close(figure)


strength_multiple = long_phon_frequency ** 2 - trans_phon_frequency ** 2

eps_sub = (
    snompy.sample.lorentz_perm(
        wavenumber,
        nu_j=trans_phon_frequency,
        gamma_j=gamma,
        A_j=strength_multiple * eps_inf,
        eps_inf=eps_inf,
    )
    + snompy.sample.lorentz_perm(
        wavenumber,
        nu_j=trans_phon_frequency/1.03,
        gamma_j=gamma/1.03,
        A_j=strength_multiple * eps_inf /1.5,
        eps_inf=0,
    )
    + snompy.sample.drude_perm(
        drude_wavenumber_grid(wavenumber),
        nu_plasma=drude_plasma_frequency,
        gamma=drude_gamma,
        eps_inf=0.0,
    )
)

def add_noise(data):
    max = np.max(data)
    noise = scipy.stats.norm.rvs(loc=0, scale=max/64, size=len(data), random_state=None)

    return data + noise


def complex_eta_from_vector(eta_vector):
    half_length = len(eta_vector) // 2
    return eta_vector[:half_length] + 1j * eta_vector[half_length:]


def lorentz_eps_sub(params):
    amplitude_scale, nu_j, gamma_j, strength_multiple = params
    return snompy.sample.lorentz_perm(
        wavenumber,
        nu_j=nu_j,
        gamma_j=gamma_j,
        A_j=amplitude_scale * strength_multiple,
        eps_inf=0.0,
    )


def lorentz_parameter_bounds():
    strength_multiple_min, strength_multiple_max = compute_strength_multiple_bounds()
    lower_bounds = np.array(
        [
            np.min(eps_inf_range),
            np.min(trans_phon_frequency_range),
            np.min(gamma_ph_range),
            strength_multiple_min,
        ],
        dtype=np.float64,
    )
    upper_bounds = np.array(
        [
            np.max(eps_inf_range),
            np.max(trans_phon_frequency_range),
            np.max(gamma_ph_range),
            strength_multiple_max,
        ],
        dtype=np.float64,
    )
    return lower_bounds, upper_bounds


def fit_lorentz_term(target_eta_vector, baseline_eps_sub, initial_params):
    lower_bounds, upper_bounds = lorentz_parameter_bounds()
    initial_params = np.clip(initial_params, lower_bounds, upper_bounds)

    def residual(params):
        predicted_eps_sub = baseline_eps_sub + lorentz_eps_sub(params)
        return forward_function(predicted_eps_sub) - target_eta_vector

    initial_eta_vector = forward_function(baseline_eps_sub + lorentz_eps_sub(initial_params))
    result = least_squares(
        residual,
        initial_params,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        x_scale=upper_bounds - lower_bounds,
    )
    refined_eta_vector = forward_function(baseline_eps_sub + lorentz_eps_sub(result.x))
    return initial_params, result, initial_eta_vector, refined_eta_vector


lorentz_sample = snompy.sample.bulk_sample(eps_sub=eps_sub)


r = lorentz_sample.refl_coef(theta_in=theta_in)

fff_sample = (1 + c_r * r) ** 2

alpha_ref, fff_ref = alpha_fff_ref()

alpha_eff_sample = snompy.fdm.eff_pol_n(
    sample=lorentz_sample,
    A_tip=A,
    n=harmonic,
    z_tip=z
)



eta = (fff_sample * alpha_eff_sample)/(fff_ref * alpha_ref)

eta = add_noise(eta)

eta_vector = eta_to_vector(eta)
target_eta_vector = eta_vector.copy()


_, joint_optimization_result, _ = optimize_parameters(
    target_eta_vector,
    MODEL_PATH,
    wavenumber,
)
joint_eps_sub = epsilon_from_params(joint_optimization_result.x, wavenumber)

kk_start_time = perf_counter()
kk_optimization_result = kk_optimize(eta, joint_eps_sub)
kk_optimization_seconds = perf_counter() - kk_start_time

initial_joint_eps_sub = kk_optimization_result.eps_recovered
initial_joint_eta = complex_eta_from_vector(kk_optimization_result.eta_recovered)
joint_eta_residual_vector = target_eta_vector - kk_optimization_result.eta_recovered

recovered_eps_sub = initial_joint_eps_sub.copy()
lorentz_seed = joint_optimization_result.x[[0, 4, 1, 5]]
peak_fit_records = []

for iteration_index in range(NUM_FIT_ITERATIONS - 1):
    initial_params, optimization_result, initial_eta_vector, recovered_eta_vector = (
        fit_lorentz_term(target_eta_vector, recovered_eps_sub, lorentz_seed)
    )
    recovered_params = optimization_result.x
    recovered_eps_sub = recovered_eps_sub + lorentz_eps_sub(recovered_params)
    lorentz_seed = recovered_params

    peak_fit_records.append(
        {
            "peak_index": iteration_index + 2,
            "target_eta_vector": target_eta_vector,
            "initial_params": initial_params.copy(),
            "refined_params": recovered_params.copy(),
            "initial_eta_vector": initial_eta_vector,
            "refined_eta_vector": recovered_eta_vector,
            "initial_residual_norm": float(
                np.linalg.norm(initial_eta_vector - target_eta_vector)
            ),
            "refined_residual_norm": float(
                np.linalg.norm(recovered_eta_vector - target_eta_vector)
            ),
        }
    )

plot_peak_fit_progress(wavenumber, peak_fit_records)

recovered_eta_vector = forward_function(recovered_eps_sub)
recovered_eta = complex_eta_from_vector(recovered_eta_vector)
eta_residual_vector = target_eta_vector - recovered_eta_vector

plot_dielectric_summary(
    wavenumber,
    eps_sub,
    initial_joint_eps_sub,
    recovered_eps_sub,
    ERROR_WINDOW,
)

plot_eta_summary(
    wavenumber,
    eta,
    initial_joint_eta,
    recovered_eta,
    eta_residual_vector,
    ERROR_WINDOW,
)


print(f"Initial joint residual norm: {np.linalg.norm(joint_eta_residual_vector):.6e}")
print(f"Final residual norm: {np.linalg.norm(eta_residual_vector):.6e}")
print(f"KK optimization time: {kk_optimization_seconds:.6f} s")
print(f"KK optimizer status: {kk_optimization_result.status}")
print(eta_residual_vector.shape)
