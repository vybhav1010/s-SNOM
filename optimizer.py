import numpy as np
import snompy
import torch
from scipy.optimize import least_squares
from time import perf_counter
import matplotlib.pyplot as plt
import sys

from conv_network_2 import ConvNetwork, MODEL_PATH
from data_gen import (
    A,
    alpha_fff_ref,
    c_r,
    eps_inf_range,
    gamma_range,
    harmonic,
    long_phon_frequency_stop,
    long_phon_frequency_step,
    n_wav,
    theta_in,
    trans_phon_frequency_range,
    wavenumber,
    z,
)




def eta_to_vector(eta):
    return np.concatenate((np.real(eta), np.imag(eta)))


def epsilon_from_params(params, wavenumbers):
    eps_inf, trans_phon_freq, gamma, strength_multiple = params
    strength = eps_inf * strength_multiple

    return snompy.sample.lorentz_perm(
        wavenumbers,
        nu_j=trans_phon_freq,
        gamma_j=gamma,
        A_j=strength,
        eps_inf=eps_inf,
    )

def forward_function(eps_sub):
    eps_sub = np.asarray(eps_sub, dtype=np.complex128)
    lorentz_sample = snompy.sample.bulk_sample(eps_sub=eps_sub)
    sample_r = lorentz_sample.refl_coef(theta_in=theta_in)
    far_field_sample = (1 + c_r * sample_r) ** 2
    alpha_ref, far_field_ref = alpha_fff_ref()
    alpha_eff_sample = snompy.fdm.eff_pol_n(
        sample=lorentz_sample,
        A_tip=A,
        n=harmonic,
        z_tip=z,
    )
    eta = (far_field_sample * alpha_eff_sample) / (far_field_ref * alpha_ref)
    return eta_to_vector(eta)


def forward_function_from_params(params, wavenumbers):
    eps_sub = epsilon_from_params(params, wavenumbers)
    return forward_function(eps_sub)


def residual(params, wavenumbers, expected_eta_vector):
    return forward_function_from_params(params, wavenumbers) - expected_eta_vector


def residual_subset(
    subset_params,
    base_params,
    subset_indices,
    wavenumbers,
    expected_eta_vector,
):
    full_params = np.array(base_params, dtype=np.float64, copy=True)
    full_params[subset_indices] = subset_params
    return residual(full_params, wavenumbers, expected_eta_vector)


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = ConvNetwork(n_wav * 2)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        label_mean = checkpoint["label_mean"].cpu().numpy()
        label_std = checkpoint["label_std"].cpu().numpy()
    else:
        model.load_state_dict(checkpoint)
        raise ValueError(
            "Checkpoint is missing label normalization stats. Retrain conv_network_2.py "
            "to create a normalized-label checkpoint."
        )
    model.eval()
    return model, label_mean, label_std


def neural_net_initial_guess(model, eta_vector):
    expected_input_dim = n_wav * 2
    if eta_vector.shape != (expected_input_dim,):
        raise ValueError(
            "eta_vector length does not match the current model input size. "
            f"Expected shape ({expected_input_dim},), got {eta_vector.shape}. "
            "Make sure the evaluation grid uses the same n_wav as data_gen.py "
            "and the trained checkpoint."
        )
    spectrum_tensor = torch.from_numpy(eta_vector.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        prediction = model(spectrum_tensor).squeeze(0).cpu().numpy()

    eps_inf = float(prediction[0])
    gamma = float(prediction[1])
    trans_phon_freq = float(prediction[2])
    strength_multiple = float(prediction[3])

    return np.array([eps_inf, trans_phon_freq, gamma, strength_multiple], dtype=np.float64)


def denormalize_prediction(prediction, label_mean, label_std):
    return prediction * label_std + label_mean


def compute_strength_multiple_bounds():
    min_trans_phon_freq = float(np.min(trans_phon_frequency_range))
    strength_multiple_min = ((min_trans_phon_freq + 10) ** 2) - (min_trans_phon_freq ** 2)
    strength_multiple_max = (long_phon_frequency_stop ** 2) - (min_trans_phon_freq ** 2)
    return float(strength_multiple_min), float(strength_multiple_max)


def every_other_midpoints(values):
    values = np.asarray(values, dtype=np.float64)
    midpoints = 0.5 * (values[:-1] + values[1:])
    return midpoints[1::2]


def build_intermediate_test_cases():
    eps_inf_values = every_other_midpoints(eps_inf_range)
    gamma_values = every_other_midpoints(gamma_range)
    trans_phon_freq_values = every_other_midpoints(trans_phon_frequency_range)
    long_phon_training_values = np.arange(
        float(np.min(trans_phon_frequency_range)) + 10.0,
        long_phon_frequency_stop + 1,
        long_phon_frequency_step,
        dtype=np.float64,
    )
    long_phon_values = every_other_midpoints(long_phon_training_values)

    test_cases = []
    for eps_inf in eps_inf_values:
        for gamma in gamma_values:
            for trans_phon_freq in trans_phon_freq_values:
                valid_long_phon_values = long_phon_values[long_phon_values > trans_phon_freq]
                for long_phon_freq in valid_long_phon_values:
                    strength_multiple = (long_phon_freq ** 2) - (trans_phon_freq ** 2)
                    test_cases.append(
                        np.array(
                            [eps_inf, trans_phon_freq, gamma, strength_multiple],
                            dtype=np.float64,
                        )
                    )

    return test_cases


def parameter_bounds():
    strength_multiple_min, strength_multiple_max = compute_strength_multiple_bounds()
    lower_bounds = np.array(
        [
            float(np.min(eps_inf_range)),
            float(np.min(trans_phon_frequency_range)),
            float(np.min(gamma_range)),
            float(strength_multiple_min),
        ],
        dtype=np.float64,
    )
    upper_bounds = np.array(
        [
            float(np.max(eps_inf_range)),
            float(np.max(trans_phon_frequency_range)),
            float(np.max(gamma_range)),
            float(strength_multiple_max),
        ],
        dtype=np.float64,
    )
    return lower_bounds, upper_bounds


def optimize_parameters(eta_vector, model_path, wavenumbers=None):
    if wavenumbers is None:
        wavenumbers = wavenumber

    model, label_mean, label_std = load_model(model_path)
    normalized_guess = neural_net_initial_guess(model, eta_vector)

    initial_guess_raw = denormalize_prediction(normalized_guess, label_mean, label_std)

    eps_inf = float(initial_guess_raw[0])
    gamma = float(initial_guess_raw[1])
    trans_phon_freq = float(initial_guess_raw[2])
    strength_multiple = float(initial_guess_raw[3])

    initial_guess = np.array([eps_inf, trans_phon_freq, gamma, strength_multiple], dtype=np.float64)
    lower_bounds, upper_bounds = parameter_bounds()
    clipped_guess = np.clip(initial_guess, lower_bounds, upper_bounds)
    initial_residual = residual(clipped_guess, wavenumbers, eta_vector)
    initial_residual_norm = float(np.linalg.norm(initial_residual))

    start_time = perf_counter()
    # peak_indices = np.array([1, 2], dtype=np.int64)
    # peak_alignment_result = least_squares(
    #     residual_subset,
    #     clipped_guess[peak_indices],
    #     args=(clipped_guess, peak_indices, wavenumbers, eta_vector),
    #     bounds=(lower_bounds[peak_indices], upper_bounds[peak_indices]),
    #     method="trf",
    #     x_scale=upper_bounds[peak_indices] - lower_bounds[peak_indices],
    # )

    best_trans_phon_freq = clipped_guess[1]
    best_residual_norm = initial_residual_norm
    test_params = clipped_guess.copy()
    for trans_phon_freq in range(int(trans_phon_frequency_range[0]), int(trans_phon_frequency_range[-1]) + 1):
        
        test_params[1] = trans_phon_freq

        if(np.linalg.norm(residual(test_params, wavenumbers, eta_vector)) < best_residual_norm):
            best_trans_phon_freq = trans_phon_freq
            best_residual_norm = np.linalg.norm(residual(test_params, wavenumbers, eta_vector))

    
    peak_aligned_guess = clipped_guess.copy()
    peak_aligned_guess[1] = best_trans_phon_freq


    result = least_squares(
        residual,
        peak_aligned_guess,
        args=(wavenumbers, eta_vector),
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        x_scale=upper_bounds - lower_bounds,
    )
    optimization_seconds = perf_counter() - start_time
    peak_alignment_residual = residual(peak_aligned_guess, wavenumbers, eta_vector)
    peak_alignment_residual_norm = float(np.linalg.norm(peak_alignment_residual))
    optimized_residual_norm = float(np.linalg.norm(result.fun))

    if initial_residual_norm <= optimized_residual_norm:
        result.x = clipped_guess.copy()
        result.fun = initial_residual.copy()
        result.cost = 0.5 * float(np.dot(initial_residual, initial_residual))
        result.optimality = np.nan
        result.nfev = 0
        result.njev = 0
        result.status = 0
        result.success = True
        result.message = (
            "Kept neural-network initial guess because it matched eta better "
            "than the TRF result."
        )

    result.initial_residual_norm = initial_residual_norm
    result.peak_aligned_guess = peak_aligned_guess.copy()
    result.peak_alignment_residual_norm = peak_alignment_residual_norm
    result.optimized_residual_norm = optimized_residual_norm

    return clipped_guess, result, optimization_seconds


def evaluate_optimizer_case(true_params, model_path, wavenumbers=None):
    if wavenumbers is None:
        wavenumbers = wavenumber

    eta_vector = forward_function_from_params(true_params, wavenumbers)
    initial_guess, result, optimization_seconds = optimize_parameters(
        eta_vector, model_path, wavenumbers=wavenumbers
    )

    initial_percent_error = 100.0 * np.abs(initial_guess - true_params) / np.abs(true_params)
    recovered_percent_error = 100.0 * np.abs(result.x - true_params) / np.abs(true_params)

    return {
        "true_params": true_params,
        "initial_guess": initial_guess,
        "peak_aligned_guess": result.peak_aligned_guess,
        "recovered_params": result.x,
        "initial_percent_error": initial_percent_error,
        "recovered_percent_error": recovered_percent_error,
        "initial_residual_norm": float(result.initial_residual_norm),
        "peak_alignment_residual_norm": float(result.peak_alignment_residual_norm),
        "optimized_residual_norm": float(result.optimized_residual_norm),
        "residual_norm": float(np.linalg.norm(result.fun)),
        "optimization_seconds": float(optimization_seconds),
        "optimizer_message": str(result.message),
        "used_initial_guess": bool(result.nfev == 0),
    }


def plot_spectrum_comparison(
    true_params,
    initial_params,
    peak_aligned_params,
    recovered_params,
    output_filename="optimizer_spectrum_comparison.png",
    relative_error_output_filename="optimizer_spectrum_relative_error.png",
    wavenumbers=None,
):
    if wavenumbers is None:
        wavenumbers = wavenumber

    true_eta = forward_function_from_params(true_params, wavenumbers)
    initial_eta = forward_function_from_params(initial_params, wavenumbers)
    peak_aligned_eta = forward_function_from_params(peak_aligned_params, wavenumbers)
    recovered_eta = forward_function_from_params(recovered_params, wavenumbers)

    half_length = len(true_eta) // 2
    true_real = true_eta[:half_length]
    true_imag = true_eta[half_length:]
    initial_real = initial_eta[:half_length]
    initial_imag = initial_eta[half_length:]
    peak_aligned_real = peak_aligned_eta[:half_length]
    peak_aligned_imag = peak_aligned_eta[half_length:]
    recovered_real = recovered_eta[:half_length]
    recovered_imag = recovered_eta[half_length:]

    plt.figure(figsize=(10, 6))
    plt.plot(wavenumbers, true_real, label="True Real", linewidth=2)
    plt.plot(wavenumbers, initial_real, label="Initial Real", linewidth=2, alpha=0.8, linestyle=":")
    plt.plot(
        wavenumbers,
        peak_aligned_real,
        label="Omega_TO-shifted Real",
        linewidth=2,
        alpha=0.8,
        linestyle="-.",
    )
    plt.plot(wavenumbers, recovered_real, label="Recovered Real", linewidth=2, alpha=0.8)
    plt.plot(wavenumbers, true_imag, label="True Imag", linewidth=2)
    plt.plot(wavenumbers, initial_imag, label="Initial Imag", linewidth=2, alpha=0.8, linestyle=":")
    plt.plot(
        wavenumbers,
        peak_aligned_imag,
        label="Omega_TO-shifted Imag",
        linewidth=2,
        alpha=0.8,
        linestyle="-.",
    )
    plt.plot(wavenumbers, recovered_imag, label="Recovered Imag", linewidth=2, alpha=0.8)
    plt.xlabel("Wavenumber")
    plt.ylabel("Eta")
    plt.title("Eta Signal vs Initial, Omega_TO-Shifted, and Recovered Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

    denom_floor = 1e-12
    initial_real_relative_error = 100.0 * (initial_real - true_real) / np.maximum(
        np.abs(true_real), denom_floor
    )
    initial_imag_relative_error = 100.0 * (initial_imag - true_imag) / np.maximum(
        np.abs(true_imag), denom_floor
    )
    peak_aligned_real_relative_error = 100.0 * (peak_aligned_real - true_real) / np.maximum(
        np.abs(true_real), denom_floor
    )
    peak_aligned_imag_relative_error = 100.0 * (peak_aligned_imag - true_imag) / np.maximum(
        np.abs(true_imag), denom_floor
    )
    real_relative_error = 100.0 * (recovered_real - true_real) / np.maximum(
        np.abs(true_real), denom_floor
    )
    imag_relative_error = 100.0 * (recovered_imag - true_imag) / np.maximum(
        np.abs(true_imag), denom_floor
    )

    plt.figure(figsize=(10, 6))
    plt.plot(wavenumbers, initial_real_relative_error, label="Initial Real % Error", linewidth=2, linestyle=":")
    plt.plot(wavenumbers, initial_imag_relative_error, label="Initial Imag % Error", linewidth=2, linestyle=":")
    plt.plot(
        wavenumbers,
        peak_aligned_real_relative_error,
        label="Omega_TO-shifted Real % Error",
        linewidth=2,
        linestyle="-.",
    )
    plt.plot(
        wavenumbers,
        peak_aligned_imag_relative_error,
        label="Omega_TO-shifted Imag % Error",
        linewidth=2,
        linestyle="-.",
    )
    plt.plot(wavenumbers, real_relative_error, label="Real % Error", linewidth=2)
    plt.plot(wavenumbers, imag_relative_error, label="Imag % Error", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.xlabel("Wavenumber")
    plt.ylabel("Relative Error (%)")
    plt.title("Initial, Omega_TO-Shifted, and Recovered Spectrum Relative Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(relative_error_output_filename)
    plt.close()


def run_intermediate_value_test_loop(model_path):
    test_cases = build_intermediate_test_cases()
    evaluations = [evaluate_optimizer_case(params, model_path) for params in test_cases]

    initial_percent_errors = np.array(
        [evaluation["initial_percent_error"] for evaluation in evaluations]
    )
    recovered_percent_errors = np.array(
        [evaluation["recovered_percent_error"] for evaluation in evaluations]
    )
    optimization_seconds = np.array(
        [evaluation["optimization_seconds"] for evaluation in evaluations]
    )
    residual_norms = np.array([evaluation["residual_norm"] for evaluation in evaluations])
    initial_residual_norms = np.array(
        [evaluation["initial_residual_norm"] for evaluation in evaluations]
    )
    peak_alignment_residual_norms = np.array(
        [evaluation["peak_alignment_residual_norm"] for evaluation in evaluations]
    )
    optimized_residual_norms = np.array(
        [evaluation["optimized_residual_norm"] for evaluation in evaluations]
    )
    used_initial_guess_count = sum(
        evaluation["used_initial_guess"] for evaluation in evaluations
    )

    print("Intermediate-value test grid:")
    print(f"  eps_inf: {every_other_midpoints(eps_inf_range)}")
    print(f"  gamma: {every_other_midpoints(gamma_range)}")
    print(f"  trans_phon_frequency: {every_other_midpoints(trans_phon_frequency_range)}")
    print(f"  cases: {len(evaluations)}")
    print("Mean percent error by quantity:")
    print(f"  initial eps_inf: {initial_percent_errors[:, 0].mean():.2f}%")
    print(f"  initial trans_phon_frequency: {initial_percent_errors[:, 1].mean():.2f}%")
    print(f"  initial gamma: {initial_percent_errors[:, 2].mean():.2f}%")
    print(f"  initial strength_multiple: {initial_percent_errors[:, 3].mean():.2f}%")
    print(f"  recovered eps_inf: {recovered_percent_errors[:, 0].mean():.2f}%")
    print(f"  recovered trans_phon_frequency: {recovered_percent_errors[:, 1].mean():.2f}%")
    print(f"  recovered gamma: {recovered_percent_errors[:, 2].mean():.2f}%")
    print(f"  recovered strength_multiple: {recovered_percent_errors[:, 3].mean():.2f}%")
    print("Worst recovered case:")
    worst_index = int(np.argmax(recovered_percent_errors.mean(axis=1)))
    worst_case = evaluations[worst_index]
    print(f"  true params: {worst_case['true_params']}")
    print(f"  initial guess: {worst_case['initial_guess']}")
    print(f"  recovered params: {worst_case['recovered_params']}")
    print(f"  recovered percent error: {worst_case['recovered_percent_error']}")
    median_index = int(np.argsort(recovered_percent_errors.mean(axis=1))[len(evaluations) // 2])
    median_case = evaluations[median_index]
    print("Median recovered case:")
    print(f"  true params: {median_case['true_params']}")
    print(f"  initial guess: {median_case['initial_guess']}")
    print(f"  recovered params: {median_case['recovered_params']}")
    print(f"  recovered percent error: {median_case['recovered_percent_error']}")
    print("Optimization timing:")
    print(f"  mean seconds: {optimization_seconds.mean():.4f}")
    print(f"  max seconds: {optimization_seconds.max():.4f}")
    print("Fallback usage:")
    print(f"  kept initial guess: {used_initial_guess_count}/{len(evaluations)}")
    print("Residual norms:")
    print(f"  initial mean: {initial_residual_norms.mean():.6e}")
    print(f"  peak-align mean: {peak_alignment_residual_norms.mean():.6e}")
    print(f"  trf mean: {optimized_residual_norms.mean():.6e}")
    print(f"  mean: {residual_norms.mean():.6e}")
    print(f"  max: {residual_norms.max():.6e}")

    plot_spectrum_comparison(
        worst_case["true_params"],
        worst_case["initial_guess"],
        worst_case["peak_aligned_guess"],
        worst_case["recovered_params"],
        output_filename="optimizer_spectrum_comparison_worst.png",
        relative_error_output_filename="optimizer_spectrum_relative_error_worst.png",
    )
    plot_spectrum_comparison(
        median_case["true_params"],
        median_case["initial_guess"],
        median_case["peak_aligned_guess"],
        median_case["recovered_params"],
        output_filename="optimizer_spectrum_comparison_median.png",
        relative_error_output_filename="optimizer_spectrum_relative_error_median.png",
    )
    print("Saved spectrum comparison plots:")
    print("  optimizer_spectrum_comparison_worst.png")
    print("  optimizer_spectrum_comparison_median.png")
    print("Saved relative error plots:")
    print("  optimizer_spectrum_relative_error_worst.png")
    print("  optimizer_spectrum_relative_error_median.png")


    return evaluations


def main():
    model_path = MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected a trained model checkpoint at {model_path}."
        )

    run_intermediate_value_test_loop(model_path)


if __name__ == "__main__":
    main()
