"""Evaluate the multi-peak fitting and KK-refinement pipeline.

Each synthetic target has one Drude term and between one and ``MAX_PEAKS``
Lorentz terms.  Every Lorentz parameter tuple is sampled from the same
discrete parameter axes used to create the training data.  For each candidate
peak count, the fitter jointly optimizes the shared background and Drude
parameters plus every Lorentz-peak parameter block.  The pre-KK candidate
with the smallest eta residual is used as the sole input to KK refinement.
"""

from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import snompy
from scipy.optimize import least_squares

from conv_network_2 import MODEL_PATH
from data_gen import (
    eps_inf_range,
    gamma_ph_range,
    gamma_p_range,
    long_phon_frequency_step,
    long_phon_frequency_stop,
    plas_freq_range,
    trans_phon_frequency_range,
    wavenumber,
)
from kk_optimizer import kk_optimize
from optimizer import (
    compute_strength_multiple_bounds,
    every_other_midpoints,
    epsilon_from_params,
    forward_function,
    optimize_parameters,
)


# Evaluation controls.
MAX_PEAKS = 5
SAMPLES_PER_PEAK_COUNT = 10
RANDOM_SEED = round(1e32 * np.random.random())
NOISE_STD_FRACTION = 1/32
EXAMPLE_PEAK_COUNT = 5

RESULTS_PATH = Path("full_test_results.npz")
PLOT_FILENAMES = {
    "best": Path("full_test_best.png"),
    "median": Path("full_test_median.png"),
    "worst": Path("full_test_worst.png"),
}
EXAMPLE_PLOT_FILENAME = Path("full_test_example.png")

PARAMETER_NAMES = (
    "eps_inf",
    "gamma_ph",
    "gamma_p",
    "plas_freq",
    "trans_phon_frequency",
    "strength_multiple",
)


def complex_eta_from_vector(eta_vector):
    eta_vector = np.asarray(eta_vector, dtype=np.float64)
    half_length = len(eta_vector) // 2
    return eta_vector[:half_length] + 1j * eta_vector[half_length:]


def format_params(params):
    return ", ".join(
        f"{name}={value:.4g}" for name, value in zip(PARAMETER_NAMES, params)
    )


def strength_multiple_values(trans_phon_frequency):
    """Return the valid strength values for one training-grid frequency."""
    long_phon_frequencies = np.arange(
        trans_phon_frequency + 10.0,
        long_phon_frequency_stop + 1.0,
        long_phon_frequency_step,
        dtype=np.float64,
    )
    return long_phon_frequencies**2 - trans_phon_frequency**2


def sample_training_params(rng):
    """Sample one valid six-parameter tuple from the training-data grid."""
    eps_inf = float(rng.choice(eps_inf_range))
    gamma_ph = float(rng.choice(gamma_ph_range))
    gamma_p = float(rng.choice(gamma_p_range))
    plas_freq = float(rng.choice(plas_freq_range))
    trans_phon_frequency = float(rng.choice(trans_phon_frequency_range))
    strength_multiple = float(
        rng.choice(strength_multiple_values(trans_phon_frequency))
    )
    return np.array(
        [
            eps_inf,
            gamma_ph,
            gamma_p,
            plas_freq,
            trans_phon_frequency,
            strength_multiple,
        ],
        dtype=np.float64,
    )


def sample_midpoint_params(rng):
    """Sample one valid parameter tuple from the held-out midpoint grid."""
    eps_inf_values = every_other_midpoints(eps_inf_range)
    gamma_ph_values = every_other_midpoints(gamma_ph_range)
    gamma_p_values = every_other_midpoints(gamma_p_range)
    plas_freq_values = every_other_midpoints(plas_freq_range)
    trans_phon_frequency_values = every_other_midpoints(trans_phon_frequency_range)
    long_phon_training_values = np.arange(
        float(np.min(trans_phon_frequency_range)) + 10.0,
        long_phon_frequency_stop + 1.0,
        long_phon_frequency_step,
        dtype=np.float64,
    )
    long_phon_frequency_values = every_other_midpoints(long_phon_training_values)

    trans_phon_frequency = float(
        trans_phon_frequency_values[
            rng.integers(len(trans_phon_frequency_values))
        ]
    )
    valid_long_phon_frequency_values = long_phon_frequency_values[
        long_phon_frequency_values > trans_phon_frequency
    ]
    if valid_long_phon_frequency_values.size == 0:
        raise ValueError(
            "The midpoint grid has no valid longitudinal phonon frequency for "
            f"transverse frequency {trans_phon_frequency}."
        )
    long_phon_frequency = float(
        valid_long_phon_frequency_values[
            rng.integers(len(valid_long_phon_frequency_values))
        ]
    )
    strength_multiple = long_phon_frequency**2 - trans_phon_frequency**2

    return np.array(
        [
            eps_inf_values[rng.integers(len(eps_inf_values))],
            gamma_ph_values[rng.integers(len(gamma_ph_values))],
            gamma_p_values[rng.integers(len(gamma_p_values))],
            plas_freq_values[rng.integers(len(plas_freq_values))],
            trans_phon_frequency,
            strength_multiple,
        ],
        dtype=np.float64,
    )
def lorentz_eps_sub(params):
    """Create one added Lorentz term from a training-parameter tuple."""
    eps_inf, gamma_ph, _, _, trans_phon_frequency, strength_multiple = params
    return snompy.sample.lorentz_perm(
        wavenumber,
        nu_j=trans_phon_frequency,
        gamma_j=gamma_ph,
        A_j=eps_inf * strength_multiple,
        eps_inf=0.0,
    )


def build_multi_peak_epsilon(base_params, additional_peak_params):
    """Build one Drude term plus the requested number of Lorentz terms."""
    eps_sub = epsilon_from_params(base_params, wavenumber)
    for peak_params in additional_peak_params:
        eps_sub = eps_sub + lorentz_eps_sub(peak_params)
    return eps_sub


def add_noise(eta_vector, rng):
    if NOISE_STD_FRACTION == 0.0:
        return eta_vector.copy()

    scale = NOISE_STD_FRACTION * float(np.max(np.abs(eta_vector)))
    return eta_vector + rng.normal(0.0, scale, size=eta_vector.shape)


def joint_parameter_bounds(peak_count):
    """Bounds for a shared background, Drude term, and ``peak_count`` Lorentz terms."""
    strength_multiple_min, strength_multiple_max = compute_strength_multiple_bounds()
    shared_lower_bounds = [
        float(np.min(eps_inf_range)),
        float(np.min(gamma_p_range)),
        float(np.min(plas_freq_range)),
    ]
    shared_upper_bounds = [
        float(np.max(eps_inf_range)),
        float(np.max(gamma_p_range)),
        float(np.max(plas_freq_range)),
    ]
    peak_lower_bounds = [
        float(np.min(eps_inf_range)),
        float(np.min(trans_phon_frequency_range)),
        float(np.min(gamma_ph_range)),
        strength_multiple_min,
    ]
    peak_upper_bounds = [
        float(np.max(eps_inf_range)),
        float(np.max(trans_phon_frequency_range)),
        float(np.max(gamma_ph_range)),
        strength_multiple_max,
    ]
    return (
        np.array(shared_lower_bounds + peak_lower_bounds * peak_count, dtype=np.float64),
        np.array(shared_upper_bounds + peak_upper_bounds * peak_count, dtype=np.float64),
    )


def joint_epsilon_from_params(params, peak_count):
    """Build a shared-background Drude model with jointly fitted Lorentz peaks.

    The parameter layout is ``[eps_inf, gamma_p, plas_freq,
    amplitude_1, nu_1, gamma_1, strength_1, ...]``.
    """
    params = np.asarray(params, dtype=np.float64)
    expected_size = 3 + 4 * peak_count
    if params.shape != (expected_size,):
        raise ValueError(
            f"Expected {expected_size} joint parameters for {peak_count} peaks, "
            f"got {params.shape}."
        )

    background_eps_inf, gamma_p, plas_freq = params[:3]
    eps_sub = np.full(wavenumber.shape, background_eps_inf, dtype=np.complex128)
    for peak_index in range(peak_count):
        amplitude_scale, nu_j, gamma_j, strength_multiple = params[
            3 + 4 * peak_index : 7 + 4 * peak_index
        ]
        eps_sub = eps_sub + fitted_lorentz_eps(
            amplitude_scale,
            nu_j,
            gamma_j,
            strength_multiple,
        )
    eps_sub = eps_sub + snompy.sample.drude_perm(
        wavenumber,
        nu_plasma=plas_freq,
        gamma=gamma_p,
        eps_inf=0.0,
    )
    return eps_sub


def fitted_lorentz_eps(amplitude_scale, nu_j, gamma_j, strength_multiple):
    return snompy.sample.lorentz_perm(
        wavenumber,
        nu_j=nu_j,
        gamma_j=gamma_j,
        A_j=amplitude_scale * strength_multiple,
        eps_inf=0.0,
    )


def joint_seed_from_single_peak_params(params):
    """Convert the optimizer's six single-peak parameters to the joint layout."""
    eps_inf, gamma_ph, gamma_p, plas_freq, nu_j, strength_multiple = params
    return np.array(
        [
            eps_inf,
            gamma_p,
            plas_freq,
            eps_inf,
            nu_j,
            gamma_ph,
            strength_multiple,
        ],
        dtype=np.float64,
    )


def jointly_fit_peaks(target_eta_vector, initial_params, peak_count):
    lower_bounds, upper_bounds = joint_parameter_bounds(peak_count)
    initial_params = np.clip(initial_params, lower_bounds, upper_bounds)

    def residual(params):
        return forward_function(joint_epsilon_from_params(params, peak_count)) - target_eta_vector

    result = least_squares(
        residual,
        initial_params,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        x_scale=upper_bounds - lower_bounds,
    )
    fitted_eps_sub = joint_epsilon_from_params(result.x, peak_count)
    fitted_eta_vector = forward_function(fitted_eps_sub)
    return result, fitted_eps_sub, fitted_eta_vector


def fit_case(true_eps_sub, true_peak_count, target_eta_vector, model_path):
    """Run single-peak initialization, joint multi-peak fitting, then KK."""
    joint_start_time = perf_counter()
    _, single_peak_result, _ = optimize_parameters(
        target_eta_vector,
        model_path,
        wavenumber,
    )
    current_params = joint_seed_from_single_peak_params(single_peak_result.x)

    best_peak_count = 1
    best_pre_kk_eps_sub = None
    best_pre_kk_eta_vector = None
    best_pre_kk_residual_norm = np.inf
    joint_eps_sub = None
    joint_eta_vector = None

    for peak_count in range(1, MAX_PEAKS + 1):
        print(peak_count)
        if peak_count > 1:
            current_params = np.concatenate((current_params, current_params[-4:]))
        joint_result, current_eps_sub, current_eta_vector = jointly_fit_peaks(
            target_eta_vector,
            current_params,
            peak_count,
        )
        current_params = joint_result.x
        residual_norm = float(np.linalg.norm(target_eta_vector - current_eta_vector))

        if residual_norm < best_pre_kk_residual_norm:
            best_peak_count = peak_count
            best_pre_kk_eps_sub = current_eps_sub.copy()
            best_pre_kk_eta_vector = current_eta_vector.copy()
            best_pre_kk_residual_norm = residual_norm

        if peak_count == 1:
            joint_eps_sub = current_eps_sub.copy()
            joint_eta_vector = current_eta_vector.copy()

    joint_seconds = perf_counter() - joint_start_time

    kk_start_time = perf_counter()
    kk_result = kk_optimize(target_eta_vector, best_pre_kk_eps_sub)
    kk_seconds = perf_counter() - kk_start_time
    final_eps_sub = kk_result.eps_recovered
    final_eta_vector = kk_result.eta_recovered

    return {
        "true_eps_sub": true_eps_sub,
        "true_peak_count": true_peak_count,
        "joint_eps_sub": joint_eps_sub,
        "joint_eta_vector": joint_eta_vector,
        "joint_residual_norm": float(np.linalg.norm(target_eta_vector - joint_eta_vector)),
        "pre_kk_eps_sub": best_pre_kk_eps_sub,
        "pre_kk_eta_vector": best_pre_kk_eta_vector,
        "pre_kk_residual_norm": best_pre_kk_residual_norm,
        "selected_total_peak_count": best_peak_count,
        "final_eps_sub": final_eps_sub,
        "final_eta_vector": final_eta_vector,
        "final_residual_norm": float(np.linalg.norm(target_eta_vector - final_eta_vector)),
        "joint_seconds": joint_seconds,
        "kk_seconds": kk_seconds,
        "kk_status": int(kk_result.status),
        "kk_nfev": int(kk_result.nfev),
        "kk_message": str(kk_result.message),
    }


def generate_cases(rng):
    """Yield test cases spanning all requested true Lorentz-peak counts."""
    for true_peak_count in range(1, MAX_PEAKS + 1):
        for _ in range(SAMPLES_PER_PEAK_COUNT):
            base_params = sample_training_params(rng)
            additional_peak_params = np.array(
                [sample_training_params(rng) for _ in range(true_peak_count - 1)],
                dtype=np.float64,
            )
            yield true_peak_count, base_params, additional_peak_params


def plot_case(case, output_filename, label):
    true_eta = complex_eta_from_vector(case["target_eta_vector"])
    joint_eta = complex_eta_from_vector(case["joint_eta_vector"])
    pre_kk_eta = complex_eta_from_vector(case["pre_kk_eta_vector"])
    final_eta = complex_eta_from_vector(case["final_eta_vector"])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
    figure.suptitle(
        f"{label.title()} final residual case | "
        f"true peaks={case['true_peak_count']}, "
        f"selected peaks={case['selected_total_peak_count']}, "
        f"final norm={case['final_residual_norm']:.4e}\n"
        f"joint-fit time={case['joint_seconds']:.2f} s"
    )

    eta_traces = (
        (true_eta.real, "Target", "black", "-"),
        (joint_eta.real, "Joint fit", "tab:blue", ":"),
        (pre_kk_eta.real, "Pre-KK", "tab:orange", "--"),
        (final_eta.real, "Final KK", "tab:green", "-"),
    )
    for values, trace_label, color, linestyle in eta_traces:
        axes[0, 0].plot(wavenumber, values, label=trace_label, color=color, linestyle=linestyle)
    axes[0, 0].set_title("Re(eta)")
    axes[0, 0].set_ylabel("Eta")
    axes[0, 0].legend()

    eta_imag_traces = (
        (true_eta.imag, "Target", "black", "-"),
        (joint_eta.imag, "Joint fit", "tab:blue", ":"),
        (pre_kk_eta.imag, "Pre-KK", "tab:orange", "--"),
        (final_eta.imag, "Final KK", "tab:green", "-"),
    )
    for values, trace_label, color, linestyle in eta_imag_traces:
        axes[0, 1].plot(wavenumber, values, label=trace_label, color=color, linestyle=linestyle)
    axes[0, 1].set_title("Im(eta)")
    axes[0, 1].legend()

    eps_traces = (
        (case["true_eps_sub"].real, "True", "black", "-"),
        (case["joint_eps_sub"].real, "Joint fit", "tab:blue", ":"),
        (case["pre_kk_eps_sub"].real, "Pre-KK", "tab:orange", "--"),
        (case["final_eps_sub"].real, "Final KK", "tab:green", "-"),
    )
    for values, trace_label, color, linestyle in eps_traces:
        axes[1, 0].plot(wavenumber, values, label=trace_label, color=color, linestyle=linestyle)
    axes[1, 0].set_title("Re(epsilon)")
    axes[1, 0].set_xlabel("Wavenumber")
    axes[1, 0].set_ylabel("Dielectric function")
    axes[1, 0].legend()

    eps_imag_traces = (
        (case["true_eps_sub"].imag, "True", "black", "-"),
        (case["joint_eps_sub"].imag, "Joint fit", "tab:blue", ":"),
        (case["pre_kk_eps_sub"].imag, "Pre-KK", "tab:orange", "--"),
        (case["final_eps_sub"].imag, "Final KK", "tab:green", "-"),
    )
    for values, trace_label, color, linestyle in eps_imag_traces:
        axes[1, 1].plot(wavenumber, values, label=trace_label, color=color, linestyle=linestyle)
    axes[1, 1].set_title("Im(epsilon)")
    axes[1, 1].set_xlabel("Wavenumber")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_filename, dpi=200)
    plt.close(figure)


def save_results(cases):
    np.savez_compressed(
        RESULTS_PATH,
        true_peak_count=np.array([case["true_peak_count"] for case in cases]),
        selected_total_peak_count=np.array(
            [case["selected_total_peak_count"] for case in cases]
        ),
        joint_residual_norm=np.array([case["joint_residual_norm"] for case in cases]),
        pre_kk_residual_norm=np.array([case["pre_kk_residual_norm"] for case in cases]),
        final_residual_norm=np.array([case["final_residual_norm"] for case in cases]),
        joint_seconds=np.array([case["joint_seconds"] for case in cases]),
        kk_seconds=np.array([case["kk_seconds"] for case in cases]),
        kk_status=np.array([case["kk_status"] for case in cases]),
    )


def run_full_test(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained model checkpoint at {model_path}.")

    rng = np.random.default_rng(RANDOM_SEED)
    cases = []
    total_cases = MAX_PEAKS * SAMPLES_PER_PEAK_COUNT
    for case_index, (true_peak_count, base_params, additional_peak_params) in enumerate(
        generate_cases(rng),
        start=1,
    ):
        true_eps_sub = build_multi_peak_epsilon(base_params, additional_peak_params)
        clean_target_eta_vector = forward_function(true_eps_sub)
        target_eta_vector = add_noise(clean_target_eta_vector, rng)
        case = fit_case(
            true_eps_sub,
            true_peak_count,
            target_eta_vector,
            model_path,
        )
        case["target_eta_vector"] = target_eta_vector
        cases.append(case)
        additional_peaks_text = "; ".join(
            format_params(params) for params in additional_peak_params
        ) or "none"
        print(
            f"[{case_index}/{total_cases}] true peaks={true_peak_count}, "
            f"selected peaks={case['selected_total_peak_count']}, "
            f"final norm={case['final_residual_norm']:.6e}\n"
            f"  base: {format_params(base_params)}\n"
            f"  additional peaks: {additional_peaks_text}"
        )

    ranking = np.argsort([case["final_residual_norm"] for case in cases])
    chosen_indices = {
        "best": int(ranking[0]),
        "median": int(ranking[len(ranking) // 2]),
        "worst": int(ranking[-1]),
    }
    for label, case_index in chosen_indices.items():
        plot_case(cases[case_index], PLOT_FILENAMES[label], label)

    save_results(cases)
    final_norms = np.array([case["final_residual_norm"] for case in cases])
    print(f"Saved results to {RESULTS_PATH}")
    print(f"Final residual norms: min={final_norms.min():.6e}, median={np.median(final_norms):.6e}, max={final_norms.max():.6e}")
    for label, case_index in chosen_indices.items():
        print(f"Saved {label} plot: {PLOT_FILENAMES[label]}")
    return cases


def example_test(model_path=MODEL_PATH):
    """Run one five-peak target sampled from the held-out midpoint grid."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained model checkpoint at {model_path}.")

    rng = np.random.default_rng(RANDOM_SEED)
    base_params = sample_midpoint_params(rng)
    additional_peak_params = np.array(
        [sample_midpoint_params(rng) for _ in range(EXAMPLE_PEAK_COUNT - 1)],
        dtype=np.float64,
    )
    print(f"Running {EXAMPLE_PEAK_COUNT}-peak midpoint-grid example")
    print(f"  base: {format_params(base_params)}")
    for peak_index, params in enumerate(additional_peak_params, start=2):
        print(f"  peak {peak_index}: {format_params(params)}")

    true_eps_sub = build_multi_peak_epsilon(base_params, additional_peak_params)
    clean_target_eta_vector = forward_function(true_eps_sub)
    target_eta_vector = add_noise(clean_target_eta_vector, rng)
    case = fit_case(
        true_eps_sub,
        EXAMPLE_PEAK_COUNT,
        target_eta_vector,
        model_path,
    )
    case["target_eta_vector"] = target_eta_vector
    plot_case(case, EXAMPLE_PLOT_FILENAME, "five-peak midpoint example")
    print(
        f"Selected peaks={case['selected_total_peak_count']}, "
        f"joint norm={case['joint_residual_norm']:.6e}, "
        f"pre-KK norm={case['pre_kk_residual_norm']:.6e}, "
        f"final norm={case['final_residual_norm']:.6e}, "
        f"joint-fit time={case['joint_seconds']:.2f} s"
    )
    print(f"Saved example plot: {EXAMPLE_PLOT_FILENAME}")
    return case


if __name__ == "__main__":
    run_full_test()
    #example_test()
