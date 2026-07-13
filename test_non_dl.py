"""Run the existing recovery pipeline on literature fused-silica data.

The imported Franta data contain wavelength, refractive index ``n``, and
extinction coefficient ``k``.  They are interpolated onto the project's
current wavenumber grid and converted to complex permittivity with
``epsilon = (n + 1j*k)**2`` before being passed through the existing FDM.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from conv_network_2 import MODEL_PATH
from data_gen import wavenumber
from full_test import complex_eta_from_vector, fit_case
from optimizer import forward_function


DATA_PATH = (
    Path(__file__).resolve().parent
    / "literature_data"
    / "fused_silica_franta_10_100um.csv"
)
PLOT_PATH = Path("test_non_dl_fused_silica.png")
RESULTS_PATH = Path("test_non_dl_fused_silica_results.npz")
NOISE_STD_FRACTION = 0
RANDOM_SEED = 220202979


def load_fused_silica_epsilon(wavenumber_values=wavenumber):
    """Return Franta fused-silica permittivity on ``wavenumber_values``.

    The source columns are wavelength in micrometres, real refractive index
    ``n``, and extinction coefficient ``k``. Linear interpolation is used
    only after checking that the literature record brackets the requested
    wavelength interval.
    """
    data = np.loadtxt(DATA_PATH, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected three columns (wavelength_um, n, k) in {DATA_PATH}.")

    wavelength_um, refractive_index, extinction_coefficient = data.T
    order = np.argsort(wavelength_um)
    wavelength_um = wavelength_um[order]
    refractive_index = refractive_index[order]
    extinction_coefficient = extinction_coefficient[order]

    target_wavenumbers = np.asarray(wavenumber_values, dtype=np.float64)
    if np.any(target_wavenumbers <= 0.0):
        raise ValueError("Wavenumbers must be positive when converting to wavelength.")
    target_wavelength_um = 1.0e4 / target_wavenumbers
    if (
        target_wavelength_um.min() < wavelength_um.min()
        or target_wavelength_um.max() > wavelength_um.max()
    ):
        raise ValueError(
            "The imported fused-silica data do not bracket the requested grid: "
            f"data={wavelength_um.min():.6g}-{wavelength_um.max():.6g} um, "
            f"requested={target_wavelength_um.min():.6g}-"
            f"{target_wavelength_um.max():.6g} um."
        )

    n_interp = np.interp(
        target_wavelength_um,
        wavelength_um,
        refractive_index,
    )
    k_interp = np.interp(
        target_wavelength_um,
        wavelength_um,
        extinction_coefficient,
    )
    complex_refractive_index = n_interp + 1j * k_interp
    return complex_refractive_index**2


def plot_recovery(case, output_path=PLOT_PATH):
    """Plot the recovered dielectric function and corresponding eta traces."""
    true_eps = case["true_eps_sub"]
    one_peak_eps = case["joint_eps_sub"]
    pre_kk_eps = case["pre_kk_eps_sub"]
    recovered_eps = case["final_eps_sub"]
    target_eta = complex_eta_from_vector(case["target_eta_vector"])
    one_peak_eta = complex_eta_from_vector(case["joint_eta_vector"])
    pre_kk_eta = complex_eta_from_vector(case["pre_kk_eta_vector"])
    recovered_eta = complex_eta_from_vector(case["final_eta_vector"])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    figure.suptitle(
        "Literature fused silica: non-Drude-Lorentz recovery\n"
        f"selected peaks={case['selected_total_peak_count']}, "
        f"pre-KK eta norm={case['pre_kk_residual_norm']:.4e}, "
        f"final eta norm={case['final_residual_norm']:.4e}"
    )

    traces = (
        (true_eps, "Literature target", "black", "-", 2.2),
        (one_peak_eps, "One-peak fit", "tab:blue", ":", 1.4),
        (pre_kk_eps, "Best multi-peak fit", "tab:orange", "--", 1.6),
        (recovered_eps, "KK-refined recovery", "tab:green", "-", 1.7),
    )
    for epsilon, label, color, linestyle, linewidth in traces:
        axes[1, 0].plot(
            wavenumber,
            epsilon.real,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        axes[1, 1].plot(
            wavenumber,
            epsilon.imag,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    axes[1, 0].set_title(r"$\mathrm{Re}(\epsilon)$")
    axes[1, 0].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    axes[1, 0].set_ylabel("Dielectric function")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)
    axes[1, 1].set_title(r"$\mathrm{Im}(\epsilon)$")
    axes[1, 1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    eta_traces = (
        (target_eta, "Noisy FDM target", "black", "-", 1.8),
        (one_peak_eta, "One-peak fit", "tab:blue", ":", 1.4),
        (pre_kk_eta, "Best multi-peak fit", "tab:orange", "--", 1.6),
        (recovered_eta, "KK-refined recovery", "tab:green", "-", 1.7),
    )
    for eta, label, color, linestyle, linewidth in eta_traces:
        axes[0, 0].plot(
            wavenumber,
            eta.real,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        axes[0, 1].plot(
            wavenumber,
            eta.imag,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    axes[0, 0].set_title(r"$\mathrm{Re}(\eta)$")
    axes[0, 0].set_ylabel(r"Normalized near-field signal $\eta$")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)
    axes[0, 1].set_title(r"$\mathrm{Im}(\eta)$")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run_non_dl_test(model_path=MODEL_PATH):
    """Run the full-test recovery stages on fused-silica literature data."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained model checkpoint at {model_path}.")

    true_eps_sub = load_fused_silica_epsilon()
    clean_target_eta_vector = forward_function(true_eps_sub)
    rng = np.random.default_rng(RANDOM_SEED)
    noise_std = NOISE_STD_FRACTION * float(
        np.mean(np.abs(clean_target_eta_vector))
    )
    target_eta_vector = clean_target_eta_vector + rng.normal(
        0.0,
        noise_std,
        size=clean_target_eta_vector.shape,
    )

    # A literature target has no meaningful true Lorentz-peak count; zero is
    # metadata only. fit_case still tries all one-to-five-peak candidates.
    case = fit_case(
        true_eps_sub=true_eps_sub,
        true_peak_count=0,
        target_eta_vector=target_eta_vector,
        model_path=model_path,
    )
    case["target_eta_vector"] = target_eta_vector
    case["clean_target_eta_vector"] = clean_target_eta_vector

    epsilon_error = case["final_eps_sub"] - true_eps_sub
    epsilon_rmse = float(np.sqrt(np.mean(np.abs(epsilon_error) ** 2)))
    epsilon_relative_l2 = float(
        np.linalg.norm(epsilon_error) / np.linalg.norm(true_eps_sub)
    )
    clean_eta = complex_eta_from_vector(clean_target_eta_vector)
    recovered_eta = complex_eta_from_vector(case["final_eta_vector"])
    clean_eta_relative_l2 = float(
        np.linalg.norm(recovered_eta - clean_eta) / np.linalg.norm(clean_eta)
    )

    plot_recovery(case)
    np.savez_compressed(
        RESULTS_PATH,
        wavenumber=wavenumber,
        true_eps_sub=true_eps_sub,
        one_peak_eps_sub=case["joint_eps_sub"],
        pre_kk_eps_sub=case["pre_kk_eps_sub"],
        recovered_eps_sub=case["final_eps_sub"],
        clean_target_eta_vector=clean_target_eta_vector,
        noisy_target_eta_vector=target_eta_vector,
        noise_std=noise_std,
        recovered_eta_vector=case["final_eta_vector"],
        selected_total_peak_count=case["selected_total_peak_count"],
        pre_kk_residual_norm=case["pre_kk_residual_norm"],
        final_residual_norm=case["final_residual_norm"],
        kk_status=case["kk_status"],
        kk_nfev=case["kk_nfev"],
        epsilon_rmse=epsilon_rmse,
        epsilon_relative_l2=epsilon_relative_l2,
        clean_eta_relative_l2=clean_eta_relative_l2,
    )

    print(f"Selected Lorentz peaks: {case['selected_total_peak_count']}")
    print(
        "Noise standard deviation "
        f"({NOISE_STD_FRACTION:.6g} * mean |eta|): {noise_std:.6e}"
    )
    print(f"Pre-KK eta residual norm: {case['pre_kk_residual_norm']:.6e}")
    print(f"Final eta residual norm: {case['final_residual_norm']:.6e}")
    print(f"KK function evaluations: {case['kk_nfev']}")
    print(f"KK termination status {case['kk_status']}: {case['kk_message']}")
    print(f"Complex epsilon RMSE: {epsilon_rmse:.6e}")
    print(f"Complex epsilon relative L2 error: {epsilon_relative_l2:.6e}")
    print(f"Clean eta relative L2 error: {clean_eta_relative_l2:.6e}")
    print(f"Saved recovery plot: {PLOT_PATH}")
    print(f"Saved numerical results: {RESULTS_PATH}")
    return case


if __name__ == "__main__":
    run_non_dl_test()
