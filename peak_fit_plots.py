import matplotlib.pyplot as plt
import numpy as np


PARAMETER_NAMES = (
    "eps_inf",
    "nu_j",
    "gamma_j",
    "strength_multiple",
)


def _split_eta_vector(eta_vector):
    eta_vector = np.asarray(eta_vector, dtype=np.float64)
    half_length = len(eta_vector) // 2
    return eta_vector[:half_length], eta_vector[half_length:]


def _format_params(params):
    return ", ".join(
        f"{name}={value:.4g}" for name, value in zip(PARAMETER_NAMES, params)
    )


def plot_peak_fit_progress(
    wavenumber_values,
    fit_records,
    output_filename="multiple_oscillator_peak_fit_progress.png",
):
    if not fit_records:
        return

    figure, axes = plt.subplots(
        len(fit_records),
        2,
        figsize=(14, 4.5 * len(fit_records)),
        squeeze=False,
        sharex=True,
    )

    for row_index, record in enumerate(fit_records):
        target_real, target_imag = _split_eta_vector(record["target_eta_vector"])
        initial_real, initial_imag = _split_eta_vector(record["initial_eta_vector"])
        refined_real, refined_imag = _split_eta_vector(record["refined_eta_vector"])

        real_axis = axes[row_index, 0]
        imag_axis = axes[row_index, 1]

        real_axis.plot(
            wavenumber_values,
            target_real,
            label="Fit target",
            linewidth=2,
            color="black",
        )
        real_axis.plot(
            wavenumber_values,
            initial_real,
            label="NN initial guess",
            linewidth=2,
            linestyle=":",
        )
        real_axis.plot(
            wavenumber_values,
            refined_real,
            label="Optimizer refinement",
            linewidth=2,
            linestyle="--",
        )
        real_axis.set_ylabel("Re(eta)")
        real_axis.set_title(
            f"Peak {record['peak_index']} Real\n"
            f"NN: {_format_params(record['initial_params'])}\n"
            f"Opt: {_format_params(record['refined_params'])}"
        )
        real_axis.legend()

        imag_axis.plot(
            wavenumber_values,
            target_imag,
            label="Fit target",
            linewidth=2,
            color="black",
        )
        imag_axis.plot(
            wavenumber_values,
            initial_imag,
            label="NN initial guess",
            linewidth=2,
            linestyle=":",
        )
        imag_axis.plot(
            wavenumber_values,
            refined_imag,
            label="Optimizer refinement",
            linewidth=2,
            linestyle="--",
        )
        imag_axis.set_ylabel("Im(eta)")
        imag_axis.set_title(
            f"Peak {record['peak_index']} Imag\n"
            f"NN residual={record['initial_residual_norm']:.4e}, "
            f"optimizer residual={record['refined_residual_norm']:.4e}"
        )
        imag_axis.legend()

    axes[-1, 0].set_xlabel("Wavenumber")
    axes[-1, 1].set_xlabel("Wavenumber")

    figure.tight_layout()
    figure.savefig(output_filename, dpi=200)
    plt.close(figure)
