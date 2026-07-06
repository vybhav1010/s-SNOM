import numpy as np
import matplotlib.pyplot as plt
import snompy


# Manually chosen sample and optical parameters for comparing the signal
# before and after applying the far-field coefficients/reference normalization.
EPS_INF = 1.0
GAMMA = 10.0
TRANS_PHON_FREQUENCY = 550.0
LONG_PHON_FREQUENCY = 560.0
THETA_IN = np.deg2rad(60.0)
C_R = 0.9
REFERENCE_EPS = 11.7

HARMONIC = 3
A_TIP = 20e-9
Z_TIP = 10e-9
N_WAV = 128 * 2
WAVENUMBER = np.linspace(0, 1000, N_WAV)

OUTPUT_FILENAME = "signal_plot.png"


def strength_from_frequencies(trans_phon_frequency, long_phon_frequency, eps_inf):
    strength_multiple = (long_phon_frequency ** 2) - (trans_phon_frequency ** 2)
    return strength_multiple * eps_inf


def signal_components(signal):
    magnitude = np.abs(signal)
    phase = np.unwrap(np.angle(signal))
    return magnitude, phase


def build_sample_signal(eps_inf, gamma, trans_phon_frequency, strength, wavenumbers):
    eps_sub = snompy.sample.lorentz_perm(
        wavenumbers,
        nu_j=trans_phon_frequency,
        gamma_j=gamma,
        A_j=strength,
        eps_inf=eps_inf,
    )
    sample = snompy.sample.bulk_sample(eps_sub=eps_sub)
    alpha_eff = snompy.fdm.eff_pol_n(
        sample=sample,
        A_tip=A_TIP,
        n=HARMONIC,
        z_tip=Z_TIP,
    )
    return sample, alpha_eff


def build_reference_signal():
    reference_sample = snompy.sample.bulk_sample(eps_sub=REFERENCE_EPS)
    reference_alpha_eff = snompy.fdm.eff_pol_n(
        sample=reference_sample,
        A_tip=A_TIP,
        n=HARMONIC,
        z_tip=Z_TIP,
    )
    reference_r = reference_sample.refl_coef(theta_in=THETA_IN)
    far_field_reference = (1 + C_R * reference_r) ** 2
    return reference_alpha_eff, far_field_reference


def apply_far_field_coefficients(sample, alpha_eff_sample, reference_alpha_eff, far_field_reference):
    sample_r = sample.refl_coef(theta_in=THETA_IN)
    far_field_sample = (1 + C_R * sample_r) ** 2
    return (far_field_sample * alpha_eff_sample) / (far_field_reference * reference_alpha_eff)


def normalize_without_far_field(alpha_eff_sample, reference_alpha_eff):
    return alpha_eff_sample / reference_alpha_eff


def plot_signals(wavenumbers, signal_without_far_field, signal_with_far_field, output_filename):
    magnitude_without, phase_without = signal_components(signal_without_far_field)
    magnitude_with, phase_with = signal_components(signal_with_far_field)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(wavenumbers, magnitude_without, label="Normalized Without Far-Field", linewidth=2)
    axes[0].plot(wavenumbers, magnitude_with, label="With Far-Field", linewidth=2)
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Signal Comparison")
    axes[0].legend()

    axes[1].plot(wavenumbers, phase_without, label="Normalized Without Far-Field", linewidth=2)
    axes[1].plot(wavenumbers, phase_with, label="With Far-Field", linewidth=2)
    axes[1].set_xlabel("Wavenumber")
    axes[1].set_ylabel("Unwrapped Phase (rad)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_filename)
    plt.close(fig)


def main():
    strength = strength_from_frequencies(
        TRANS_PHON_FREQUENCY,
        LONG_PHON_FREQUENCY,
        EPS_INF,
    )
    sample, alpha_eff_sample = build_sample_signal(
        EPS_INF,
        GAMMA,
        TRANS_PHON_FREQUENCY,
        strength,
        WAVENUMBER,
    )
    reference_alpha_eff, far_field_reference = build_reference_signal()
    signal_without_far_field = normalize_without_far_field(
        alpha_eff_sample,
        reference_alpha_eff,
    )
    signal_with_far_field = apply_far_field_coefficients(
        sample,
        alpha_eff_sample,
        reference_alpha_eff,
        far_field_reference,
    )

    plot_signals(
        WAVENUMBER,
        signal_without_far_field,
        signal_with_far_field,
        OUTPUT_FILENAME,
    )
    print(f"Saved signal comparison plot to {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
