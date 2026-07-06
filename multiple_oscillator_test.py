import matplotlib.pyplot as plt
import numpy as np
from conv_network_2 import MODEL_PATH
from data_gen import eta_to_vector
from optimizer import optimize_parameters, forward_function
import snompy


eps_inf = 1.6
gamma = 10
trans_phon_frequency = 876
long_phon_frequency = 886

batch_size = 16

harmonic = 3
A = 20e-9
z = 10e-9
n_wav = 128 * 2
long_phon_frequency_stop = 960
long_phon_frequency_step = 100

wavenumber = np.linspace(0, 1000, n_wav)

theta_in = np.deg2rad(60)
c_r = 0.9
PERCENT_ERROR_WINDOW = (550.0, 800.0)

def alpha_fff_ref():
    eps = 11.7
    ref_sample = snompy.sample.bulk_sample(eps_sub=eps)  # Si
    alpha_eff = snompy.fdm.eff_pol_n(sample=ref_sample, A_tip=A, n=harmonic, z_tip=z)

    r = ref_sample.refl_coef(theta_in=theta_in)
    fff_ref = (1 + c_r * r) ** 2

    return alpha_eff, fff_ref


def plot_eta_percent_error(
    wavenumber_values,
    true_eta,
    recovered_eta,
    window,
    output_filename="multiple_oscillator_eta_percent_error.png",
):
    window_min, window_max = window
    window_mask = (wavenumber_values >= window_min) & (wavenumber_values <= window_max)
    windowed_wavenumber = wavenumber_values[window_mask]

    denom_floor = 1e-12
    recovered_real_error = 100.0 * (
        recovered_eta.real - true_eta.real
    ) / np.maximum(np.abs(true_eta.real), denom_floor)
    recovered_imag_error = 100.0 * (
        recovered_eta.imag - true_eta.imag
    ) / np.maximum(np.abs(true_eta.imag), denom_floor)

    plt.figure(figsize=(10, 6))
    plt.plot(
        windowed_wavenumber,
        recovered_real_error[window_mask],
        label="Recovered Re(eta) % Error",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        windowed_wavenumber,
        recovered_imag_error[window_mask],
        label="Recovered Im(eta) % Error",
        linewidth=2,
        linestyle="--",
    )
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.xlabel("Wavenumber")
    plt.ylabel("Percent Error")
    plt.title(f"Eta Percent Error ({window_min:.0f} to {window_max:.0f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()


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
        nu_j=trans_phon_frequency/1.3,
        gamma_j=gamma/1.3,
        A_j=strength_multiple * eps_inf /1.3,
        eps_inf=0,
    )
)
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

eta_vector = eta_to_vector(eta)


recovered_eps_sub = np.zeros_like(eps_sub, dtype=np.complex128)
recovered_eta_vector_total = np.zeros_like(eta_vector, dtype=np.float64)


_, optimization_result, _ = optimize_parameters(eta_vector, MODEL_PATH, wavenumber)
recovered_params = optimization_result.x
recovered_eps_inf, recovered_trans_phon_frequency, recovered_gamma, recovered_strength_multiple = recovered_params
recovered_eps_sub_1 = recovered_eps_sub + snompy.sample.lorentz_perm(
    wavenumber,
    nu_j=recovered_trans_phon_frequency,
    gamma_j=recovered_gamma,
    A_j=recovered_eps_inf * recovered_strength_multiple,
    eps_inf=0
)

recovered_eta_vector = forward_function(recovered_params, wavenumber)
recovered_eta_vector_total = recovered_eta_vector_total + recovered_eta_vector
eta_vector = eta_vector - recovered_eta_vector

_, optimization_result, _ = optimize_parameters(eta_vector, MODEL_PATH, wavenumber)
recovered_params = optimization_result.x
recovered_eps_inf, recovered_trans_phon_frequency, recovered_gamma, recovered_strength_multiple = recovered_params
recovered_eps_sub_2 = snompy.sample.lorentz_perm(
    wavenumber,
    nu_j=recovered_trans_phon_frequency,
    gamma_j=recovered_gamma,
    A_j=recovered_eps_inf * recovered_strength_multiple,
    eps_inf=eps_inf
)

recovered_eta_vector = forward_function(recovered_params, wavenumber)
recovered_eta_vector_total = recovered_eta_vector_total + recovered_eta_vector
eta_vector = eta_vector - recovered_eta_vector

recovered_eps_sub = recovered_eps_sub_1 + recovered_eps_sub_2


half_length = len(recovered_eta_vector_total) // 2
recovered_eta = (
    recovered_eta_vector_total[:half_length]
    + 1j * recovered_eta_vector_total[half_length:]
)

plt.figure(figsize=(10, 6))
plt.plot(wavenumber, eps_sub.real, label="Multi-oscillator Re(eps_sub)", linewidth=2)
plt.plot(wavenumber, eps_sub.imag, label="Multi-oscillator Im(eps_sub)", linewidth=2)
plt.plot(
    wavenumber,
    recovered_eps_sub.real,
    label="Recovered Lorentz Re(eps)",
    linewidth=2,
    linestyle="--",
)
plt.plot(
    wavenumber,
    recovered_eps_sub.imag,
    label="Recovered Lorentz Im(eps)",
    linewidth=2,
    linestyle="--",
)
plt.xlabel("Wavenumber")
plt.ylabel("Dielectric Function")
plt.title("Multiple Oscillator Dielectric Function")
plt.legend()
plt.tight_layout()
plt.savefig("multiple_oscillator_eps_sub.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(wavenumber, eta.real, label="True Re(eta)", linewidth=2)
plt.plot(wavenumber, eta.imag, label="True Im(eta)", linewidth=2)
plt.plot(
    wavenumber,
    recovered_eta.real,
    label="Recovered Re(eta)",
    linewidth=2,
    linestyle="--",
)
plt.plot(
    wavenumber,
    recovered_eta.imag,
    label="Recovered Im(eta)",
    linewidth=2,
    linestyle="--",
)
plt.xlabel("Wavenumber")
plt.ylabel("Eta")
plt.title("Eta Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("multiple_oscillator_eta_comparison.png", dpi=200)
plt.close()


print(eta_vector.shape)


plt.figure(figsize=(10, 6))
plt.plot(wavenumber, eta_vector[:256], label="True Re(eta)", linewidth=2)
plt.plot(wavenumber, eta_vector[256:], label="True Im(eta)", linewidth=2)
# plt.plot(
#     wavenumber,
#     recovered_eta.real,
#     label="Recovered Re(eta)",
#     linewidth=2,
#     linestyle="--",
# )
# plt.plot(
#     wavenumber,
#     recovered_eta.imag,
#     label="Recovered Im(eta)",
#     linewidth=2,
#     linestyle="--",
# )
plt.xlabel("Wavenumber")
plt.ylabel("Eta Vector")
plt.title("Eta Vector")
plt.legend()
plt.tight_layout()
plt.savefig("Eta_vector", dpi=200)
plt.close()





plot_eta_percent_error(
    wavenumber,
    eta,
    recovered_eta,
    PERCENT_ERROR_WINDOW,
)



