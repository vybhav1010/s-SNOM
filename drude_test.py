import matplotlib.pyplot as plt
import numpy as np
import snompy

from data_gen import wavenumber
from optimizer import forward_function


DRUDE_EPS_INF = 5
DRUDE_OMEGA_P = 500
DRUDE_GAMMA = 150
SPEED_OF_LIGHT_CM_PER_S = 2.99792458e10
OUTPUT_FILENAME = "drude_eta_dielectric.png"


#def angular_frequency_to_wavenumber(angular_frequency):
    #return angular_frequency / (2.0 * np.pi * SPEED_OF_LIGHT_CM_PER_S)


def drude_wavenumber_grid(wavenumber_values):
    drude_wavenumbers = np.asarray(wavenumber_values, dtype=np.float64).copy()
    positive_wavenumbers = drude_wavenumbers[drude_wavenumbers > 0.0]
    if positive_wavenumbers.size == 0:
        raise ValueError("Drude permittivity requires at least one positive wavenumber.")
    drude_wavenumbers[drude_wavenumbers <= 0.0] = np.min(positive_wavenumbers)
    return drude_wavenumbers


def eta_from_vector(eta_vector):
    half_length = len(eta_vector) // 2
    return eta_vector[:half_length] + 1j * eta_vector[half_length:]


def build_drude_perm(wavenumber_values):
    nu_plasma = DRUDE_OMEGA_P
    gamma = DRUDE_GAMMA
    return snompy.sample.drude_perm(
        drude_wavenumber_grid(wavenumber_values),
        nu_plasma=nu_plasma,
        gamma=gamma,
        eps_inf=DRUDE_EPS_INF,
    )



def plot_drude_response(wavenumber_values, eps_sub, eta, output_filename):
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(wavenumber_values, eps_sub.real, label="Re(eps)", linewidth=2)
    axes[0].plot(wavenumber_values, eps_sub.imag, label="Im(eps)", linewidth=2)
    axes[0].set_ylabel("Dielectric Function")
    axes[0].set_title("Drude Dielectric Function")
    axes[0].legend()

    axes[1].plot(wavenumber_values, eta.real, label="Re(eta)", linewidth=2)
    axes[1].plot(wavenumber_values, eta.imag, label="Im(eta)", linewidth=2)
    axes[1].set_xlabel("Wavenumber")
    axes[1].set_ylabel("Eta")
    axes[1].set_title("Eta from Drude Permittivity")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_filename, dpi=200)
    plt.close(figure)


def main():
    eps_sub = build_drude_perm(wavenumber)
    eta = eta_from_vector(forward_function(eps_sub))
    plot_drude_response(wavenumber, eps_sub, eta, OUTPUT_FILENAME)
    print(f"Saved Drude eta and dielectric plot to {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
