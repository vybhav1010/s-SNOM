import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import snompy

eps_inf_range = np.linspace(1, 5, 5)
gamma_ph_range = np.linspace(10, 50, 5)
gamma_p_range = np.linspace(10, 50, 5)
trans_phon_frequency_range = np.linspace(550, 950,5)
plas_freq_range = np.linspace(400, 800, 5)
PARAMETER_NAMES = (
    "eps_inf",
    "gamma_ph",
    "gamma_p",
    "plas_freq",
    "trans_phon_frequency",
    "strength_multiple",
)
NUM_PARAMETERS = len(PARAMETER_NAMES)


num_oscillators = np.linspace(1, 5, 5)

batch_size = 16

harmonic=3 
A=20e-9
z=10e-9
n_wav = 512
long_phon_frequency_stop = 960
long_phon_frequency_step = 100



wavenumber = np.linspace(0, 1000, n_wav)

theta_in = np.deg2rad(60)
c_r = 0.9



def alpha_fff_ref():
    eps = 11.7
    ref_sample = snompy.sample.bulk_sample(eps_sub=11.7) # Si
    alpha_eff = snompy.fdm.eff_pol_n(sample=ref_sample, A_tip=A, n=harmonic, z_tip=z)

    r = ref_sample.refl_coef(theta_in=theta_in)
    fff_ref = (1 + c_r * r) ** 2

    return alpha_eff, fff_ref

def eta_to_vector(eta):
    return np.concatenate((np.real(eta), np.imag(eta)))


def drude_wavenumber_grid(wavenumber_values):
    drude_wavenumbers = np.asarray(wavenumber_values, dtype=np.float64).copy()
    positive_wavenumbers = drude_wavenumbers[drude_wavenumbers > 0.0]
    if positive_wavenumbers.size == 0:
        raise ValueError("Drude permittivity requires at least one positive wavenumber.")
    drude_wavenumbers[drude_wavenumbers <= 0.0] = np.min(positive_wavenumbers)
    return drude_wavenumbers


def generate_eta_vector(
    eps_inf,
    gamma_ph,
    gamma_p,
    trans_phon_frequency,
    plas_freq,
    strength,
    wavenumber_values=None
):
    if wavenumber_values is None:
        wavenumber_values = wavenumber

    eps_lorentz = snompy.sample.lorentz_perm(
        wavenumber_values,
        nu_j=trans_phon_frequency,
        gamma_j=gamma_ph,
        A_j=strength,
        eps_inf=eps_inf
    )

    eps_drude = snompy.sample.drude_perm(
        drude_wavenumber_grid(wavenumber_values),
        nu_plasma=plas_freq,
        gamma=gamma_p,
        eps_inf=0.0,
    )

    eps_sub = eps_lorentz + eps_drude


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

    return eta_to_vector(eta)


def plot_first_sample(eta_vector, wavenumber_values=None, output_filename="first_training_instance.png"):
    if wavenumber_values is None:
        wavenumber_values = wavenumber

    half_length = len(eta_vector) // 2
    eta_real = eta_vector[:half_length]
    eta_imag = eta_vector[half_length:]
    eta = eta_real + 1j * eta_imag
    magnitude = np.abs(eta)
    phase = np.angle(eta)

    plt.figure(figsize=(8, 5))
    plt.plot(wavenumber_values, magnitude, label="Magnitude")
    plt.plot(wavenumber_values, phase, label="Phase")
    plt.xlabel("Wavenumber")
    plt.ylabel("Response")
    plt.title("First Training Data Instance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()



def create_training_data():
    data = []
    labels = []
    for eps_inf in eps_inf_range:
        for gamma_ph in gamma_ph_range:
            for gamma_p in gamma_p_range:
                for plas_freq in plas_freq_range:
                    for trans_phon_frequency in trans_phon_frequency_range:
                        # Include the endpoint so trans_phon_frequency=950 still yields long_phon_frequency=960.
                        long_phon_frequency_range = np.arange(
                            trans_phon_frequency + 10,
                            long_phon_frequency_stop + 1,
                            long_phon_frequency_step,
                        )
                        strength_multiple_range = (long_phon_frequency_range ** 2) - (trans_phon_frequency ** 2)
                        for strength_multiple in strength_multiple_range:
                            eta_vector = generate_eta_vector(
                                eps_inf,
                                gamma_ph,
                                gamma_p,
                                trans_phon_frequency,
                                plas_freq,
                                strength_multiple * eps_inf
                            )

                            data.append(eta_vector)
                            labels.append([eps_inf, gamma_ph, gamma_p, plas_freq, trans_phon_frequency, strength_multiple])

    if data:
        plot_first_sample(np.array(data[0], dtype=np.float32))

    data = torch.from_numpy(np.array(data, dtype=np.float32))


    labels = torch.from_numpy(np.array(labels, dtype=np.float32))

    dataset = TensorDataset(data, labels)
    train_length = int(len(dataset) * 0.7)
    val_length = int(len(dataset) * 0.15)
    test_length = len(dataset) - train_length - val_length

    train, val, test = random_split(dataset, [train_length, val_length, test_length])

    train_dataloader = DataLoader(train, batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def main():
    train_dataloader, val_dataloader, test_dataloader = create_training_data()
    print(train_dataloader, val_dataloader, test_dataloader)


dataset = create_training_data()

    
                    




                    
