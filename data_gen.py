import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import snompy
from pathlib import Path

eps_inf_range = np.linspace(1, 5, 5)
gamma_ph_range = np.linspace(10, 50, 5)
gamma_p_range = np.linspace(250, 500, 5)
trans_phon_frequency_range = np.linspace(200,950,8)
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
TRAINING_DATA_PATH = "training_data.npz"


num_oscillators = np.linspace(1, 5, 5)

batch_size = 16

harmonic=3 
A=20e-9
z=10e-9
n_wav = 512
long_phon_frequency_stop = 960
long_phon_frequency_step = 100



wavenumber = np.linspace(100, 1000, n_wav)

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

    plt.figure(figsize=(8, 5))
    plt.plot(wavenumber_values, eta_real, label="Real")
    plt.plot(wavenumber_values, eta_imag, label="Imaginary")
    plt.xlabel("Wavenumber")
    plt.ylabel("Response")
    plt.title("First Training Data Instance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved first training sample plot to {output_filename}")


def save_training_data_npz(data, labels, output_path=TRAINING_DATA_PATH):
    np.savez_compressed(
        output_path,
        data=data,
        labels=labels,
        wavenumber=wavenumber.astype(np.float32),
        parameter_names=np.array(PARAMETER_NAMES),
    )
    print(f"Saved training data to {output_path}")


def build_dataloaders(data, labels):
    data = torch.from_numpy(np.asarray(data, dtype=np.float32))
    labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))

    dataset = TensorDataset(data, labels)
    train_length = int(len(dataset) * 0.7)
    val_length = int(len(dataset) * 0.15)
    test_length = len(dataset) - train_length - val_length

    train, val, test = random_split(dataset, [train_length, val_length, test_length])

    train_dataloader = DataLoader(train, batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def create_training_data(save_npz=True):
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

    data_array = np.array(data, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.float32)
    if save_npz:
        save_training_data_npz(data_array, labels_array)

    return build_dataloaders(data_array, labels_array)


def create_training_data_from_npz(npz_path=TRAINING_DATA_PATH):
    with np.load(npz_path) as archive:
        data = archive["data"]
        labels = archive["labels"]
        wavenumber_values = archive.get("wavenumber", wavenumber)

    if data.shape[1] != n_wav * 2:
        raise ValueError(
            f"Expected data vectors of length {n_wav * 2}, got {data.shape[1]}."
        )
    if labels.shape[1] != NUM_PARAMETERS:
        raise ValueError(
            f"Expected {NUM_PARAMETERS} label columns, got {labels.shape[1]}."
        )
    if len(data) == 0:
        raise ValueError("Training archive contains no samples.")

    print(f"Loaded training data from {npz_path}")
    plot_first_sample(data[0], wavenumber_values=wavenumber_values)
    return build_dataloaders(data, labels)


def load_or_create_training_data(npz_path=TRAINING_DATA_PATH):
    # if Path(npz_path).exists():
    #     return create_training_data_from_npz(npz_path)
    return create_training_data()


def main():
    train_dataloader, val_dataloader, test_dataloader = load_or_create_training_data()
    print(train_dataloader, val_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
