import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import snompy

eps_inf_range = np.linspace(1, 5, 5)
gamma_range = np.linspace(10, 50, 5)
trans_phon_frequency_range = np.linspace(550, 950,5)

batch_size = 16

harmonic=2 
A=20e-9
z=10e-9
n_wav = 128 * 2

wavenumber = np.linspace(0, 1000, n_wav)


def polarizability_to_vector(polarizability):
    return np.concatenate((np.real(polarizability), np.imag(polarizability)))


def generate_sample_vector(
    eps_inf,
    gamma,
    trans_phon_frequency,
    strength,
    wavenumber_values=None
):
    if wavenumber_values is None:
        wavenumber_values = wavenumber

    eps_sub = snompy.sample.lorentz_perm(
        wavenumber_values,
        nu_j=trans_phon_frequency,
        gamma_j=gamma,
        A_j=strength,
        eps_inf=eps_inf
    )
    lorentz_sample = snompy.sample.bulk_sample(eps_sub=eps_sub)

    polarizability = snompy.fdm.eff_pol_n(
        sample=lorentz_sample,
        A_tip=A,
        n=harmonic,
        z_tip=z
    )

    return polarizability_to_vector(polarizability)


def plot_first_sample(sample_vector, wavenumber_values=None, output_filename="average_training_instance.png"):
    if wavenumber_values is None:
        wavenumber_values = wavenumber

    half_length = len(sample_vector) // 2
    real_part = sample_vector[:half_length]
    imag_part = sample_vector[half_length:]

    plt.figure(figsize=(8, 5))
    plt.plot(wavenumber_values, real_part, label="Real Part")
    plt.plot(wavenumber_values, imag_part, label="Imaginary Part")
    plt.xlabel("Wavenumber")
    plt.ylabel("Polarizability")
    plt.title("Average Training Data Instance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()



def create_training_data():
    data = []
    labels = []
    for eps_inf in eps_inf_range:
        for gamma in gamma_range:
            for trans_phon_frequency in trans_phon_frequency_range:
                long_phon_frequency_range = np.arange(trans_phon_frequency+10, 960, 100)
                length_long = len(long_phon_frequency_range)
                length_trans = len(trans_phon_frequency_range)
                strength_multiple_range = np.pow(long_phon_frequency_range, 2) - np.pow(trans_phon_frequency_range[length_trans-length_long:], 2)
                for strength_multiple in strength_multiple_range:
                    sample_vector = generate_sample_vector(
                        eps_inf,
                        gamma,
                        trans_phon_frequency,
                        strength_multiple * eps_inf
                    )

                    data.append(sample_vector)
                    labels.append([eps_inf, gamma, trans_phon_frequency, strength_multiple])

    average_sample_vector = np.mean(np.array(data, dtype=np.float32), axis=0)
    plot_first_sample(average_sample_vector)

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

    
                    




                    
