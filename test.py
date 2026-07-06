import numpy as np

from data_gen import (
    eps_inf_range,
    gamma_range,
    generate_eta_vector,
    long_phon_frequency_stop,
    plot_first_sample,
    trans_phon_frequency_range,
)


n_wav = 128
em_spectrum_wavenumber = np.linspace(0, 20000, n_wav)

eps_inf = eps_inf_range[0]
gamma = gamma_range[0]
trans_phon_frequency = trans_phon_frequency_range[0]

long_phon_frequency_range = np.arange(trans_phon_frequency + 10, long_phon_frequency_stop + 1, 100)
strength_multiple_range = (long_phon_frequency_range ** 2) - (trans_phon_frequency ** 2)
strength_multiple = strength_multiple_range[0]
strength = strength_multiple * eps_inf

eta_vector = generate_eta_vector(
    eps_inf,
    gamma,
    trans_phon_frequency,
    strength,
    wavenumber_values=em_spectrum_wavenumber,
)

plot_first_sample(
    eta_vector,
    wavenumber_values=em_spectrum_wavenumber,
    output_filename="first_em_spectrum_instance.png",
)
