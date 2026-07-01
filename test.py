import numpy as np

from fdmDataGen import (
    eps_inf_range,
    gamma_range,
    generate_sample_vector,
    plot_first_sample,
    trans_phon_frequency_range,
)


n_wav = 128
em_spectrum_wavenumber = np.linspace(0, 20000, n_wav)

eps_inf = eps_inf_range[0]
gamma = gamma_range[0]
trans_phon_frequency = trans_phon_frequency_range[0]

long_phon_frequency_range = np.arange(trans_phon_frequency + 10, 960, 100)
length_long = len(long_phon_frequency_range)
length_trans = len(trans_phon_frequency_range)
strength_multiple_range = (
    np.pow(long_phon_frequency_range, 2)
    - np.pow(trans_phon_frequency_range[length_trans - length_long:], 2)
)
strength_multiple = strength_multiple_range[0]
strength = strength_multiple * eps_inf

sample_vector = generate_sample_vector(
    eps_inf,
    gamma,
    trans_phon_frequency,
    strength,
    wavenumber_values=em_spectrum_wavenumber,
)

plot_first_sample(
    sample_vector,
    wavenumber_values=em_spectrum_wavenumber,
    output_filename="first_em_spectrum_instance.png",
)
