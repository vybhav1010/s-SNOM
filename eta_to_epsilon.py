import numpy as np

from conv_network_2 import MODEL_PATH
from data_gen import n_wav
from full_test import fit_case
from optimizer import eta_to_vector


def eta_to_epsilon(eta, model_path=MODEL_PATH, return_details=False):
    """Recover complex epsilon from eta using the current project pipeline."""
    eta = np.asarray(eta)
    if eta.shape == (n_wav,) and np.iscomplexobj(eta):
        target_eta_vector = eta_to_vector(eta)
    elif eta.shape == (2 * n_wav,):
        target_eta_vector = eta.astype(np.float64, copy=False)
    else:
        raise ValueError(
            f"eta must be complex shape ({n_wav},) or vector shape ({2 * n_wav},); "
            f"got {eta.shape}."
        )

    details = fit_case(
        true_eps_sub=None,
        true_peak_count=0,
        target_eta_vector=target_eta_vector,
        model_path=model_path,
    )
    return details if return_details else details["final_eps_sub"]
