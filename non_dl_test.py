"""Benchmark eta-to-epsilon recovery on screened Polyanskiy datasets."""

import argparse
from pathlib import Path
import re
import zipfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from conv_network_2 import MODEL_PATH
from data_gen import n_wav, wavenumber
from eta_to_epsilon import eta_to_epsilon
from optimizer import forward_function


CASE_DIRECTORY = Path("non_dl_case_results")
# Switch this single flag when changing the development/held-out dataset.
isvalidation = True
SPLIT_NAME = "validation" if isvalidation else "test"
DATA_ARCHIVE = Path(f"literature_data/polyanskiy_{SPLIT_NAME}_csv.zip")
SUMMARY_PATH = Path(f"non_dl_{SPLIT_NAME}_results.npz")
PLOT_PATHS = {
    "best": Path(f"non_dl_{SPLIT_NAME}_best.png"),
    "median": Path(f"non_dl_{SPLIT_NAME}_median.png"),
    "worst": Path(f"non_dl_{SPLIT_NAME}_worst.png"),
}
EXAMPLE_PLOT_PATH = Path(f"non_dl_{SPLIT_NAME}_example.png")


POLYANSKIY_RECORDS = (
    "glass/misc/soda-lime/nk/Rubin-IR.yml",
    "glass/optical/LZOS K108/nk/Bassarab.yml",
    "main/Al/nk/Rakic-BB.yml",
    "main/Al/nk/Rakic-LD.yml",
    "main/Al2O3/nk/Franta.yml",
    "main/Al2O3/nk/Rodriguez de Marcos.yml",
    "main/BaF2/nk/Querry.yml",
    "main/CsBr/nk/Querry.yml",
    "main/GaAs/nk/Franta-300K.yml",
    "main/GaAs/nk/Franta-370K.yml",
    "main/GaAs/nk/Franta-440K.yml",
    "main/GaSe/nk/Chen-nk-e.yml",
    "main/GaSe/nk/Chen-nk-o.yml",
    "main/GdF3/nk/Franta.yml",
    "main/H2O/nk/Hale.yml",
    "main/H2O/nk/Rowe-240K.yml",
    "main/H2O/nk/Rowe-253K.yml",
    "main/H2O/nk/Rowe-263K.yml",
    "main/H2O/nk/Rowe-273K.yml",
    "main/H2O/nk/Segelstein.yml",
    "main/H2O/nk/Warren-1984.yml",
    "main/H2O/nk/Warren-2008.yml",
    "main/HfO2/nk/Bright.yml",
    "main/HfO2/nk/Franta.yml",
    "main/KCl/nk/Querry.yml",
    "main/Lu/nk/Garcia-Cortes.yml",
    "main/MgF2/nk/Franta.yml",
    "main/Mo/nk/Querry.yml",
    "main/NaCl/nk/Querry.yml",
    "main/Nb2O5/nk/Franta.yml",
    "main/SiC/nk/Larruquert.yml",
    "main/SiO2/nk/Franta.yml",
    "main/SiO2/nk/Franta-25C.yml",
    "main/SiO2/nk/Franta-300C.yml",
    "main/SrF2/nk/Rodriguez-de Marcos.yml",
    "main/Ta2O5/nk/Bright-amorphous.yml",
    "main/Ta2O5/nk/Bright-nanocrystalline.yml",
    "main/Ta2O5/nk/Franta-2015.yml",
    "main/Ta2O5/nk/Franta-2025.yml",
    "main/Tb3Ga5O12/nk/Franta.yml",
    "main/TiO2/nk/Franta.yml",
    "main/TiO2/nk/Siefke.yml",
    "main/Y3Al5O12/nk/Franta.yml",
    "main/ZnS/nk/Querry.yml",
    "other/clays/illite/nk/Querry.yml",
    "other/clays/kaolinite/nk/Querry.yml",
    "other/clays/montmorillonite/nk/Querry.yml",
)

VALIDATION_RECORDS = tuple(
    (index, record)
    for index, record in enumerate(POLYANSKIY_RECORDS, start=1)
    if index % 2 == 1
)
TEST_RECORDS = tuple(
    (index, record)
    for index, record in enumerate(POLYANSKIY_RECORDS, start=1)
    if index % 2 == 0
)
ACTIVE_RECORDS = VALIDATION_RECORDS if isvalidation else TEST_RECORDS


def vector_to_eta(eta_vector):
    eta_vector = np.asarray(eta_vector, dtype=np.float64)
    return eta_vector[:n_wav] + 1j * eta_vector[n_wav:]


def load_polyanskiy_epsilon(record_path, archive_path=DATA_ARCHIVE):
    """Load one compressed CSV and interpolate n,k onto the model grid."""
    member = record_path[:-4] + ".csv" if record_path.endswith(".yml") else record_path
    with zipfile.ZipFile(archive_path) as archive:
        try:
            with archive.open(member) as stream:
                rows = np.loadtxt(stream, delimiter=",", comments="#", ndmin=2)
        except KeyError as error:
            raise FileNotFoundError(
                f"{record_path} is not present in {archive_path}."
            ) from error
    rows = rows[np.argsort(rows[:, 0])]
    wavelength_um, refractive_index, extinction = rows[:, :3].T
    target_wavelength_um = 1.0e4 / wavenumber
    if (
        target_wavelength_um.min() < wavelength_um.min()
        or target_wavelength_um.max() > wavelength_um.max()
    ):
        raise ValueError(f"{record_path} does not bracket 10--100 um.")
    n_interp = np.interp(target_wavelength_um, wavelength_um, refractive_index)
    k_interp = np.interp(target_wavelength_um, wavelength_um, extinction)
    return (n_interp + 1j * k_interp) ** 2


def record_slug(record_path):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", record_path).strip("_")


def save_case(
    path,
    record_path,
    true_epsilon,
    clean_eta_vector,
    target_eta_vector,
    details,
):
    final_epsilon = details["final_eps_sub"]
    np.savez_compressed(
        path,
        record_path=record_path,
        wavenumber=wavenumber,
        true_epsilon=true_epsilon,
        clean_eta_vector=clean_eta_vector,
        target_eta_vector=target_eta_vector,
        one_peak_epsilon=details["joint_eps_sub"],
        one_peak_eta_vector=details["joint_eta_vector"],
        pre_kk_epsilon=details["pre_kk_eps_sub"],
        pre_kk_eta_vector=details["pre_kk_eta_vector"],
        final_epsilon=final_epsilon,
        final_eta_vector=details["final_eta_vector"],
        selected_peak_count=details["selected_total_peak_count"],
        pre_kk_residual_norm=details["pre_kk_residual_norm"],
        final_residual_norm=details["final_residual_norm"],
        epsilon_relative_l2=(
            np.linalg.norm(final_epsilon - true_epsilon) / np.linalg.norm(true_epsilon)
        ),
        kk_status=details["kk_status"],
        kk_nfev=details["kk_nfev"],
        elapsed_seconds=details["joint_seconds"] + details["kk_seconds"],
    )


def example_test(
    model_path=MODEL_PATH,
    archive_path=DATA_ARCHIVE,
    noise_fraction=0.0,
    seed=20260712,
    plot=True,
):
    seed = round(np.random.random() * 1e8)
    """Evaluate one randomly selected dataset from the active split."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"CSV archive not found: {archive_path}")
    active_records = ACTIVE_RECORDS
    if not active_records:
        raise RuntimeError(f"No records are configured for the {SPLIT_NAME} split.")

    rng = np.random.default_rng(seed)
    selected_position = int(rng.integers(len(active_records)))
    global_index, record_path = active_records[selected_position]
    true_epsilon = load_polyanskiy_epsilon(record_path, archive_path)
    clean_eta_vector = forward_function(true_epsilon)
    noise_std = noise_fraction * float(np.mean(np.abs(clean_eta_vector)))
    target_eta_vector = clean_eta_vector + rng.normal(
        0.0, noise_std, size=clean_eta_vector.shape
    )
    details = eta_to_epsilon(
        target_eta_vector, model_path=model_path, return_details=True
    )
    case = {
        "record_path": record_path,
        "true_epsilon": true_epsilon,
        "target_eta_vector": target_eta_vector,
        "one_peak_epsilon": details["joint_eps_sub"],
        "one_peak_eta_vector": details["joint_eta_vector"],
        "pre_kk_epsilon": details["pre_kk_eps_sub"],
        "pre_kk_eta_vector": details["pre_kk_eta_vector"],
        "final_epsilon": details["final_eps_sub"],
        "final_eta_vector": details["final_eta_vector"],
        "selected_peak_count": details["selected_total_peak_count"],
        "pre_kk_residual_norm": details["pre_kk_residual_norm"],
        "final_residual_norm": details["final_residual_norm"],
    }
    print(
        f"Example {SPLIT_NAME} dataset [{global_index}]: {record_path}\n"
        f"  selected peaks={int(case['selected_peak_count'])}, "
        f"pre-KK norm={float(case['pre_kk_residual_norm']):.6e}, "
        f"final norm={float(case['final_residual_norm']):.6e}"
    )
    if plot:
        plot_case(case, EXAMPLE_PLOT_PATH, "random example")
        print(f"Saved example plot: {EXAMPLE_PLOT_PATH}")
    return case


def load_case(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def plot_case(case, output_path, rank_label):
    target_eta = vector_to_eta(case["target_eta_vector"])
    one_peak_eta = vector_to_eta(case["one_peak_eta_vector"])
    pre_kk_eta = vector_to_eta(case["pre_kk_eta_vector"])
    final_eta = vector_to_eta(case["final_eta_vector"])
    record_path = str(case["record_path"])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
    figure.suptitle(
        f"{rank_label.title()} non-DL recovery: {record_path}\n"
        f"selected peaks={int(case['selected_peak_count'])}, "
        f"pre-KK norm={float(case['pre_kk_residual_norm']):.4e}, "
        f"final norm={float(case['final_residual_norm']):.4e}"
    )
    eta_traces = (
        (target_eta, "FDM target", "black", "-"),
        (one_peak_eta, "One-peak fit", "tab:blue", ":"),
        (pre_kk_eta, "Best multi-peak fit", "tab:orange", "--"),
        (final_eta, "KK-refined recovery", "tab:green", "-"),
    )
    epsilon_traces = (
        (case["true_epsilon"], "Literature target", "black", "-"),
        (case["one_peak_epsilon"], "One-peak fit", "tab:blue", ":"),
        (case["pre_kk_epsilon"], "Best multi-peak fit", "tab:orange", "--"),
        (case["final_epsilon"], "KK-refined recovery", "tab:green", "-"),
    )
    for values, label, color, linestyle in eta_traces:
        axes[0, 0].plot(wavenumber, values.real, label=label, color=color, linestyle=linestyle)
        axes[0, 1].plot(wavenumber, values.imag, label=label, color=color, linestyle=linestyle)
    for values, label, color, linestyle in epsilon_traces:
        axes[1, 0].plot(wavenumber, values.real, label=label, color=color, linestyle=linestyle)
        axes[1, 1].plot(wavenumber, values.imag, label=label, color=color, linestyle=linestyle)

    axes[0, 0].set_title(r"$\mathrm{Re}(\eta)$")
    axes[0, 1].set_title(r"$\mathrm{Im}(\eta)$")
    axes[1, 0].set_title(r"$\mathrm{Re}(\epsilon)$")
    axes[1, 1].set_title(r"$\mathrm{Im}(\epsilon)$")
    axes[0, 0].set_ylabel(r"Normalized near-field signal $\eta$")
    axes[1, 0].set_ylabel("Dielectric function")
    axes[1, 0].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    axes[1, 1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    for axis in axes.flat:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run_non_dl_test(
    archive_path=DATA_ARCHIVE,
    model_path=MODEL_PATH,
    noise_fraction=0.0,
    seed=20260712,
    force=False,
    start_index=1,
    stop_index=None,
    finalize=True,
):
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"CSV archive not found: {archive_path}")
    CASE_DIRECTORY.mkdir(exist_ok=True)
    if stop_index is None:
        stop_index = len(POLYANSKIY_RECORDS)
    cases = []
    failures = []

    active_records = ACTIVE_RECORDS
    for active_index, (index, record_path) in enumerate(active_records, start=1):
        if index < start_index or index > stop_index:
            continue
        case_path = CASE_DIRECTORY / f"{index:02d}_{record_slug(record_path)}.npz"
        if case_path.exists() and not force:
            cases.append(load_case(case_path))
            print(f"[{active_index}/{len(active_records)}] cached {record_path}", flush=True)
            continue
        try:
            true_epsilon = load_polyanskiy_epsilon(record_path, archive_path)
            clean_eta_vector = forward_function(true_epsilon)
            noise_std = noise_fraction * float(np.mean(np.abs(clean_eta_vector)))
            rng = np.random.default_rng(seed + index)
            target_eta_vector = clean_eta_vector + rng.normal(
                0.0,
                noise_std,
                size=clean_eta_vector.shape,
            )
            details = eta_to_epsilon(
                target_eta_vector,
                model_path=model_path,
                return_details=True,
            )
            save_case(
                case_path,
                record_path,
                true_epsilon,
                clean_eta_vector,
                target_eta_vector,
                details,
            )
            case = load_case(case_path)
            cases.append(case)
            print(
                f"[{active_index}/{len(active_records)}] {record_path} | "
                f"peaks={int(case['selected_peak_count'])} | "
                f"pre={float(case['pre_kk_residual_norm']):.4e} | "
                f"final={float(case['final_residual_norm']):.4e}",
                flush=True,
            )
        except Exception as error:
            failures.append((record_path, f"{type(error).__name__}: {error}"))
            print(f"FAILED {record_path}: {error}", flush=True)

    if not cases:
        raise RuntimeError("No benchmark cases completed successfully.")
    if not finalize:
        return cases

    final_norms = np.array([float(case["final_residual_norm"]) for case in cases])
    ranking = np.argsort(final_norms)
    chosen = {
        "best": int(ranking[0]),
        "median": int(ranking[len(ranking) // 2]),
        "worst": int(ranking[-1]),
    }
    for label, case_index in chosen.items():
        plot_case(cases[case_index], PLOT_PATHS[label], label)

    np.savez_compressed(
        SUMMARY_PATH,
        record_paths=np.array([str(case["record_path"]) for case in cases]),
        selected_peak_count=np.array([int(case["selected_peak_count"]) for case in cases]),
        pre_kk_residual_norm=np.array([float(case["pre_kk_residual_norm"]) for case in cases]),
        final_residual_norm=final_norms,
        epsilon_relative_l2=np.array([float(case["epsilon_relative_l2"]) for case in cases]),
        kk_nfev=np.array([int(case["kk_nfev"]) for case in cases]),
        elapsed_seconds=np.array([float(case["elapsed_seconds"]) for case in cases]),
        failed_record_paths=np.array([item[0] for item in failures]),
        failure_messages=np.array([item[1] for item in failures]),
    )
    print(f"Completed {len(cases)}/{len(active_records)} {SPLIT_NAME} records.")
    for label, case_index in chosen.items():
        print(
            f"{label}: {cases[case_index]['record_path']} | "
            f"norm={float(cases[case_index]['final_residual_norm']):.6e} | "
            f"plot={PLOT_PATHS[label]}"
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DATA_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--noise-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--no-finalize", action="store_true")
    args = parser.parse_args()
    # run_non_dl_test(
    #     archive_path=args.archive,
    #     model_path=args.model_path,
    #     noise_fraction=args.noise_fraction,
    #     seed=args.seed,
    #     force=args.force,
    #     start_index=args.start_index,
    #     stop_index=args.stop_index,
    #     finalize=not args.no_finalize,
    # )

    example_test()


if __name__ == "__main__":
    main()
