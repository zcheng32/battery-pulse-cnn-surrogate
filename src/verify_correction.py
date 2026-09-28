"""Audit online-correction data, split isolation, metrics, and deliverables."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import LOCAL, RESULTS, ROOT, now_utc, read_json, repository_files, sha256, write_json


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    config_path = ROOT / "correction_config.json"
    config = read_json(config_path)
    config_id = sha256(config_path)
    required = [
        ROOT / "ONLINE_CORRECTION.md",
        ROOT / "figures" / "online_correction.png",
        ROOT / "figures" / "online_correction_browser.html",
        RESULTS / "correction_data_metadata.json",
        RESULTS / "correction_training_metadata.json",
        RESULTS / "correction_training_summary.csv",
        RESULTS / "correction_sequence_manifest.csv",
        RESULTS / "correction_step_manifest.csv",
        RESULTS / "correction_test_metrics.csv",
        RESULTS / "correction_step_metrics.csv",
        RESULTS / "correction_sequence_metrics.csv",
        RESULTS / "correction_classification_metrics.csv",
        RESULTS / "correction_display_predictions.csv",
        RESULTS / "correction_display_selection.json",
        RESULTS / "correction_speed_metrics.csv",
        RESULTS / "correction_summary.json",
        LOCAL / "correction_sequences.npz",
        LOCAL / "correction_preprocessing.npz",
        LOCAL / "correction_training_examples.npz",
        LOCAL / "correction_ridge.joblib",
        LOCAL / "correction_test_predictions.npz",
    ]
    for path in required:
        check(path.exists() and path.stat().st_size > 0, f"Missing or empty: {path}")

    base_split = pd.read_csv(ROOT / "split_manifest.csv")
    sequences = pd.read_csv(RESULTS / "correction_sequence_manifest.csv")
    steps = pd.read_csv(RESULTS / "correction_step_manifest.csv")
    check(len(sequences) == 216, "Unexpected sequence count.")
    check(sequences["sequence_id"].is_unique, "Sequence IDs are not unique.")
    check(
        sequences.groupby("history_id").size().eq(config["sequences_per_history"]).all(),
        "A history does not have the configured number of sequences.",
    )
    joined = sequences.merge(
        base_split[["history_id", "split"]],
        on="history_id",
        suffixes=("", "_base"),
    )
    check(joined["split"].eq(joined["split_base"]).all(), "Correction/base split mismatch.")
    split_counts = sequences["split"].value_counts().to_dict()
    check(split_counts == {"train": 144, "val": 36, "test": 36}, "Sequence split counts changed.")
    check(set(sequences["correction_config_id"]) == {config_id}, "Sequence config hash mismatch.")
    check(set(steps["correction_config_id"]) == {config_id}, "Step config hash mismatch.")
    check(not steps["status"].eq("solver_failure").any(), "A solver failure is present.")
    check(len(steps) == 2296, "Attempted-step count changed.")
    check(steps["status"].eq("complete").sum() == 2250, "Complete-step count changed.")
    check(steps["status"].eq("physical_cutoff").sum() == 46, "Cutoff count changed.")

    source = np.load(LOCAL / "correction_sequences.npz")
    data = {key: source[key] for key in source.files}
    check(data["states"].shape == (216, 13, 2, 96), "Unexpected state-array shape.")
    check(data["voltage_V"].shape == (216, 12, 41), "Unexpected voltage-array shape.")
    check(data["currents_A"].shape == (216, 12), "Unexpected current-array shape.")
    check(data["attempted"].sum() == 2296, "Array attempted-step count changed.")
    check(data["complete"].sum() == 2250, "Array complete-step count changed.")
    check(np.isfinite(data["states"][:, 0]).all(), "Initial states contain nonfinite values.")
    complete = data["complete"]
    attempted_cutoff = data["attempted"] & ~complete
    next_states = data["states"][:, 1:]
    check(np.isfinite(data["voltage_V"][complete]).all(), "Complete voltages are nonfinite.")
    check(np.isfinite(next_states[complete]).all(), "Complete next states are nonfinite.")
    check(np.isnan(data["voltage_V"][attempted_cutoff]).all(), "Cutoff voltages were padded.")
    check(np.isnan(next_states[attempted_cutoff]).all(), "Cutoff next states were padded.")
    check(
        np.array_equal(data["sequence_ids"], sequences["sequence_id"].to_numpy()),
        "Array/manifest sequence order changed.",
    )

    data_metadata = read_json(RESULTS / "correction_data_metadata.json")
    training_metadata = read_json(RESULTS / "correction_training_metadata.json")
    summary = read_json(RESULTS / "correction_summary.json")
    for record in [data_metadata, training_metadata, summary]:
        check(record["correction_config_id"] == config_id, "Metadata config hash mismatch.")
    check(data_metadata["solver_failures"] == 0, "Metadata records a solver failure.")
    check(training_metadata["test_data_used_for_selection"] is False, "Training test-leakage flag is set.")
    check(summary["test_data_used_for_selection"] is False, "Evaluation test-leakage flag is set.")
    check(training_metadata["training_examples"] == 1509, "Training-example count changed.")
    check(training_metadata["validation_examples"] == 365, "Validation-example count changed.")
    check(training_metadata["test_examples_reserved"] == 376, "Reserved-test count changed.")
    check(training_metadata["pca_explained_variance"] > 0.999, "PCA coverage fell below 99.9%.")

    training = pd.read_csv(RESULTS / "correction_training_summary.csv")
    selected = training.sort_values(["validation_posterior_state_rmse", "method"]).iloc[0]
    check(selected["method"] == "Linear corrector", "Validation-selected method changed.")
    nonlinear_seeds = sorted(
        training.loc[training["method"].eq("Nonlinear corrector"), "seed"].astype(int).tolist()
    )
    check(nonlinear_seeds == sorted(config["corrector_seeds"]), "Corrector seeds changed.")

    metrics = pd.read_csv(RESULTS / "correction_test_metrics.csv")
    clean = metrics[metrics["measurement_noise_mV"].eq(0.0)].set_index("method")
    open_row = clean.loc["Open-loop"]
    linear_row = clean.loc["Linear corrector"]
    check(int(open_row["complete_steps"]) == 376, "Complete test count changed.")
    check(int(open_row["cutoff_attempts"]) == 7, "Test cutoff count changed.")
    check(linear_row["voltage_mae_mV"] < open_row["voltage_mae_mV"], "Voltage correction no longer improves MAE.")
    check(
        linear_row["posterior_state_rmse_fraction"] < open_row["posterior_state_rmse_fraction"],
        "State correction no longer improves RMSE.",
    )
    check(int(open_row["false_feasible"]) == 4, "Open-loop false-feasible count changed.")
    check(int(linear_row["false_feasible"]) == 3, "Corrected false-feasible count changed.")
    check(metrics["state_bound_clipped_fraction"].eq(0.0).all(), "A corrected state hit clipping bounds.")
    open_noise = metrics[metrics["method"].eq("Open-loop")].sort_values("measurement_noise_mV")
    check(open_noise["voltage_mae_mV"].nunique() == 1, "Open-loop result depends on unused measurement noise.")
    linear_5 = metrics[
        metrics["method"].eq("Linear corrector")
        & metrics["measurement_noise_mV"].eq(5.0)
    ].iloc[0]
    check(abs(linear_5["voltage_mae_mV"] - linear_row["voltage_mae_mV"]) < 0.1, "Noise robustness changed.")
    check(summary["primary_method_selected_by_validation"] == "Linear corrector", "Summary selection changed.")

    step_metrics = pd.read_csv(RESULTS / "correction_step_metrics.csv")
    clean_steps = step_metrics[step_metrics["measurement_noise_mV"].eq(0.0)]
    first = clean_steps[clean_steps["step"].eq(1)].set_index("method")
    check(first["voltage_mae_mV"].nunique() == 1, "Methods differ before the first correction.")
    last = clean_steps[clean_steps["step"].eq(12)].set_index("method")
    check(
        last.loc["Linear corrector", "posterior_state_rmse_fraction"]
        < last.loc["Open-loop", "posterior_state_rmse_fraction"],
        "Step-12 state drift is no longer reduced.",
    )

    display = pd.read_csv(RESULTS / "correction_display_predictions.csv")
    selection = read_json(RESULTS / "correction_display_selection.json")
    check(display["sequence_id"].nunique() == 1, "Display data contain multiple sequences.")
    check(display["sequence_id"].iloc[0] == selection["sequence_id"], "Display selection mismatch.")
    check(display["step"].nunique() == 12, "Display sequence is incomplete.")
    check(display["method"].nunique() == 3, "Display method count changed.")
    check(len(display) == 3 * 12 * 41, "Display point count changed.")

    tracked_files = repository_files()
    forbidden = {".npz", ".pt", ".joblib", ".log"}
    check(
        not [path for path in tracked_files if path.suffix in forbidden],
        "A local artifact appears outside the ignored local directory.",
    )
    largest = sorted(
        ({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size} for path in tracked_files),
        key=lambda item: item["bytes"],
        reverse=True,
    )
    check(largest[0]["bytes"] < 2_000_000, "A tracked artifact exceeds 2 MB.")

    report = {
        "verified_utc": now_utc(),
        "correction_config_id": config_id,
        "checks_passed": 51,
        "sequence_split_counts": split_counts,
        "complete_steps": int(complete.sum()),
        "physical_cutoffs": int(attempted_cutoff.sum()),
        "primary_method": "Linear corrector",
        "open_loop_voltage_mae_mV": float(open_row["voltage_mae_mV"]),
        "corrected_voltage_mae_mV": float(linear_row["voltage_mae_mV"]),
        "open_loop_state_rmse": float(open_row["posterior_state_rmse_fraction"]),
        "corrected_state_rmse": float(linear_row["posterior_state_rmse_fraction"]),
        "tracked_file_count": len(tracked_files),
        "tracked_total_bytes": sum(item["bytes"] for item in largest),
        "largest_tracked_files": largest[:10],
        "status": "pass",
    }
    write_json(RESULTS / "correction_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
