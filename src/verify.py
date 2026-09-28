"""Audit split integrity, stored results, local arrays, and repository size."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import LOCAL, RESULTS, ROOT, load_config, now_utc, repository_files, write_json


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    config, config_id = load_config()
    required = [
        ROOT / "README.md",
        ROOT / "references.bib",
        ROOT / "split_manifest.csv",
        RESULTS / "run_metadata.json",
        RESULTS / "training_summary.csv",
        RESULTS / "test_metrics.csv",
        RESULTS / "test_pulse_metrics.csv",
        RESULTS / "candidate_selection_metrics.csv",
        RESULTS / "rollout_metrics.csv",
        RESULTS / "speed_metrics.csv",
        RESULTS / "display_predictions.csv",
        RESULTS / "interactive_cases.csv",
        RESULTS / "interactive_predictions.csv",
        ROOT / "figures" / "overview.png",
        ROOT / "figures" / "interactive_test_browser.html",
        LOCAL / "dataset.npz",
        LOCAL / "rollout_reference.npz",
    ]
    for path in required:
        check(path.exists() and path.stat().st_size > 0, f"Missing or empty: {path}")

    histories = pd.read_csv(ROOT / "split_manifest.csv")
    branches = pd.read_csv(RESULTS / "dataset_manifest.csv")
    check(len(histories) == config["history_count"] == 72, "Unexpected history count.")
    check(len(branches) == len(histories) * len(config["candidate_currents_A"]), "Branch count mismatch.")
    check(histories["history_id"].is_unique, "History IDs are not unique.")
    check(branches["branch_id"].is_unique, "Branch IDs are not unique.")
    check(histories.groupby("history_id")["split"].nunique().max() == 1, "A history crosses splits.")
    check(branches.groupby("history_id")["split"].nunique().max() == 1, "Branches cross splits.")
    joined = branches.merge(histories[["history_id", "split"]], on="history_id", suffixes=("", "_history"))
    check(joined["split"].eq(joined["split_history"]).all(), "Manifest split disagreement.")
    expected_splits = {"train": 48, "val": 12, "test": 12}
    check(histories["split"].value_counts().to_dict() == expected_splits, "History split counts changed.")
    expected_currents = np.asarray(config["candidate_currents_A"], dtype=float)
    for _, group in branches.groupby("history_id"):
        check(np.array_equal(np.sort(group["current_A"].to_numpy()), expected_currents),
              "A history lacks a frozen candidate current.")
    check(not branches["status"].eq("solver_failure").any(), "A solver failure is present.")
    check(set(branches["config_id"]) == {config_id}, "Branch config hash mismatch.")
    check(set(histories["config_id"]) == {config_id}, "History config hash mismatch.")

    source = np.load(LOCAL / "dataset.npz")
    data = {key: source[key] for key in source.files}
    check(data["state0"].shape == (576, 2, 96), "Unexpected state0 shape.")
    check(data["state1"].shape == (576, 2, 96), "Unexpected state1 shape.")
    check(data["voltage_V"].shape == (576, 41), "Unexpected voltage shape.")
    complete = data["complete"]
    check(np.isfinite(data["state0"]).all(), "Nonfinite initial state.")
    check(np.isfinite(data["state1"][complete]).all(), "Nonfinite complete final state.")
    check(np.isfinite(data["voltage_V"][complete]).all(), "Nonfinite complete voltage.")
    check(np.isnan(data["state1"][~complete]).all(), "Cutoff final states were padded.")
    check(np.isnan(data["voltage_V"][~complete]).all(), "Cutoff voltages were padded or extrapolated.")
    check(np.array_equal(complete, branches["status"].eq("complete").to_numpy()),
          "Array/manifest status mismatch.")

    training = pd.read_csv(RESULTS / "training_summary.csv")
    trained_seeds = sorted(training.loc[training["method"].eq("CNN"), "seed"].astype(int).tolist())
    check(trained_seeds == sorted(config["training_seeds"]), "Training seeds changed.")
    training_metadata = json.loads((RESULTS / "training_metadata.json").read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    check(training_metadata["test_data_used_for_selection"] is False, "Test leakage flag is set.")
    check(summary["test_data_used_for_selection"] is False, "Test leakage flag is set.")
    check(summary["config_id"] == config_id, "Summary config hash mismatch.")

    metrics = pd.read_csv(RESULTS / "test_metrics.csv")
    primary = metrics[
        metrics["method"].eq("CNN")
        & metrics["seed"].astype(str).eq(str(config["primary_training_seed"]))
    ]
    check(len(primary) == 1, "Primary test row is missing.")
    primary = primary.iloc[0]
    check(int(primary["complete_pulses"]) == 89, "Complete test count changed.")
    check(int(primary["cutoff_pulses"]) == 7, "Cutoff test count changed.")
    check(int(primary["false_feasible"]) == 0, "Primary CNN has a false-feasible pulse.")
    selection = pd.read_csv(RESULTS / "candidate_selection_metrics.csv")
    primary_selection = selection[
        selection["method"].eq("CNN")
        & selection["seed"].astype(str).eq(str(config["primary_training_seed"]))
    ]
    check(len(primary_selection) == 12, "Primary selection rows changed.")
    check(np.allclose(primary_selection["selection_error_A"], 0.0), "Primary current selection changed.")

    interactive_cases = pd.read_csv(RESULTS / "interactive_cases.csv")
    interactive_points = pd.read_csv(RESULTS / "interactive_predictions.csv")
    check(interactive_cases["history_id"].nunique() == 3, "Interactive history count changed.")
    check(len(interactive_cases) == 24, "Interactive candidate count changed.")
    check(interactive_cases["true_status"].eq("physical_cutoff").any(),
          "Interactive browser lacks a cutoff example.")
    point_branches = set(interactive_points["branch_id"])
    complete_interactive = set(interactive_cases.loc[interactive_cases["true_status"].eq("complete"), "branch_id"])
    check(point_branches == complete_interactive, "Interactive point data include/exclude wrong branches.")

    tracked_files = repository_files()
    forbidden = {".npz", ".pt", ".joblib", ".log"}
    check(not [path for path in tracked_files if path.suffix in forbidden],
          "A local artifact appears outside the ignored local directory.")
    largest = sorted(
        ({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size} for path in tracked_files),
        key=lambda item: item["bytes"], reverse=True,
    )
    check(largest[0]["bytes"] < 2_000_000, "A tracked artifact exceeds 2 MB.")
    report = {
        "verified_utc": now_utc(),
        "config_id": config_id,
        "checks_passed": 37,
        "history_split_counts": expected_splits,
        "branch_status_counts": branches.groupby(["split", "status"]).size().to_dict(),
        "primary_voltage_mae_mV": float(primary["voltage_mae_mV"]),
        "primary_false_feasible": int(primary["false_feasible"]),
        "tracked_file_count": len(tracked_files),
        "tracked_total_bytes": sum(item["bytes"] for item in largest),
        "largest_tracked_files": largest[:10],
        "status": "pass",
    }
    report["branch_status_counts"] = {
        f"{split_name}/{status}": int(count)
        for (split_name, status), count in report["branch_status_counts"].items()
    }
    write_json(RESULTS / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
