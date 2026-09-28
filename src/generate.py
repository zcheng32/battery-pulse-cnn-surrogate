"""Generate grouped SPM pulse transitions and independent rollout references."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time

import numpy as np
import pandas as pd

from common import LOCAL, RESULTS, ROOT, load_config, now_utc, write_json
from physics import Cell


def make_history(rng, config, initial_soc, target_soc, capacity_Ah):
    removed_soc = initial_soc - target_soc
    fractions = rng.dirichlet([2.0, 2.0, 2.0])
    currents = rng.uniform(*config["history_current_range_A"], size=3)
    if target_soc < config["history_low_soc_switch"]:
        currents[2] = config["history_low_soc_last_current_A"]
        fractions = np.array([0.2, 0.2, 0.6])
    durations = removed_soc * capacity_Ah * 3600.0 * fractions / currents
    return [
        (float(currents[0]), float(durations[0])),
        (0.0, float(rng.uniform(*config["history_mid_rest_range_s"]))),
        (float(currents[1]), float(durations[1])),
        (float(currents[2]), float(durations[2])),
        (0.0, float(rng.choice(config["history_final_rest_choices_s"]))),
    ]


def select_rollout_histories(history_frame: pd.DataFrame, config: dict):
    eligible = history_frame[
        history_frame["split"].eq("test")
        & history_frame["target_soc"].ge(config["rollout_minimum_soc"])
    ].sort_values(["target_soc", "history_id"])
    count = int(config["rollout_case_count"])
    if len(eligible) < count:
        raise RuntimeError("Not enough eligible test histories for the frozen rollout rule.")
    positions = np.linspace(0, len(eligible) - 1, count).round().astype(int)
    return eligible.iloc[positions].drop_duplicates("history_id")


def generate_rollouts(history_frame, history_records, config):
    selected = select_rollout_histories(history_frame, config)
    currents = np.asarray(config["rollout_currents_A"], dtype=float)
    states = []
    voltages = []
    solve_times = []
    history_ids = []
    records = {record["history_id"]: record for record in history_records}
    for row in selected.itertuples(index=False):
        record = records[row.history_id]
        cell = Cell(record["initial_soc"], config)
        solution = cell.run_history(record["segments"])
        case_states = [cell.concentration_state(solution)]
        case_voltages = []
        case_times = []
        for current in currents:
            transition = cell.pulse_transition(solution, float(current))
            if transition["status"] != "complete":
                raise RuntimeError(
                    f"Frozen rollout {row.history_id} cut off at {current:g} A; revise before training."
                )
            case_voltages.append(transition["voltage_V"])
            case_states.append(transition["state1"])
            case_times.append(transition["solve_s"])
            solution = transition["solution"]
        history_ids.append(row.history_id)
        states.append(np.stack(case_states))
        voltages.append(np.stack(case_voltages))
        solve_times.append(case_times)
    np.savez_compressed(
        LOCAL / "rollout_reference.npz",
        history_ids=np.asarray(history_ids),
        currents_A=currents,
        states=np.stack(states),
        voltage_V=np.stack(voltages),
        solve_times_s=np.asarray(solve_times),
    )
    return history_ids


def main():
    config, config_id = load_config()
    if (LOCAL / "dataset.npz").exists():
        raise RuntimeError("Existing local dataset is protected; use a new directory or archive it first.")
    LOCAL.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rng = np.random.default_rng(config["history_seed"])
    split_rng = np.random.default_rng(config["split_seed"])
    nodes = int(config["radial_nodes_per_particle"])
    sample_count = int(round(config["pulse_duration_s"] / config["sample_interval_s"])) + 1

    state0_rows = []
    state1_rows = []
    voltage_rows = []
    current_rows = []
    complete_rows = []
    branch_rows = []
    history_rows = []
    history_records = []
    history_index = 0

    edges = config["soc_stratum_edges"]
    for stratum, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        splits = split_rng.permutation(config["split_per_stratum"])
        for split in splits:
            history_id = f"H{history_index:03d}"
            history_index += 1
            target_soc = float(rng.uniform(low, high))
            removed_soc = float(rng.uniform(*config["history_soc_removed_range"]))
            initial_soc = target_soc + removed_soc
            cell = Cell(initial_soc, config)
            segments = make_history(rng, config, initial_soc, target_soc, cell.capacity_Ah)
            history_solution = cell.run_history(segments)
            state0 = cell.concentration_state(history_solution)
            actual_soc = cell.soc_from_state(state0)
            if abs(actual_soc - target_soc) > 2e-5:
                raise RuntimeError(f"History {history_id} missed target SOC: {actual_soc} vs {target_soc}.")
            record = {
                "history_id": history_id,
                "split": str(split),
                "stratum": stratum,
                "initial_soc": initial_soc,
                "target_soc": target_soc,
                "actual_soc": actual_soc,
                "segments": segments,
            }
            history_records.append(record)
            history_rows.append(
                {
                    **{k: v for k, v in record.items() if k != "segments"},
                    "segments_A_s": json.dumps(segments),
                    "config_id": config_id,
                }
            )

            statuses = []
            for current_A in config["candidate_currents_A"]:
                transition = cell.pulse_transition(history_solution, float(current_A))
                statuses.append(transition["status"])
                if transition["status"] == "solver_failure":
                    state1 = np.full((2, nodes), np.nan)
                    voltage = np.full(sample_count, np.nan)
                elif transition["status"] == "physical_cutoff":
                    state1 = np.full((2, nodes), np.nan)
                    voltage = np.full(sample_count, np.nan)
                else:
                    state1 = transition["state1"]
                    voltage = transition["voltage_V"]
                branch_id = f"{history_id}_I{float(current_A):g}"
                state0_rows.append(state0)
                state1_rows.append(state1)
                voltage_rows.append(voltage)
                current_rows.append(float(current_A))
                complete_rows.append(transition["status"] == "complete")
                branch_rows.append(
                    {
                        "branch_id": branch_id,
                        "history_id": history_id,
                        "split": str(split),
                        "current_A": float(current_A),
                        "status": transition["status"],
                        "event": transition["event"],
                        "end_s": transition["end_s"],
                        "solve_s": transition["solve_s"],
                        "config_id": config_id,
                    }
                )
            print(f"{history_id} {str(split):5s} SOC={actual_soc:.4f} {statuses}", flush=True)

    if history_index != config["history_count"]:
        raise RuntimeError(f"Generated {history_index} histories; expected {config['history_count']}.")
    history_frame = pd.DataFrame(history_rows)
    branch_frame = pd.DataFrame(branch_rows)
    history_frame.to_csv(ROOT / "split_manifest.csv", index=False)
    branch_frame.to_csv(RESULTS / "dataset_manifest.csv", index=False)
    write_json(LOCAL / "histories.json", history_records)
    np.savez_compressed(
        LOCAL / "dataset.npz",
        state0=np.stack(state0_rows),
        state1=np.stack(state1_rows),
        voltage_V=np.stack(voltage_rows),
        current_A=np.asarray(current_rows),
        complete=np.asarray(complete_rows),
    )
    rollout_ids = generate_rollouts(history_frame, history_records, config)

    versions = {
        package: importlib.metadata.version(package)
        for package in [
            "pybamm",
            "pybammsolvers",
            "torch",
            "numpy",
            "scipy",
            "scikit-learn",
            "pandas",
            "matplotlib",
            "joblib",
            "plotly",
        ]
    }
    status_counts = (
        branch_frame.groupby(["split", "status"]).size().reset_index(name="count").to_dict("records")
    )
    metadata = {
        "experiment": config["experiment"],
        "config_id": config_id,
        "generated_utc": now_utc(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "solver": "PyBaMM IDAKLU, one thread",
        "histories": len(history_frame),
        "branches": len(branch_frame),
        "history_split_counts": history_frame["split"].value_counts().to_dict(),
        "branch_status_counts": status_counts,
        "solver_failures": int(branch_frame["status"].eq("solver_failure").sum()),
        "rollout_history_ids": rollout_ids,
        "generation_wall_s": time.perf_counter() - started,
        "scope": "Synthetic SPM reference data; fixed Chen2020 parameters; no aging or real-cell validation.",
    }
    write_json(RESULTS / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
