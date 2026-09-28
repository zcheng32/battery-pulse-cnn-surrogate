"""Generate grouped continuous-pulse sequences for online state correction."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from common import LOCAL, RESULTS, ROOT, now_utc, read_json, sha256, write_json
from physics import Cell


CORRECTION_CONFIG = ROOT / "correction_config.json"


def main():
    config = read_json(ROOT / "config.json")
    correction = read_json(CORRECTION_CONFIG)
    correction_id = sha256(CORRECTION_CONFIG)
    output = LOCAL / "correction_sequences.npz"
    if output.exists():
        raise RuntimeError("Existing correction sequence data are protected; archive them before regenerating.")
    histories = read_json(LOCAL / "histories.json")
    rng = np.random.default_rng(correction["sequence_seed"])
    choices = np.asarray(correction["current_choices_A"], dtype=float)
    probabilities = np.asarray(correction["current_probabilities"], dtype=float)
    probabilities = probabilities / probabilities.sum()
    repeats = int(correction["sequences_per_history"])
    steps = int(correction["sequence_length"])
    nodes = int(config["radial_nodes_per_particle"])
    points = int(round(config["pulse_duration_s"] / config["sample_interval_s"])) + 1
    sequence_ids = []
    history_ids = []
    splits = []
    currents = []
    attempted = []
    complete = []
    states = []
    voltages = []
    solve_times = []
    sequence_rows = []
    step_rows = []
    started = time.perf_counter()

    for history in histories:
        for repeat in range(repeats):
            sequence_id = f"{history['history_id']}_S{repeat}"
            schedule = rng.choice(choices, size=steps, replace=True, p=probabilities).astype(float)
            cell = Cell(history["initial_soc"], config)
            solution = cell.run_history(history["segments"])
            state_array = np.full((steps + 1, 2, nodes), np.nan)
            voltage_array = np.full((steps, points), np.nan)
            attempted_array = np.zeros(steps, dtype=bool)
            complete_array = np.zeros(steps, dtype=bool)
            solve_array = np.full(steps, np.nan)
            state_array[0] = cell.concentration_state(solution)
            termination = "sequence_length"
            completed_steps = 0
            for step, current_A in enumerate(schedule):
                attempted_array[step] = True
                transition = cell.pulse_transition(solution, float(current_A))
                solve_array[step] = transition["solve_s"]
                complete_array[step] = transition["status"] == "complete"
                step_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "history_id": history["history_id"],
                        "split": history["split"],
                        "step": step + 1,
                        "elapsed_s": (step + 1) * config["pulse_duration_s"],
                        "current_A": current_A,
                        "status": transition["status"],
                        "event": transition["event"],
                        "end_s": transition["end_s"],
                        "solve_s": transition["solve_s"],
                        "correction_config_id": correction_id,
                    }
                )
                if transition["status"] != "complete":
                    termination = transition["status"]
                    break
                voltage_array[step] = transition["voltage_V"]
                state_array[step + 1] = transition["state1"]
                solution = transition["solution"]
                completed_steps += 1
            sequence_ids.append(sequence_id)
            history_ids.append(history["history_id"])
            splits.append(history["split"])
            currents.append(schedule)
            attempted.append(attempted_array)
            complete.append(complete_array)
            states.append(state_array)
            voltages.append(voltage_array)
            solve_times.append(solve_array)
            sequence_rows.append(
                {
                    "sequence_id": sequence_id,
                    "history_id": history["history_id"],
                    "split": history["split"],
                    "initial_soc": history["actual_soc"],
                    "completed_steps": completed_steps,
                    "attempted_steps": int(attempted_array.sum()),
                    "termination": termination,
                    "current_schedule_A": json.dumps(schedule.tolist()),
                    "correction_config_id": correction_id,
                }
            )
            print(
                f"{sequence_id} {history['split']:5s} complete={completed_steps:2d}/{steps} "
                f"termination={termination}",
                flush=True,
            )

    np.savez_compressed(
        output,
        sequence_ids=np.asarray(sequence_ids),
        history_ids=np.asarray(history_ids),
        splits=np.asarray(splits),
        currents_A=np.stack(currents),
        attempted=np.stack(attempted),
        complete=np.stack(complete),
        states=np.stack(states),
        voltage_V=np.stack(voltages),
        solve_times_s=np.stack(solve_times),
    )
    sequence_frame = pd.DataFrame(sequence_rows)
    step_frame = pd.DataFrame(step_rows)
    sequence_frame.to_csv(RESULTS / "correction_sequence_manifest.csv", index=False)
    step_frame.to_csv(RESULTS / "correction_step_manifest.csv", index=False)
    metadata = {
        "experiment": correction["experiment"],
        "correction_config_id": correction_id,
        "generated_utc": now_utc(),
        "sequences": len(sequence_frame),
        "sequence_split_counts": sequence_frame["split"].value_counts().to_dict(),
        "attempted_steps": int(step_frame.shape[0]),
        "complete_steps": int(step_frame["status"].eq("complete").sum()),
        "physical_cutoffs": int(step_frame["status"].eq("physical_cutoff").sum()),
        "solver_failures": int(step_frame["status"].eq("solver_failure").sum()),
        "generation_wall_s": time.perf_counter() - started,
        "scope": correction["scope_note"],
    }
    write_json(RESULTS / "correction_data_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
