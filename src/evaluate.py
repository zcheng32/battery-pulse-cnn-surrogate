"""Evaluate one-step accuracy, cutoff decisions, recursive rollouts, and latency."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from common import LOCAL, RESULTS, ROOT, load_config, now_utc, write_json
from model import make_model


def load_scalers():
    source = np.load(LOCAL / "preprocessing.npz")
    return {key: source[key] for key in source.files}


def normalize_input(state, current, scalers):
    state_z = (state - scalers["state_mean"]) / scalers["state_scale"]
    current_z = (np.asarray(current) - scalers["current_mean"][0]) / scalers["current_scale"][0]
    return state_z.astype(np.float32), np.asarray(current_z, dtype=np.float32)


def cnn_predict(model, state, current, scalers):
    state_z, current_z = normalize_input(state, current, scalers)
    with torch.inference_mode():
        voltage_z, delta_z, cutoff_logit = model(
            torch.from_numpy(state_z), torch.from_numpy(current_z)
        )
    voltage = voltage_z.numpy() * scalers["voltage_scale"] + scalers["voltage_mean"]
    delta = delta_z.numpy() * scalers["delta_scale"] + scalers["delta_mean"]
    probability = torch.sigmoid(cutoff_logit).numpy()
    return voltage, state + delta, probability


def ridge_predict(bundle, state, current, scalers):
    state_z, current_z = normalize_input(state, current, scalers)
    features = np.concatenate([state_z.reshape(len(state_z), -1), current_z.reshape(-1, 1)], axis=1)
    output = bundle["regressor"].predict(features)
    points = scalers["voltage_mean"].shape[1]
    voltage = output[:, :points] * scalers["voltage_scale"] + scalers["voltage_mean"]
    delta_z = output[:, points:].reshape(-1, *state.shape[1:])
    delta = delta_z * scalers["delta_scale"] + scalers["delta_mean"]
    probability = bundle["classifier"].predict_proba(features)[:, 1]
    return voltage, state + delta, probability


def classification_counts(true_cutoff, probability, threshold):
    predicted = probability >= threshold
    true_cutoff = np.asarray(true_cutoff, dtype=bool)
    return {
        "cutoff_true_positive": int((predicted & true_cutoff).sum()),
        "cutoff_true_negative": int((~predicted & ~true_cutoff).sum()),
        "false_feasible": int((~predicted & true_cutoff).sum()),
        "false_infeasible": int((predicted & ~true_cutoff).sum()),
        "true_cutoff_count": int(true_cutoff.sum()),
        "true_complete_count": int((~true_cutoff).sum()),
    }


def aggregate_metrics(method, seed, voltage, state1, probability, threshold, truth, test_index):
    complete_local = truth["complete"][test_index]
    complete_global = test_index[complete_local]
    error = voltage[complete_local] - truth["voltage_V"][complete_global]
    state_error = state1[complete_local] - truth["state1"][complete_global]
    min_error = voltage[complete_local].min(axis=1) - truth["voltage_V"][complete_global].min(axis=1)
    counts = classification_counts(~complete_local, probability, threshold)
    sensitivity = counts["cutoff_true_positive"] / max(counts["true_cutoff_count"], 1)
    specificity = counts["cutoff_true_negative"] / max(counts["true_complete_count"], 1)
    return {
        "method": method,
        "seed": seed,
        "complete_pulses": int(complete_local.sum()),
        "cutoff_pulses": int((~complete_local).sum()),
        "voltage_mae_mV": 1000.0 * float(np.mean(np.abs(error))),
        "voltage_max_abs_mV": 1000.0 * float(np.max(np.abs(error))),
        "voltage_endpoint_mae_mV": 1000.0 * float(np.mean(np.abs(error[:, -1]))),
        "minimum_voltage_mae_mV": 1000.0 * float(np.mean(np.abs(min_error))),
        "maximum_overprediction_mV": 1000.0 * float(np.max(error)),
        "state_rmse_fraction": float(np.sqrt(np.mean(state_error**2))),
        "state_max_abs_fraction": float(np.max(np.abs(state_error))),
        "cutoff_balanced_accuracy": 0.5 * (sensitivity + specificity),
        **counts,
        "cutoff_threshold": float(threshold),
    }


def pulse_metric_rows(method, seed, voltage, state1, probability, threshold, truth, manifest, test_index):
    rows = []
    predicted_cutoff = probability >= threshold
    for local, global_index in enumerate(test_index):
        row = manifest.iloc[global_index]
        record = {
            "branch_id": row["branch_id"],
            "history_id": row["history_id"],
            "current_A": row["current_A"],
            "true_status": row["status"],
            "method": method,
            "seed": seed,
            "cutoff_probability": float(probability[local]),
            "predicted_status": "physical_cutoff" if predicted_cutoff[local] else "complete",
            "false_feasible": bool((not predicted_cutoff[local]) and (not truth["complete"][global_index])),
        }
        if truth["complete"][global_index]:
            error = voltage[local] - truth["voltage_V"][global_index]
            state_error = state1[local] - truth["state1"][global_index]
            record.update(
                {
                    "voltage_mae_mV": 1000.0 * float(np.mean(np.abs(error))),
                    "voltage_max_abs_mV": 1000.0 * float(np.max(np.abs(error))),
                    "endpoint_error_mV": 1000.0 * float(error[-1]),
                    "minimum_voltage_error_mV": 1000.0
                    * float(voltage[local].min() - truth["voltage_V"][global_index].min()),
                    "maximum_overprediction_mV": 1000.0 * float(np.max(error)),
                    "state_rmse_fraction": float(np.sqrt(np.mean(state_error**2))),
                }
            )
        else:
            record.update(
                {
                    "voltage_mae_mV": np.nan,
                    "voltage_max_abs_mV": np.nan,
                    "endpoint_error_mV": np.nan,
                    "minimum_voltage_error_mV": np.nan,
                    "maximum_overprediction_mV": np.nan,
                    "state_rmse_fraction": np.nan,
                }
            )
        rows.append(record)
    return rows


def candidate_selection_rows(method, seed, probability, threshold, manifest, test_index):
    frame = manifest.iloc[test_index][["history_id", "current_A", "status"]].copy()
    frame["predicted_complete"] = probability < threshold
    rows = []
    for history_id, group in frame.groupby("history_id", sort=True):
        true_currents = group.loc[group["status"].eq("complete"), "current_A"]
        predicted_currents = group.loc[group["predicted_complete"], "current_A"]
        true_max = float(true_currents.max()) if len(true_currents) else np.nan
        predicted_max = float(predicted_currents.max()) if len(predicted_currents) else np.nan
        rows.append(
            {
                "history_id": history_id,
                "method": method,
                "seed": seed,
                "true_case": (
                    "no_feasible_candidate"
                    if len(true_currents) == 0
                    else "all_candidates_feasible"
                    if len(true_currents) == len(group)
                    else "mixed"
                ),
                "predicted_case": (
                    "no_feasible_candidate"
                    if len(predicted_currents) == 0
                    else "all_candidates_feasible"
                    if len(predicted_currents) == len(group)
                    else "mixed"
                ),
                "true_max_feasible_current_A": true_max,
                "predicted_max_feasible_current_A": predicted_max,
                "selection_error_A": predicted_max - true_max
                if np.isfinite(predicted_max) and np.isfinite(true_max)
                else np.nan,
            }
        )
    return rows


def rollout(method, predictor, model, threshold, reference, scalers):
    records = []
    voltage_predictions = []
    state_predictions = []
    for case, history_id in enumerate(reference["history_ids"]):
        state = reference["states"][case, 0][None, ...]
        case_voltage = []
        case_states = [state[0].copy()]
        for step, current in enumerate(reference["currents_A"]):
            voltage, state1, probability = predictor(
                model, state, np.asarray([current]), scalers
            )
            true_voltage = reference["voltage_V"][case, step]
            true_state = reference["states"][case, step + 1]
            error = voltage[0] - true_voltage
            state_error = state1[0] - true_state
            records.append(
                {
                    "history_id": str(history_id),
                    "method": method,
                    "step": step + 1,
                    "elapsed_s": float((step + 1) * 10.0),
                    "current_A": float(current),
                    "voltage_mae_mV": 1000.0 * float(np.mean(np.abs(error))),
                    "voltage_max_abs_mV": 1000.0 * float(np.max(np.abs(error))),
                    "endpoint_error_mV": 1000.0 * float(error[-1]),
                    "state_rmse_fraction": float(np.sqrt(np.mean(state_error**2))),
                    "cutoff_probability": float(probability[0]),
                    "predicted_status": "physical_cutoff"
                    if probability[0] >= threshold
                    else "complete",
                }
            )
            case_voltage.append(voltage[0])
            case_states.append(state1[0])
            state = state1
        voltage_predictions.append(np.stack(case_voltage))
        state_predictions.append(np.stack(case_states))
    return records, np.stack(voltage_predictions), np.stack(state_predictions)


def benchmark(name, function, warmup, repeats):
    for _ in range(warmup):
        function()
    durations = np.empty(repeats)
    for index in range(repeats):
        started = time.perf_counter_ns()
        function()
        durations[index] = (time.perf_counter_ns() - started) * 1e-6
    return {
        "method": name,
        "median_latency_ms": float(np.median(durations)),
        "p95_latency_ms": float(np.quantile(durations, 0.95)),
        "repeats": int(repeats),
    }


def main():
    torch.set_num_threads(1)
    config, config_id = load_config()
    truth_source = np.load(LOCAL / "dataset.npz")
    truth = {key: truth_source[key] for key in truth_source.files}
    manifest = pd.read_csv(RESULTS / "dataset_manifest.csv")
    histories = pd.read_csv(ROOT / "split_manifest.csv")
    scalers = load_scalers()
    test_index = np.flatnonzero(manifest["split"].eq("test").to_numpy())
    test_state = truth["state0"][test_index]
    test_current = truth["current_A"][test_index]
    metric_rows = []
    pulse_rows = []
    selection_rows = []
    saved = {
        "test_index": test_index,
        "true_voltage_V": truth["voltage_V"][test_index],
        "true_state1": truth["state1"][test_index],
        "complete": truth["complete"][test_index],
    }
    models = {}

    for seed in config["training_seeds"]:
        checkpoint = torch.load(LOCAL / f"cnn_seed{seed}.pt", map_location="cpu", weights_only=True)
        model = make_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        voltage, state1, probability = cnn_predict(model, test_state, test_current, scalers)
        threshold = float(checkpoint["cutoff_threshold"])
        method = f"CNN seed {seed}"
        metric_rows.append(
            aggregate_metrics("CNN", seed, voltage, state1, probability, threshold, truth, test_index)
        )
        pulse_rows.extend(
            pulse_metric_rows("CNN", seed, voltage, state1, probability, threshold, truth, manifest, test_index)
        )
        selection_rows.extend(
            candidate_selection_rows("CNN", seed, probability, threshold, manifest, test_index)
        )
        saved[f"cnn_seed{seed}_voltage_V"] = voltage
        saved[f"cnn_seed{seed}_state1"] = state1
        saved[f"cnn_seed{seed}_cutoff_probability"] = probability
        models[method] = (cnn_predict, model, threshold)

    ridge = joblib.load(LOCAL / "ridge.joblib")
    ridge_voltage, ridge_state1, ridge_probability = ridge_predict(
        ridge, test_state, test_current, scalers
    )
    ridge_threshold = float(ridge["cutoff_threshold"])
    metric_rows.append(
        aggregate_metrics(
            "Ridge", "deterministic", ridge_voltage, ridge_state1, ridge_probability,
            ridge_threshold, truth, test_index
        )
    )
    pulse_rows.extend(
        pulse_metric_rows(
            "Ridge", "deterministic", ridge_voltage, ridge_state1, ridge_probability,
            ridge_threshold, truth, manifest, test_index
        )
    )
    selection_rows.extend(
        candidate_selection_rows(
            "Ridge", "deterministic", ridge_probability, ridge_threshold, manifest, test_index
        )
    )
    saved.update(
        {
            "ridge_voltage_V": ridge_voltage,
            "ridge_state1": ridge_state1,
            "ridge_cutoff_probability": ridge_probability,
        }
    )
    np.savez_compressed(LOCAL / "test_predictions.npz", **saved)

    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(RESULTS / "test_metrics.csv", index=False)
    pulse_frame = pd.DataFrame(pulse_rows)
    pulse_frame.to_csv(RESULTS / "test_pulse_metrics.csv", index=False)
    selection_frame = pd.DataFrame(selection_rows)
    selection_frame.to_csv(RESULTS / "candidate_selection_metrics.csv", index=False)

    reference_source = np.load(LOCAL / "rollout_reference.npz")
    reference = {key: reference_source[key] for key in reference_source.files}
    rollout_rows = []
    rollout_saved = {
        "history_ids": reference["history_ids"],
        "currents_A": reference["currents_A"],
        "true_voltage_V": reference["voltage_V"],
        "true_states": reference["states"],
    }
    for method, (predictor, model, threshold) in models.items():
        records, voltage, states = rollout(method, predictor, model, threshold, reference, scalers)
        rollout_rows.extend(records)
        key = method.lower().replace(" ", "_")
        rollout_saved[f"{key}_voltage_V"] = voltage
        rollout_saved[f"{key}_states"] = states
    records, voltage, states = rollout(
        "Ridge", ridge_predict, ridge, ridge_threshold, reference, scalers
    )
    rollout_rows.extend(records)
    rollout_saved["ridge_voltage_V"] = voltage
    rollout_saved["ridge_states"] = states
    pd.DataFrame(rollout_rows).to_csv(RESULTS / "rollout_metrics.csv", index=False)
    np.savez_compressed(LOCAL / "rollout_predictions.npz", **rollout_saved)

    primary = int(config["primary_training_seed"])
    primary_model = models[f"CNN seed {primary}"][1]
    speed_index = int(test_index[np.flatnonzero(truth["complete"][test_index])[0]])
    speed_state = truth["state0"][speed_index : speed_index + 1]
    speed_current = truth["current_A"][speed_index : speed_index + 1]
    warmup = int(config["speed_warmup_repeats"])
    repeats = int(config["speed_timed_repeats"])
    speed = [
        benchmark(
            f"CNN seed {primary}",
            lambda: cnn_predict(primary_model, speed_state, speed_current, scalers),
            warmup,
            repeats,
        ),
        benchmark(
            "Ridge",
            lambda: ridge_predict(ridge, speed_state, speed_current, scalers),
            warmup,
            repeats,
        ),
    ]
    solver_times_ms = (
        1000.0
        * manifest.loc[
            manifest["split"].eq("test") & manifest["status"].eq("complete"), "solve_s"
        ].to_numpy()
    )
    speed.append(
        {
            "method": "PyBaMM IDAKLU",
            "median_latency_ms": float(np.median(solver_times_ms)),
            "p95_latency_ms": float(np.quantile(solver_times_ms, 0.95)),
            "repeats": int(len(solver_times_ms)),
        }
    )
    speed_frame = pd.DataFrame(speed)
    reference_median = speed_frame.loc[
        speed_frame["method"].eq("PyBaMM IDAKLU"), "median_latency_ms"
    ].iloc[0]
    speed_frame["speedup_vs_pybamm_median"] = reference_median / speed_frame["median_latency_ms"]
    speed_frame.to_csv(RESULTS / "speed_metrics.csv", index=False)

    primary_pulses = pulse_frame[
        pulse_frame["method"].eq("CNN")
        & pulse_frame["seed"].astype(str).eq(str(primary))
        & pulse_frame["true_status"].eq("complete")
    ].sort_values(["voltage_mae_mV", "history_id", "current_A"])
    median_value = float(primary_pulses["voltage_mae_mV"].median())
    typical = primary_pulses.iloc[(primary_pulses["voltage_mae_mV"] - median_value).abs().argmin()]
    worst = primary_pulses.iloc[-1]
    selections = [
        ("typical", typical["branch_id"]),
        ("worst", worst["branch_id"]),
    ]
    time_grid = np.arange(truth["voltage_V"].shape[1]) * float(config["sample_interval_s"])
    display_rows = []
    primary_voltage = saved[f"cnn_seed{primary}_voltage_V"]
    for role, branch_id in selections:
        global_index = int(manifest.index[manifest["branch_id"].eq(branch_id)][0])
        local_index = int(np.flatnonzero(test_index == global_index)[0])
        history_id = manifest.loc[global_index, "history_id"]
        soc = float(histories.loc[histories["history_id"].eq(history_id), "actual_soc"].iloc[0])
        for method, prediction in [("CNN", primary_voltage), ("Ridge", ridge_voltage)]:
            for point, time_s in enumerate(time_grid):
                true_voltage = truth["voltage_V"][global_index, point]
                predicted = prediction[local_index, point]
                display_rows.append(
                    {
                        "case_role": role,
                        "branch_id": branch_id,
                        "history_id": history_id,
                        "initial_soc": soc,
                        "current_A": manifest.loc[global_index, "current_A"],
                        "time_s": time_s,
                        "method": method,
                        "pybamm_voltage_V": true_voltage,
                        "predicted_voltage_V": predicted,
                        "signed_error_mV": 1000.0 * (predicted - true_voltage),
                        "config_id": config_id,
                    }
                )
    pd.DataFrame(display_rows).to_csv(RESULTS / "display_predictions.csv", index=False)
    write_json(
        RESULTS / "display_selection.json",
        {
            "rule": config["display_rule"],
            "primary_seed": primary,
            "typical_branch_id": typical["branch_id"],
            "typical_curve_mae_mV": typical["voltage_mae_mV"],
            "worst_branch_id": worst["branch_id"],
            "worst_curve_mae_mV": worst["voltage_mae_mV"],
        },
    )

    cutoff_counts = (
        manifest.iloc[test_index]
        .assign(is_cutoff=lambda frame: frame["status"].eq("physical_cutoff"))
        .groupby("history_id")["is_cutoff"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    boundary_history = str(cutoff_counts[cutoff_counts.eq(cutoff_counts.max())].index.min())
    selected_histories = []
    for role, branch_id in selections:
        selected_histories.append((role, str(manifest.loc[manifest["branch_id"].eq(branch_id), "history_id"].iloc[0])))
    selected_histories.append(("cutoff_boundary", boundary_history))
    selected_histories = list(dict.fromkeys(selected_histories))
    interactive_cases = []
    interactive_points = []
    for role, history_id in selected_histories:
        history_rows = manifest.iloc[test_index]
        history_rows = history_rows[history_rows["history_id"].eq(history_id)].sort_values("current_A")
        soc = float(histories.loc[histories["history_id"].eq(history_id), "actual_soc"].iloc[0])
        for global_index, branch in history_rows.iterrows():
            local_index = int(np.flatnonzero(test_index == global_index)[0])
            cnn_predicted = (
                "physical_cutoff"
                if saved[f"cnn_seed{primary}_cutoff_probability"][local_index]
                >= models[f"CNN seed {primary}"][2]
                else "complete"
            )
            ridge_predicted = (
                "physical_cutoff"
                if ridge_probability[local_index] >= ridge_threshold
                else "complete"
            )
            interactive_cases.append(
                {
                    "case_role": role,
                    "branch_id": branch["branch_id"],
                    "history_id": history_id,
                    "initial_soc": soc,
                    "current_A": branch["current_A"],
                    "true_status": branch["status"],
                    "cnn_cutoff_probability": saved[f"cnn_seed{primary}_cutoff_probability"][local_index],
                    "cnn_predicted_status": cnn_predicted,
                    "ridge_cutoff_probability": ridge_probability[local_index],
                    "ridge_predicted_status": ridge_predicted,
                    "config_id": config_id,
                }
            )
            if branch["status"] != "complete":
                continue
            for method, prediction in [("CNN", primary_voltage), ("Ridge", ridge_voltage)]:
                for point, time_s in enumerate(time_grid):
                    true_voltage = truth["voltage_V"][global_index, point]
                    predicted = prediction[local_index, point]
                    interactive_points.append(
                        {
                            "branch_id": branch["branch_id"],
                            "time_s": time_s,
                            "method": method,
                            "pybamm_voltage_V": true_voltage,
                            "predicted_voltage_V": predicted,
                            "signed_error_mV": 1000.0 * (predicted - true_voltage),
                        }
                    )
    pd.DataFrame(interactive_cases).to_csv(RESULTS / "interactive_cases.csv", index=False)
    pd.DataFrame(interactive_points).to_csv(RESULTS / "interactive_predictions.csv", index=False)
    selection_path = RESULTS / "display_selection.json"
    selection_record = json.loads(selection_path.read_text())
    selection_record.update(
        {
            "cutoff_boundary_history_id": boundary_history,
            "interactive_history_ids": [history_id for _, history_id in selected_histories],
            "interactive_rule": (
                "All candidate currents for the typical, worst-error, and most-cutoff test histories; "
                "ties are resolved by history ID."
            ),
        }
    )
    write_json(selection_path, selection_record)

    cnn_rows = metric_frame[metric_frame["method"].eq("CNN")]
    primary_metrics = metric_frame[
        metric_frame["method"].eq("CNN") & metric_frame["seed"].astype(str).eq(str(primary))
    ].iloc[0].to_dict()
    rollout_frame = pd.DataFrame(rollout_rows)
    summary = {
        "experiment": config["experiment"],
        "config_id": config_id,
        "evaluated_utc": now_utc(),
        "test_histories": int(manifest.loc[test_index, "history_id"].nunique()),
        "test_complete_pulses": int(truth["complete"][test_index].sum()),
        "test_physical_cutoffs": int((~truth["complete"][test_index]).sum()),
        "primary_cnn": primary_metrics,
        "cnn_seed_voltage_mae_mV_mean": float(cnn_rows["voltage_mae_mV"].mean()),
        "cnn_seed_voltage_mae_mV_std": float(cnn_rows["voltage_mae_mV"].std(ddof=0)),
        "primary_cnn_rollout_final_step_voltage_mae_mV": float(
            rollout_frame[
                rollout_frame["method"].eq(f"CNN seed {primary}")
                & rollout_frame["step"].eq(len(config["rollout_currents_A"]))
            ]["voltage_mae_mV"].mean()
        ),
        "speed": speed_frame.to_dict("records"),
        "timing_scope": (
            "Proxy timings include input normalization, inference, and output de-normalization for one sample. "
            "PyBaMM timings are solver.step only for complete test pulses; model build and history generation are excluded."
        ),
        "test_data_used_for_selection": False,
    }
    write_json(RESULTS / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
