"""Evaluate recursive CNN rollouts with online voltage-innovation state correction."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from common import LOCAL, RESULTS, ROOT, now_utc, read_json, sha256, write_json
from correction_model import make_corrector
from evaluate import cnn_predict, load_scalers
from train_correction import feature_vector, load_base_model


CORRECTION_CONFIG = ROOT / "correction_config.json"


def reconstruct_correction(prediction_z, bundle):
    transforms = bundle["transforms"]
    coefficients = (
        prediction_z * transforms["coefficient_scale"] + transforms["coefficient_mean"]
    )
    return bundle["pca"].inverse_transform(coefficients).reshape(-1, 2, 96)


def linear_correction(bundle, raw_features):
    transforms = bundle["transforms"]
    features = (raw_features - transforms["feature_mean"]) / transforms["feature_scale"]
    return reconstruct_correction(bundle["model"].predict(features), bundle)


def nonlinear_correction(bundle, model, raw_features):
    transforms = bundle["transforms"]
    features = (raw_features - transforms["feature_mean"]) / transforms["feature_scale"]
    with torch.inference_mode():
        prediction = model(torch.from_numpy(features.astype(np.float32))).numpy()
    return reconstruct_correction(prediction, bundle)


def load_nonlinear_models(correction_config, feature_count):
    models = {}
    for seed in correction_config["corrector_seeds"]:
        checkpoint = torch.load(
            LOCAL / f"correction_mlp_seed{seed}.pt", map_location="cpu", weights_only=True
        )
        model = make_corrector(correction_config, feature_count)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[int(seed)] = model
    return models


def rollout_method(
    data,
    sequence_indices,
    base_model,
    threshold,
    scalers,
    correction_name,
    correction_object,
    bundle,
    measured_voltage,
    noise_mV,
):
    count = len(sequence_indices)
    steps = data["currents_A"].shape[1]
    points = data["voltage_V"].shape[2]
    state_estimate = data["states"][sequence_indices, 0].copy()
    voltage_prediction = np.full((count, steps, points), np.nan)
    prior_state_prediction = np.full((count, steps, 2, 96), np.nan)
    posterior_state = np.full((count, steps, 2, 96), np.nan)
    cutoff_probability = np.full((count, steps), np.nan)
    clipped_fraction = np.full((count, steps), np.nan)
    rows = []

    for step in range(steps):
        active_local = np.flatnonzero(data["attempted"][sequence_indices, step])
        if len(active_local) == 0:
            continue
        global_indices = sequence_indices[active_local]
        current = data["currents_A"][global_indices, step]
        voltage_hat, state_hat, probability = cnn_predict(
            base_model, state_estimate[active_local], current, scalers
        )
        voltage_prediction[active_local, step] = voltage_hat
        prior_state_prediction[active_local, step] = state_hat
        cutoff_probability[active_local, step] = probability
        complete = data["complete"][global_indices, step]
        predicted_cutoff = probability >= threshold
        for local_position, global_index, predicted in zip(
            active_local, global_indices, predicted_cutoff
        ):
            rows.append(
                {
                    "sequence_index": int(global_index),
                    "sequence_id": str(data["sequence_ids"][global_index]),
                    "history_id": str(data["history_ids"][global_index]),
                    "step": step + 1,
                    "current_A": float(data["currents_A"][global_index, step]),
                    "true_status": "complete"
                    if data["complete"][global_index, step]
                    else "physical_cutoff",
                    "predicted_status": "physical_cutoff" if predicted else "complete",
                    "cutoff_probability": float(
                        cutoff_probability[local_position, step]
                    ),
                }
            )
        if not complete.any():
            continue
        selected_local = active_local[complete]
        selected_global = global_indices[complete]
        selected_state = state_hat[complete]
        if correction_name == "Open-loop":
            corrected = selected_state
            clipping = np.zeros(len(selected_state))
        else:
            raw_features = feature_vector(
                selected_state,
                current[complete],
                measured_voltage[selected_global, step],
                voltage_hat[complete],
                scalers,
            )
            if correction_name == "Linear corrector":
                update = linear_correction(bundle, raw_features)
            else:
                update = nonlinear_correction(bundle, correction_object, raw_features)
            unprojected = selected_state + update
            corrected = np.clip(unprojected, 0.0, 1.0)
            clipping = np.mean(unprojected != corrected, axis=(1, 2))
        posterior_state[selected_local, step] = corrected
        clipped_fraction[selected_local, step] = clipping
        state_estimate[selected_local] = corrected

    row_frame = pd.DataFrame(rows)
    row_frame["method"] = correction_name
    row_frame["measurement_noise_mV"] = float(noise_mV)
    row_frame["false_feasible"] = row_frame["true_status"].eq("physical_cutoff") & row_frame[
        "predicted_status"
    ].eq("complete")
    row_frame["false_infeasible"] = row_frame["true_status"].eq("complete") & row_frame[
        "predicted_status"
    ].eq("physical_cutoff")
    return {
        "voltage_prediction": voltage_prediction,
        "prior_state_prediction": prior_state_prediction,
        "posterior_state": posterior_state,
        "cutoff_probability": cutoff_probability,
        "clipped_fraction": clipped_fraction,
        "classification_rows": row_frame,
    }


def metric_rows(method_label, noise_mV, result, data, sequence_indices):
    complete = data["complete"][sequence_indices]
    predicted_voltage = result["voltage_prediction"]
    true_voltage = data["voltage_V"][sequence_indices]
    true_state = data["states"][sequence_indices, 1:]
    prior = result["prior_state_prediction"]
    posterior = result["posterior_state"]
    voltage_error = predicted_voltage[complete] - true_voltage[complete]
    prior_error = prior[complete] - true_state[complete]
    posterior_error = posterior[complete] - true_state[complete]
    classes = result["classification_rows"]
    true_cutoff = classes["true_status"].eq("physical_cutoff")
    true_complete = ~true_cutoff
    tp = int((true_cutoff & classes["predicted_status"].eq("physical_cutoff")).sum())
    tn = int((true_complete & classes["predicted_status"].eq("complete")).sum())
    false_feasible = int(classes["false_feasible"].sum())
    false_infeasible = int(classes["false_infeasible"].sum())
    clipped = result["clipped_fraction"][complete]
    overall = {
        "method": method_label,
        "measurement_noise_mV": float(noise_mV),
        "complete_steps": int(complete.sum()),
        "cutoff_attempts": int(true_cutoff.sum()),
        "voltage_mae_mV": 1000.0 * float(np.mean(np.abs(voltage_error))),
        "voltage_max_abs_mV": 1000.0 * float(np.max(np.abs(voltage_error))),
        "voltage_endpoint_mae_mV": 1000.0 * float(np.mean(np.abs(voltage_error[:, -1]))),
        "prior_state_rmse_fraction": float(np.sqrt(np.mean(prior_error**2))),
        "posterior_state_rmse_fraction": float(np.sqrt(np.mean(posterior_error**2))),
        "cutoff_true_positive": tp,
        "cutoff_true_negative": tn,
        "false_feasible": false_feasible,
        "false_infeasible": false_infeasible,
        "state_bound_clipped_fraction": float(np.nanmean(clipped)),
    }
    by_step = []
    for step in range(complete.shape[1]):
        mask = complete[:, step]
        class_step = classes[classes["step"].eq(step + 1)]
        record = {
            "method": method_label,
            "measurement_noise_mV": float(noise_mV),
            "step": step + 1,
            "elapsed_s": float((step + 1) * 10.0),
            "complete_steps": int(mask.sum()),
            "cutoff_attempts": int(class_step["true_status"].eq("physical_cutoff").sum()),
            "false_feasible": int(class_step["false_feasible"].sum()),
            "false_infeasible": int(class_step["false_infeasible"].sum()),
        }
        if mask.any():
            v_error = predicted_voltage[mask, step] - true_voltage[mask, step]
            prior_error_step = prior[mask, step] - true_state[mask, step]
            posterior_error_step = posterior[mask, step] - true_state[mask, step]
            record.update(
                {
                    "voltage_mae_mV": 1000.0 * float(np.mean(np.abs(v_error))),
                    "voltage_max_abs_mV": 1000.0 * float(np.max(np.abs(v_error))),
                    "prior_state_rmse_fraction": float(np.sqrt(np.mean(prior_error_step**2))),
                    "posterior_state_rmse_fraction": float(
                        np.sqrt(np.mean(posterior_error_step**2))
                    ),
                }
            )
        by_step.append(record)
    return overall, by_step


def sequence_metric_rows(method_label, noise_mV, result, data, sequence_indices):
    rows = []
    for local_index, global_index in enumerate(sequence_indices):
        complete = data["complete"][global_index]
        if complete.any():
            error = (
                result["voltage_prediction"][local_index, complete]
                - data["voltage_V"][global_index, complete]
            )
            prior_error = (
                result["prior_state_prediction"][local_index, complete]
                - data["states"][global_index, 1:][complete]
            )
            posterior_error = (
                result["posterior_state"][local_index, complete]
                - data["states"][global_index, 1:][complete]
            )
            voltage_mae = 1000.0 * float(np.mean(np.abs(error)))
            prior_rmse = float(np.sqrt(np.mean(prior_error**2)))
            posterior_rmse = float(np.sqrt(np.mean(posterior_error**2)))
        else:
            voltage_mae = np.nan
            prior_rmse = np.nan
            posterior_rmse = np.nan
        classes = result["classification_rows"]
        classes = classes[classes["sequence_index"].eq(global_index)]
        rows.append(
            {
                "sequence_id": str(data["sequence_ids"][global_index]),
                "history_id": str(data["history_ids"][global_index]),
                "method": method_label,
                "measurement_noise_mV": float(noise_mV),
                "complete_steps": int(complete.sum()),
                "voltage_mae_mV": voltage_mae,
                "prior_state_rmse_fraction": prior_rmse,
                "posterior_state_rmse_fraction": posterior_rmse,
                "false_feasible": int(classes["false_feasible"].sum()),
                "false_infeasible": int(classes["false_infeasible"].sum()),
            }
        )
    return rows


def benchmark(name, function, repeats=2000, warmup=100):
    for _ in range(warmup):
        function()
    values = np.empty(repeats)
    for index in range(repeats):
        started = time.perf_counter_ns()
        function()
        values[index] = (time.perf_counter_ns() - started) * 1e-6
    return {
        "method": name,
        "median_latency_ms": float(np.median(values)),
        "p95_latency_ms": float(np.quantile(values, 0.95)),
        "repeats": repeats,
    }


def main():
    torch.set_num_threads(1)
    base_config = read_json(ROOT / "config.json")
    correction_config = read_json(CORRECTION_CONFIG)
    correction_id = sha256(CORRECTION_CONFIG)
    source = np.load(LOCAL / "correction_sequences.npz")
    data = {key: source[key] for key in source.files}
    scalers = load_scalers()
    base_model, threshold = load_base_model(base_config, correction_config)
    bundle = joblib.load(LOCAL / "correction_ridge.joblib")
    feature_count = int(bundle["transforms"]["feature_mean"].shape[1])
    nonlinear_models = load_nonlinear_models(correction_config, feature_count)
    test_indices = np.flatnonzero(data["splits"] == "test")
    noise_rng = np.random.default_rng(int(correction_config["measurement_noise_seed"]))
    overall_rows = []
    step_rows = []
    sequence_rows = []
    classification_rows = []
    saved = {
        "sequence_indices": test_indices,
        "sequence_ids": data["sequence_ids"][test_indices],
        "true_voltage_V": data["voltage_V"][test_indices],
        "true_states": data["states"][test_indices],
        "currents_A": data["currents_A"][test_indices],
        "attempted": data["attempted"][test_indices],
        "complete": data["complete"][test_indices],
    }
    methods = [("Open-loop", None), ("Linear corrector", bundle)] + [
        (f"Nonlinear corrector seed {seed}", model)
        for seed, model in nonlinear_models.items()
    ]
    for noise_mV in correction_config["test_measurement_noise_mV"]:
        measurement = data["voltage_V"].copy()
        finite = np.isfinite(measurement)
        noise = noise_rng.normal(0.0, float(noise_mV) / 1000.0, size=measurement.shape)
        measurement[finite] += noise[finite]
        for method_label, correction_object in methods:
            correction_name = (
                "Nonlinear corrector"
                if method_label.startswith("Nonlinear")
                else method_label
            )
            result = rollout_method(
                data,
                test_indices,
                base_model,
                threshold,
                scalers,
                correction_name,
                correction_object,
                bundle,
                measurement,
                noise_mV,
            )
            overall, per_step = metric_rows(
                method_label, noise_mV, result, data, test_indices
            )
            overall_rows.append(overall)
            step_rows.extend(per_step)
            sequence_rows.extend(
                sequence_metric_rows(
                    method_label, noise_mV, result, data, test_indices
                )
            )
            classes = result["classification_rows"].copy()
            classes["method"] = method_label
            classification_rows.extend(classes.to_dict("records"))
            key = method_label.lower().replace(" ", "_") + f"_noise_{float(noise_mV):g}"
            saved[f"{key}_voltage_V"] = result["voltage_prediction"]
            saved[f"{key}_posterior_states"] = result["posterior_state"]
            saved[f"{key}_cutoff_probability"] = result["cutoff_probability"]

    overall_frame = pd.DataFrame(overall_rows)
    step_frame = pd.DataFrame(step_rows)
    sequence_frame = pd.DataFrame(sequence_rows)
    classification_frame = pd.DataFrame(classification_rows)
    overall_frame.to_csv(RESULTS / "correction_test_metrics.csv", index=False)
    step_frame.to_csv(RESULTS / "correction_step_metrics.csv", index=False)
    sequence_frame.to_csv(RESULTS / "correction_sequence_metrics.csv", index=False)
    classification_frame.to_csv(RESULTS / "correction_classification_metrics.csv", index=False)
    np.savez_compressed(LOCAL / "correction_test_predictions.npz", **saved)

    validation_summary = pd.read_csv(RESULTS / "correction_training_summary.csv")
    primary_row = validation_summary.sort_values(
        ["validation_posterior_state_rmse", "method"]
    ).iloc[0]
    primary_method = str(primary_row["method"])
    clean_sequence = sequence_frame[
        sequence_frame["measurement_noise_mV"].eq(0.0)
        & sequence_frame["method"].eq(primary_method)
        & sequence_frame["complete_steps"].ge(8)
    ]
    clean_open = sequence_frame[
        sequence_frame["measurement_noise_mV"].eq(0.0)
        & sequence_frame["method"].eq("Open-loop")
    ][["sequence_id", "voltage_mae_mV"]].rename(
        columns={"voltage_mae_mV": "open_loop_voltage_mae_mV"}
    )
    selection = clean_sequence.merge(clean_open, on="sequence_id")
    median = float(selection["open_loop_voltage_mae_mV"].median())
    selected = selection.iloc[
        (selection["open_loop_voltage_mae_mV"] - median).abs().argmin()
    ]
    selected_sequence = str(selected["sequence_id"])
    selected_global = int(
        np.flatnonzero(data["sequence_ids"] == selected_sequence)[0]
    )
    selected_local = int(np.flatnonzero(test_indices == selected_global)[0])
    display_rows = []
    time_grid = np.arange(data["voltage_V"].shape[2]) * float(
        base_config["sample_interval_s"]
    )
    display_methods = ["Open-loop", primary_method, f"Nonlinear corrector seed {correction_config['primary_corrector_seed']}"]
    for method_label in dict.fromkeys(display_methods):
        key = method_label.lower().replace(" ", "_") + "_noise_0"
        prediction = saved[f"{key}_voltage_V"]
        for step in np.flatnonzero(data["complete"][selected_global]):
            for point, within_pulse in enumerate(time_grid):
                true_voltage = data["voltage_V"][selected_global, step, point]
                predicted_voltage = prediction[selected_local, step, point]
                display_rows.append(
                    {
                        "sequence_id": selected_sequence,
                        "history_id": str(data["history_ids"][selected_global]),
                        "method": method_label,
                        "step": int(step + 1),
                        "current_A": float(data["currents_A"][selected_global, step]),
                        "elapsed_s": float(step * 10.0 + within_pulse),
                        "pybamm_voltage_V": float(true_voltage),
                        "predicted_voltage_V": float(predicted_voltage),
                        "signed_error_mV": 1000.0 * float(predicted_voltage - true_voltage),
                        "measurement_noise_mV": 0.0,
                        "correction_config_id": correction_id,
                    }
                )
    pd.DataFrame(display_rows).to_csv(
        RESULTS / "correction_display_predictions.csv", index=False
    )
    write_json(
        RESULTS / "correction_display_selection.json",
        {
            "rule": correction_config["display_rule"],
            "primary_method_selected_by_validation": primary_method,
            "sequence_id": selected_sequence,
            "complete_steps": int(selected["complete_steps"]),
            "open_loop_voltage_mae_mV": float(selected["open_loop_voltage_mae_mV"]),
        },
    )

    example_index = int(test_indices[np.flatnonzero(data["complete"][test_indices, 0])[0]])
    example_current = data["currents_A"][example_index : example_index + 1, 0]
    example_true = data["voltage_V"][example_index : example_index + 1, 0]
    example_pred, example_state, _ = cnn_predict(
        base_model, data["states"][example_index : example_index + 1, 0], example_current, scalers
    )
    raw_feature = feature_vector(
        example_state, example_current, example_true, example_pred, scalers
    )
    primary_nonlinear = nonlinear_models[int(correction_config["primary_corrector_seed"])]
    speed_rows = [
        benchmark("Linear correction only", lambda: linear_correction(bundle, raw_feature)),
        benchmark(
            f"Nonlinear correction only seed {correction_config['primary_corrector_seed']}",
            lambda: nonlinear_correction(bundle, primary_nonlinear, raw_feature),
        ),
    ]
    pd.DataFrame(speed_rows).to_csv(RESULTS / "correction_speed_metrics.csv", index=False)

    clean = overall_frame[overall_frame["measurement_noise_mV"].eq(0.0)].set_index("method")
    noisy = overall_frame[overall_frame["measurement_noise_mV"].eq(5.0)].set_index("method")
    summary = {
        "experiment": correction_config["experiment"],
        "correction_config_id": correction_id,
        "evaluated_utc": now_utc(),
        "test_sequences": int(len(test_indices)),
        "test_complete_steps": int(data["complete"][test_indices].sum()),
        "test_physical_cutoffs": int(
            (data["attempted"][test_indices] & ~data["complete"][test_indices]).sum()
        ),
        "primary_method_selected_by_validation": primary_method,
        "clean_open_loop_voltage_mae_mV": float(clean.loc["Open-loop", "voltage_mae_mV"]),
        "clean_corrected_voltage_mae_mV": float(clean.loc[primary_method, "voltage_mae_mV"]),
        "clean_voltage_mae_reduction_percent": 100.0
        * (
            1.0
            - float(clean.loc[primary_method, "voltage_mae_mV"])
            / float(clean.loc["Open-loop", "voltage_mae_mV"])
        ),
        "clean_open_loop_posterior_state_rmse": float(
            clean.loc["Open-loop", "posterior_state_rmse_fraction"]
        ),
        "clean_corrected_posterior_state_rmse": float(
            clean.loc[primary_method, "posterior_state_rmse_fraction"]
        ),
        "noise_5mV_corrected_voltage_mae_mV": float(
            noisy.loc[primary_method, "voltage_mae_mV"]
        ),
        "clean_open_loop_false_feasible": int(clean.loc["Open-loop", "false_feasible"]),
        "clean_corrected_false_feasible": int(clean.loc[primary_method, "false_feasible"]),
        "scope": correction_config["scope_note"],
        "test_data_used_for_selection": False,
        "correction_latency": speed_rows,
    }
    write_json(RESULTS / "correction_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
