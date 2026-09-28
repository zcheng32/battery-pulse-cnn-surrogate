"""Train three CNN seeds and a same-input Ridge transition baseline."""

from __future__ import annotations

import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import LOCAL, RESULTS, load_config, now_utc, write_json
from model import make_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def load_data():
    data = np.load(LOCAL / "dataset.npz")
    manifest = pd.read_csv(RESULTS / "dataset_manifest.csv")
    if len(manifest) != len(data["current_A"]):
        raise RuntimeError("Dataset and manifest row counts differ.")
    return data, manifest


def fit_scalers(data, manifest):
    train = manifest["split"].eq("train").to_numpy()
    train_complete = train & data["complete"]
    state = data["state0"][train]
    current = data["current_A"][train]
    delta = data["state1"][train_complete] - data["state0"][train_complete]
    voltage = data["voltage_V"][train_complete]
    scalers = {
        "state_mean": state.mean(axis=(0, 2), keepdims=True),
        "state_scale": np.maximum(state.std(axis=(0, 2), keepdims=True), 1e-8),
        "current_mean": np.asarray([current.mean()]),
        "current_scale": np.asarray([max(current.std(), 1e-8)]),
        "delta_mean": delta.mean(axis=0, keepdims=True),
        "delta_scale": np.maximum(delta.std(axis=0, keepdims=True), 1e-8),
        "voltage_mean": voltage.mean(axis=0, keepdims=True),
        "voltage_scale": np.maximum(voltage.std(axis=0, keepdims=True), 1e-8),
    }
    np.savez(LOCAL / "preprocessing.npz", **scalers)
    return scalers


def standardized(data, scalers):
    state = (data["state0"] - scalers["state_mean"]) / scalers["state_scale"]
    current = (data["current_A"] - scalers["current_mean"][0]) / scalers["current_scale"][0]
    delta = (data["state1"] - data["state0"] - scalers["delta_mean"]) / scalers["delta_scale"]
    voltage = (data["voltage_V"] - scalers["voltage_mean"]) / scalers["voltage_scale"]
    cutoff = (~data["complete"]).astype(np.float32)
    return {
        "state": state.astype(np.float32),
        "current": current.astype(np.float32),
        "delta": delta.astype(np.float32),
        "voltage": voltage.astype(np.float32),
        "cutoff": cutoff,
    }


def choose_threshold(y_true, probability, candidates):
    y_true = np.asarray(y_true, dtype=bool)
    records = []
    for threshold in candidates:
        predicted = probability >= threshold
        sensitivity = float(predicted[y_true].mean()) if y_true.any() else 1.0
        specificity = float((~predicted[~y_true]).mean()) if (~y_true).any() else 1.0
        false_feasible = int((~predicted & y_true).sum())
        balanced = 0.5 * (sensitivity + specificity)
        records.append((balanced, -false_feasible, -abs(threshold - 0.5), float(threshold)))
    best = max(records)
    return best[3], {"balanced_accuracy": best[0], "false_feasible": -best[1]}


def predict_cnn(model, arrays, indices, scalers):
    model.eval()
    with torch.inference_mode():
        state = torch.from_numpy(arrays["state"][indices])
        current = torch.from_numpy(arrays["current"][indices])
        voltage_z, delta_z, cutoff_logit = model(state, current)
    voltage = voltage_z.numpy() * scalers["voltage_scale"] + scalers["voltage_mean"]
    delta = delta_z.numpy() * scalers["delta_scale"] + scalers["delta_mean"]
    probability = torch.sigmoid(cutoff_logit).numpy()
    return voltage, delta, probability


def train_one_seed(seed, data, manifest, arrays, scalers, config):
    set_seed(int(seed))
    train_index = np.flatnonzero(manifest["split"].eq("train").to_numpy())
    val_complete = np.flatnonzero(
        manifest["split"].eq("val").to_numpy() & data["complete"]
    )
    val_all = np.flatnonzero(manifest["split"].eq("val").to_numpy())
    dataset = TensorDataset(
        torch.from_numpy(arrays["state"][train_index]),
        torch.from_numpy(arrays["current"][train_index]),
        torch.from_numpy(arrays["voltage"][train_index]),
        torch.from_numpy(arrays["delta"][train_index]),
        torch.from_numpy(arrays["cutoff"][train_index]),
        torch.from_numpy(data["complete"][train_index]),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    model = make_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_cutoffs = arrays["cutoff"][train_index]
    positive = float(train_cutoffs.sum())
    negative = float(len(train_cutoffs) - positive)
    cutoff_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))
    best = {"mae_V": float("inf"), "epoch": 0, "state": None}
    history = []
    patience_checks = max(
        1, int(config["early_stop_patience_epochs"] / config["validation_every_epochs"])
    )
    checks_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        for state, current, voltage, delta, cutoff, complete in loader:
            optimizer.zero_grad(set_to_none=True)
            voltage_hat, delta_hat, cutoff_logit = model(state, current)
            regression = complete.bool()
            voltage_loss = nn.functional.mse_loss(voltage_hat[regression], voltage[regression])
            state_loss = nn.functional.mse_loss(delta_hat[regression], delta[regression])
            classification_loss = cutoff_loss(cutoff_logit, cutoff)
            loss = (
                voltage_loss
                + float(config["state_loss_weight"]) * state_loss
                + float(config["cutoff_loss_weight"]) * classification_loss
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(state)

        if epoch % int(config["validation_every_epochs"]) != 0:
            continue
        voltage_hat, _, _ = predict_cnn(model, arrays, val_complete, scalers)
        mae = float(np.mean(np.abs(voltage_hat - data["voltage_V"][val_complete])))
        history.append({"epoch": epoch, "train_loss": epoch_loss / len(dataset), "val_mae_V": mae})
        if mae < best["mae_V"] - 1e-8:
            best = {
                "mae_V": mae,
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        if checks_without_improvement >= patience_checks:
            break

    if best["state"] is None:
        raise RuntimeError("No CNN checkpoint was selected.")
    model.load_state_dict(best["state"])
    _, _, val_probability = predict_cnn(model, arrays, val_all, scalers)
    threshold, threshold_metrics = choose_threshold(
        ~data["complete"][val_all], val_probability, config["cutoff_threshold_candidates"]
    )
    checkpoint = {
        "seed": int(seed),
        "best_epoch": int(best["epoch"]),
        "validation_voltage_mae_V": float(best["mae_V"]),
        "cutoff_threshold": float(threshold),
        "model_state_dict": best["state"],
    }
    torch.save(checkpoint, LOCAL / f"cnn_seed{seed}.pt")
    pd.DataFrame(history).to_csv(LOCAL / f"training_history_seed{seed}.csv", index=False)
    result = {
        "method": "CNN",
        "seed": int(seed),
        "best_epoch": int(best["epoch"]),
        "validation_voltage_mae_mV": 1000.0 * float(best["mae_V"]),
        "cutoff_threshold": float(threshold),
        "validation_cutoff_balanced_accuracy": threshold_metrics["balanced_accuracy"],
        "validation_false_feasible": threshold_metrics["false_feasible"],
        "training_wall_s": time.perf_counter() - started,
    }
    print(json.dumps(result), flush=True)
    return result


def train_ridge(data, manifest, arrays, scalers, config):
    train = manifest["split"].eq("train").to_numpy()
    val = manifest["split"].eq("val").to_numpy()
    train_complete = train & data["complete"]
    val_complete = val & data["complete"]
    features = np.concatenate(
        [arrays["state"].reshape(len(train), -1), arrays["current"][:, None]], axis=1
    )
    targets = np.concatenate(
        [arrays["voltage"], arrays["delta"].reshape(len(train), -1)], axis=1
    )
    records = []
    best = None
    started = time.perf_counter()
    for alpha in config["ridge_alpha_candidates"]:
        model = Ridge(alpha=float(alpha))
        model.fit(features[train_complete], targets[train_complete])
        prediction = model.predict(features[val_complete])[:, : data["voltage_V"].shape[1]]
        voltage = prediction * scalers["voltage_scale"] + scalers["voltage_mean"]
        mae = float(np.mean(np.abs(voltage - data["voltage_V"][val_complete])))
        records.append({"alpha": float(alpha), "validation_voltage_mae_mV": 1000.0 * mae})
        candidate = (mae, float(alpha), model)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=0)
    classifier.fit(features[train], ~data["complete"][train])
    val_probability = classifier.predict_proba(features[val])[:, 1]
    threshold, threshold_metrics = choose_threshold(
        ~data["complete"][val], val_probability, config["cutoff_threshold_candidates"]
    )
    bundle = {
        "regressor": best[2],
        "classifier": classifier,
        "alpha": best[1],
        "cutoff_threshold": threshold,
    }
    joblib.dump(bundle, LOCAL / "ridge.joblib")
    pd.DataFrame(records).to_csv(LOCAL / "ridge_validation.csv", index=False)
    result = {
        "method": "Ridge",
        "seed": "deterministic",
        "best_epoch": "",
        "validation_voltage_mae_mV": 1000.0 * best[0],
        "cutoff_threshold": float(threshold),
        "validation_cutoff_balanced_accuracy": threshold_metrics["balanced_accuracy"],
        "validation_false_feasible": threshold_metrics["false_feasible"],
        "ridge_alpha": best[1],
        "training_wall_s": time.perf_counter() - started,
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    config, config_id = load_config()
    data, manifest = load_data()
    scalers = fit_scalers(data, manifest)
    arrays = standardized(data, scalers)
    summaries = [
        train_one_seed(seed, data, manifest, arrays, scalers, config)
        for seed in config["training_seeds"]
    ]
    summaries.append(train_ridge(data, manifest, arrays, scalers, config))
    pd.DataFrame(summaries).to_csv(RESULTS / "training_summary.csv", index=False)
    write_json(
        RESULTS / "training_metadata.json",
        {
            "experiment": config["experiment"],
            "config_id": config_id,
            "trained_utc": now_utc(),
            "selection": "Hyperparameters, epochs, and cutoff thresholds use validation data only.",
            "test_data_used_for_selection": False,
            "summaries": summaries,
        },
    )


if __name__ == "__main__":
    main()
