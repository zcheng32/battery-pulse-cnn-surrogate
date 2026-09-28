"""Train low-rank linear and nonlinear voltage-innovation state correctors."""

from __future__ import annotations

import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

from common import LOCAL, RESULTS, ROOT, now_utc, read_json, sha256, write_json
from correction_model import make_corrector
from evaluate import cnn_predict, load_scalers
from model import make_model


CORRECTION_CONFIG = ROOT / "correction_config.json"


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def load_base_model(base_config, correction_config):
    seed = int(correction_config["base_cnn_seed"])
    checkpoint = torch.load(LOCAL / f"cnn_seed{seed}.pt", map_location="cpu", weights_only=True)
    model = make_model(base_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(checkpoint["cutoff_threshold"])


def feature_vector(predicted_state, current, measured_voltage, predicted_voltage, scalers):
    residual_mV = 1000.0 * (measured_voltage - predicted_voltage)
    state_z = (predicted_state - scalers["state_mean"]) / scalers["state_scale"]
    current_z = (current - scalers["current_mean"][0]) / scalers["current_scale"][0]
    return np.concatenate(
        [residual_mV, state_z.reshape(len(predicted_state), -1), current_z.reshape(-1, 1)],
        axis=1,
    )


def collect_open_loop_examples(data, split, model, scalers):
    sequence_indices = np.flatnonzero(data["splits"] == split)
    state_estimate = data["states"][sequence_indices, 0].copy()
    features = []
    corrections = []
    sequence_rows = []
    step_rows = []
    for step in range(data["currents_A"].shape[1]):
        active_local = np.flatnonzero(data["attempted"][sequence_indices, step])
        if len(active_local) == 0:
            continue
        global_indices = sequence_indices[active_local]
        current = data["currents_A"][global_indices, step]
        voltage_hat, state_hat, _ = cnn_predict(
            model, state_estimate[active_local], current, scalers
        )
        complete = data["complete"][global_indices, step]
        if complete.any():
            selected_local = active_local[complete]
            selected_global = global_indices[complete]
            selected_state = state_hat[complete]
            features.append(
                feature_vector(
                    selected_state,
                    current[complete],
                    data["voltage_V"][selected_global, step],
                    voltage_hat[complete],
                    scalers,
                )
            )
            corrections.append(
                (
                    data["states"][selected_global, step + 1]
                    - selected_state
                ).reshape(len(selected_global), -1)
            )
            sequence_rows.extend(selected_global.tolist())
            step_rows.extend([step] * len(selected_global))
            state_estimate[selected_local] = selected_state
    return {
        "features": np.concatenate(features),
        "corrections": np.concatenate(corrections),
        "sequence_index": np.asarray(sequence_rows),
        "step_index": np.asarray(step_rows),
    }


def fit_preprocessing(train_examples, correction_config):
    pca = PCA(n_components=int(correction_config["pca_components"]), random_state=0)
    coefficients = pca.fit_transform(train_examples["corrections"])
    feature_mean = train_examples["features"].mean(axis=0, keepdims=True)
    feature_scale = np.maximum(train_examples["features"].std(axis=0, keepdims=True), 1e-6)
    coefficient_mean = coefficients.mean(axis=0, keepdims=True)
    coefficient_scale = np.maximum(coefficients.std(axis=0, keepdims=True), 1e-8)
    return pca, {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "coefficient_mean": coefficient_mean,
        "coefficient_scale": coefficient_scale,
    }


def standardized(examples, pca, transforms):
    features = (examples["features"] - transforms["feature_mean"]) / transforms["feature_scale"]
    coefficients = pca.transform(examples["corrections"])
    coefficients = (
        coefficients - transforms["coefficient_mean"]
    ) / transforms["coefficient_scale"]
    return features.astype(np.float32), coefficients.astype(np.float32)


def correction_rmse(prediction_z, corrections, pca, transforms):
    coefficients = (
        prediction_z * transforms["coefficient_scale"] + transforms["coefficient_mean"]
    )
    prediction = pca.inverse_transform(coefficients)
    return float(np.sqrt(np.mean((corrections - prediction) ** 2)))


def fit_ridge(train_examples, val_examples, pca, transforms, correction_config):
    train_x, train_y = standardized(train_examples, pca, transforms)
    val_x, _ = standardized(val_examples, pca, transforms)
    # Noise is applied in standardized feature units using the raw-feature scale.
    rng = np.random.default_rng(int(correction_config["measurement_noise_seed"]))
    noise_mV = float(correction_config["training_measurement_noise_mV"])
    noisy_copies = [train_x]
    for _ in range(2):
        noisy = train_x.copy()
        noisy[:, :41] += rng.normal(0.0, noise_mV, size=(len(noisy), 41)) / transforms[
            "feature_scale"
        ][0, :41]
        noisy_copies.append(noisy)
    fit_x = np.concatenate(noisy_copies)
    fit_y = np.concatenate([train_y] * len(noisy_copies))
    val_noisy = val_x.copy()
    val_noisy[:, :41] += rng.normal(0.0, noise_mV, size=(len(val_noisy), 41)) / transforms[
        "feature_scale"
    ][0, :41]
    records = []
    best = None
    for alpha in correction_config["ridge_alpha_candidates"]:
        model = Ridge(alpha=float(alpha))
        model.fit(fit_x, fit_y)
        prediction = model.predict(val_noisy)
        rmse = correction_rmse(
            prediction, val_examples["corrections"], pca, transforms
        )
        records.append({"alpha": float(alpha), "validation_posterior_state_rmse": rmse})
        candidate = (rmse, float(alpha), model)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return {
        "model": best[2],
        "alpha": best[1],
        "validation_posterior_state_rmse": best[0],
        "validation_records": records,
    }


def train_mlp(seed, train_examples, val_examples, pca, transforms, correction_config):
    set_seed(seed)
    train_x, train_y = standardized(train_examples, pca, transforms)
    val_x, _ = standardized(val_examples, pca, transforms)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(correction_config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    model = make_corrector(correction_config, train_x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(correction_config["learning_rate"]),
        weight_decay=float(correction_config["weight_decay"]),
    )
    noise_mV = float(correction_config["training_measurement_noise_mV"])
    noise_scale = torch.from_numpy(
        (noise_mV / transforms["feature_scale"][0, :41]).astype(np.float32)
    )
    val_rng = np.random.default_rng(int(correction_config["measurement_noise_seed"]) + int(seed))
    val_noisy = val_x.copy()
    val_noisy[:, :41] += val_rng.normal(0.0, noise_mV, size=(len(val_x), 41)) / transforms[
        "feature_scale"
    ][0, :41]
    val_tensor = torch.from_numpy(val_noisy.astype(np.float32))
    best = {"rmse": float("inf"), "epoch": 0, "state": None}
    history = []
    patience_checks = max(
        1,
        int(
            correction_config["early_stop_patience_epochs"]
            / correction_config["validation_every_epochs"]
        ),
    )
    without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, int(correction_config["max_epochs"]) + 1):
        model.train()
        total = 0.0
        for features, targets in loader:
            noisy_features = features.clone()
            noisy_features[:, :41] += torch.randn_like(noisy_features[:, :41]) * noise_scale
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy_features)
            loss = torch.nn.functional.mse_loss(prediction, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(features)
        if epoch % int(correction_config["validation_every_epochs"]) != 0:
            continue
        model.eval()
        with torch.inference_mode():
            prediction = model(val_tensor).numpy()
        rmse = correction_rmse(
            prediction, val_examples["corrections"], pca, transforms
        )
        history.append({"epoch": epoch, "train_loss": total / len(dataset), "val_state_rmse": rmse})
        if rmse < best["rmse"] - 1e-9:
            best = {
                "rmse": rmse,
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= patience_checks:
            break
    model.load_state_dict(best["state"])
    torch.save(
        {
            "seed": int(seed),
            "best_epoch": int(best["epoch"]),
            "validation_posterior_state_rmse": float(best["rmse"]),
            "model_state_dict": best["state"],
        },
        LOCAL / f"correction_mlp_seed{seed}.pt",
    )
    pd.DataFrame(history).to_csv(LOCAL / f"correction_training_history_seed{seed}.csv", index=False)
    return {
        "method": "Nonlinear corrector",
        "seed": int(seed),
        "best_epoch": int(best["epoch"]),
        "validation_posterior_state_rmse": float(best["rmse"]),
        "training_wall_s": time.perf_counter() - started,
    }


def main():
    torch.set_num_threads(1)
    base_config = read_json(ROOT / "config.json")
    correction_config = read_json(CORRECTION_CONFIG)
    correction_id = sha256(CORRECTION_CONFIG)
    source = np.load(LOCAL / "correction_sequences.npz")
    data = {key: source[key] for key in source.files}
    scalers = load_scalers()
    model, _ = load_base_model(base_config, correction_config)
    started = time.perf_counter()
    examples = {
        split: collect_open_loop_examples(data, split, model, scalers)
        for split in ["train", "val", "test"]
    }
    pca, transforms = fit_preprocessing(examples["train"], correction_config)
    np.savez(
        LOCAL / "correction_preprocessing.npz",
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        **transforms,
    )
    np.savez_compressed(
        LOCAL / "correction_training_examples.npz",
        **{
            f"{split}_{name}": values
            for split, record in examples.items()
            for name, values in record.items()
        },
    )
    ridge = fit_ridge(
        examples["train"], examples["val"], pca, transforms, correction_config
    )
    bundle = {
        "model": ridge["model"],
        "alpha": ridge["alpha"],
        "pca": pca,
        "transforms": transforms,
    }
    joblib.dump(bundle, LOCAL / "correction_ridge.joblib")
    rows = [
        {
            "method": "Linear corrector",
            "seed": "deterministic",
            "best_epoch": "",
            "validation_posterior_state_rmse": ridge["validation_posterior_state_rmse"],
            "ridge_alpha": ridge["alpha"],
            "training_wall_s": 0.0,
        }
    ]
    rows.extend(
        train_mlp(
            seed,
            examples["train"],
            examples["val"],
            pca,
            transforms,
            correction_config,
        )
        for seed in correction_config["corrector_seeds"]
    )
    pd.DataFrame(rows).to_csv(RESULTS / "correction_training_summary.csv", index=False)
    pd.DataFrame(ridge["validation_records"]).to_csv(
        LOCAL / "correction_ridge_validation.csv", index=False
    )
    metadata = {
        "experiment": correction_config["experiment"],
        "correction_config_id": correction_id,
        "trained_utc": now_utc(),
        "base_cnn_seed": correction_config["base_cnn_seed"],
        "training_examples": len(examples["train"]["features"]),
        "validation_examples": len(examples["val"]["features"]),
        "test_examples_reserved": len(examples["test"]["features"]),
        "pca_components": int(correction_config["pca_components"]),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "selection": "PCA, scaling, Ridge alpha, and MLP checkpoints use train/validation sequences only.",
        "test_data_used_for_selection": False,
        "training_wall_s": time.perf_counter() - started,
        "models": rows,
    }
    write_json(RESULTS / "correction_training_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
