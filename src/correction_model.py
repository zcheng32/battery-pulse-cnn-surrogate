"""Low-rank nonlinear corrector driven by the measured voltage innovation."""

from __future__ import annotations

from torch import nn


class LatentVoltageCorrector(nn.Module):
    """Predict PCA coefficients of a state correction from an innovation feature vector."""

    def __init__(self, feature_count: int, component_count: int, hidden_widths):
        super().__init__()
        layers = []
        previous = int(feature_count)
        for width in hidden_widths:
            layers.extend([nn.Linear(previous, int(width)), nn.GELU()])
            previous = int(width)
        layers.append(nn.Linear(previous, int(component_count)))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


def make_corrector(correction_config: dict, feature_count: int):
    return LatentVoltageCorrector(
        feature_count=feature_count,
        component_count=correction_config["pca_components"],
        hidden_widths=correction_config["hidden_widths"],
    )
