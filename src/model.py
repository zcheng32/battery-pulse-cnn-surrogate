"""Compact convolutional transition surrogate for an SPM pulse."""

from __future__ import annotations

import torch
from torch import nn


class PulseTransitionCNN(nn.Module):
    """Map two radial concentration profiles and current to one pulse transition."""

    def __init__(self, nodes: int, time_points: int, channels, latent_width: int):
        super().__init__()
        c1, c2, c3 = [int(value) for value in channels]
        self.nodes = int(nodes)
        self.time_points = int(time_points)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, c1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(c2, c3, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(12),
        )
        self.trunk = nn.Sequential(
            nn.Linear(c3 * 12 + 1, latent_width),
            nn.GELU(),
            nn.Linear(latent_width, latent_width),
            nn.GELU(),
        )
        self.voltage_head = nn.Linear(latent_width, time_points)
        self.state_head = nn.Linear(latent_width, 2 * nodes)
        self.cutoff_head = nn.Linear(latent_width, 1)

    def forward(self, state, current):
        encoded = self.encoder(state).flatten(start_dim=1)
        hidden = self.trunk(torch.cat([encoded, current.reshape(-1, 1)], dim=1))
        voltage = self.voltage_head(hidden)
        delta_state = self.state_head(hidden).reshape(-1, 2, self.nodes)
        cutoff_logit = self.cutoff_head(hidden).reshape(-1)
        return voltage, delta_state, cutoff_logit


def make_model(config: dict) -> PulseTransitionCNN:
    points = int(round(config["pulse_duration_s"] / config["sample_interval_s"])) + 1
    return PulseTransitionCNN(
        nodes=config["radial_nodes_per_particle"],
        time_points=points,
        channels=config["conv_channels"],
        latent_width=config["latent_width"],
    )
