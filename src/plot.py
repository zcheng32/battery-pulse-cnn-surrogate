"""Rebuild the static overview and interactive browser from tracked result tables."""

from __future__ import annotations

import json
import os

from common import LOCAL, RESULTS, ROOT

os.environ.setdefault("MPLCONFIGDIR", str(LOCAL / "matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from plotly.subplots import make_subplots

FIGURES = ROOT / "figures"
BLUE = "#2463A7"
ORANGE = "#E17C35"
GREEN = "#2A9D78"
RED = "#C94C4C"
INK = "#24313D"
GRID = "#DCE3E8"


def box(ax, x, y, width, height, text, color, fontsize=10):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color, edgecolor="none", alpha=0.14,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            color=INK, fontsize=fontsize, weight="bold")


def static_overview():
    metrics = pd.read_csv(RESULTS / "test_metrics.csv")
    display = pd.read_csv(RESULTS / "display_predictions.csv")
    rollout = pd.read_csv(RESULTS / "rollout_metrics.csv")
    speed = pd.read_csv(RESULTS / "speed_metrics.csv")
    selection = json.loads((RESULTS / "display_selection.json").read_text())
    primary = int(selection["primary_seed"])
    primary_metrics = metrics[(metrics["method"] == "CNN") & (metrics["seed"].astype(str) == str(primary))].iloc[0]
    typical = display[display["case_role"] == "typical"]
    cnn = typical[typical["method"] == "CNN"]
    ridge = typical[typical["method"] == "Ridge"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#AAB6BF",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )
    fig = plt.figure(figsize=(14, 9), facecolor="#F7F9FA")
    grid = fig.add_gridspec(2, 2, left=0.055, right=0.98, top=0.88, bottom=0.08,
                           hspace=0.34, wspace=0.24)
    fig.suptitle("Learning a 10-second battery pulse transition", x=0.055, y=0.965,
                 ha="left", fontsize=22, weight="bold", color=INK)
    fig.text(
        0.055, 0.918,
        "A compact CNN replaces one state-continuous PyBaMM SPM solve: voltage, next state, and cutoff risk.",
        ha="left", fontsize=11.5, color="#52616B",
    )

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A  One learned transition", loc="left", weight="bold")
    box(ax, 0.02, 0.60, 0.25, 0.22, "Negative particle\nprofile  (96 points)", BLUE)
    box(ax, 0.02, 0.29, 0.25, 0.22, "Positive particle\nprofile  (96 points)", GREEN)
    box(ax, 0.02, 0.05, 0.25, 0.13, "Pulse current  I", ORANGE)
    box(ax, 0.38, 0.31, 0.22, 0.30, "1D CNN\ntransition\nsurrogate", BLUE, 12)
    box(ax, 0.72, 0.64, 0.25, 0.18, "Voltage curve\n41 time points", ORANGE)
    box(ax, 0.72, 0.37, 0.25, 0.18, "Profiles after 10 s\n2 × 96 points", GREEN)
    box(ax, 0.72, 0.10, 0.25, 0.18, "Cutoff probability\nwithin 10 s", RED)
    for y0, y1 in [(0.71, 0.51), (0.40, 0.46), (0.115, 0.41)]:
        ax.add_patch(FancyArrowPatch((0.28, y0), (0.38, y1), arrowstyle="-|>",
                                     mutation_scale=12, color="#7E8D98", lw=1.5))
    for y in [0.73, 0.46, 0.19]:
        ax.add_patch(FancyArrowPatch((0.60, 0.46), (0.72, y), arrowstyle="-|>",
                                     mutation_scale=12, color="#7E8D98", lw=1.5))
    ax.text(0.02, 0.93, "Input: simulated internal state + candidate current",
            fontsize=10, color="#52616B")

    ax = fig.add_subplot(grid[0, 1])
    branch = str(cnn["branch_id"].iloc[0])
    current = float(cnn["current_A"].iloc[0])
    soc = float(cnn["initial_soc"].iloc[0])
    ax.plot(cnn["time_s"], cnn["pybamm_voltage_V"], color=INK, lw=2.5, label="PyBaMM")
    ax.plot(cnn["time_s"], cnn["predicted_voltage_V"], color=BLUE, lw=2, label="CNN")
    ax.plot(ridge["time_s"], ridge["predicted_voltage_V"], color=ORANGE, lw=1.8,
            ls="--", label="Ridge")
    ax.set_title(f"B  Typical unseen pulse  ·  {branch}  ·  SOC {soc:.2f}  ·  {current:g} A",
                 loc="left", weight="bold")
    ax.set_xlabel("Time in pulse (s)")
    ax.set_ylabel("Terminal voltage (V)")
    ax.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    inset = ax.inset_axes([0.12, 0.10, 0.82, 0.26])
    inset.axhline(0, color="#9AA6AE", lw=0.8)
    inset.plot(cnn["time_s"], cnn["signed_error_mV"], color=BLUE, lw=1.4)
    inset.plot(ridge["time_s"], ridge["signed_error_mV"], color=ORANGE, lw=1.2, ls="--")
    inset.set_ylabel("error\n(mV)", fontsize=8, rotation=0, labelpad=18)
    inset.tick_params(labelsize=7)
    inset.grid(True, color=GRID, lw=0.5)

    ax = fig.add_subplot(grid[1, 0])
    cnn_metrics = metrics[metrics["method"] == "CNN"].copy()
    cnn_metrics["seed_number"] = pd.to_numeric(cnn_metrics["seed"])
    cnn_metrics = cnn_metrics.sort_values("seed_number")
    ridge_metric = metrics[metrics["method"] == "Ridge"].iloc[0]
    labels = [f"CNN\nseed {int(seed)}" for seed in cnn_metrics["seed"]] + ["Ridge"]
    values = cnn_metrics["voltage_mae_mV"].tolist() + [ridge_metric["voltage_mae_mV"]]
    bars = ax.bar(labels, values, color=[BLUE, "#4A83BE", "#72A0CE", ORANGE], width=0.68)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.1f}",
                ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_ylim(0, max(values) * 1.20)
    ax.set_ylabel("Test voltage MAE (mV)")
    ax.set_title("C  Unseen-history test accuracy", loc="left", weight="bold")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.02, 0.95,
        f"Primary CNN: {int(primary_metrics['cutoff_true_positive'])}/"
        f"{int(primary_metrics['true_cutoff_count'])} cutoffs detected · "
        f"{int(primary_metrics['false_feasible'])} false-feasible",
        transform=ax.transAxes, va="top", fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID},
    )

    ax = fig.add_subplot(grid[1, 1])
    subset = rollout[rollout["method"].isin([f"CNN seed {primary}", "Ridge"])]
    average = subset.groupby(["method", "step"], as_index=False)["voltage_mae_mV"].mean()
    for method, color, marker in [(f"CNN seed {primary}", BLUE, "o"), ("Ridge", ORANGE, "s")]:
        values = average[average["method"] == method]
        ax.plot(values["step"] * 10, values["voltage_mae_mV"], color=color, marker=marker,
                lw=2, label=method)
    ax.set_xlabel("Recursive rollout time (s)")
    ax.set_ylabel("Mean voltage MAE (mV)")
    ax.set_title("D  Reusing the predicted state", loc="left", weight="bold")
    ax.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    cnn_speed = speed[speed["method"] == f"CNN seed {primary}"].iloc[0]
    ax.text(
        0.98, 0.08,
        f"CNN latency  {cnn_speed['median_latency_ms']:.3f} ms\n"
        f"{cnn_speed['speedup_vs_pybamm_median']:.1f}× vs PyBaMM median\n"
        f"step 6 MAE  {average[(average['method'] == f'CNN seed {primary}') & (average['step'] == 6)]['voltage_mae_mV'].iloc[0]:.1f} mV",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID},
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.055, 0.025,
             "Synthetic Chen2020 SPM data · 48/12/12 history split · primary model fixed before test · no real-cell validation",
             fontsize=9.5, color="#61717C")
    fig.savefig(FIGURES / "overview.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def interactive_browser():
    cases = pd.read_csv(RESULTS / "interactive_cases.csv")
    points = pd.read_csv(RESULTS / "interactive_predictions.csv")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        row_heights=[0.55, 0.27, 0.18], vertical_spacing=0.10,
        subplot_titles=("Terminal voltage", "Signed prediction error", "Cutoff classification"),
    )
    buttons = []
    traces_per_case = 7
    initial_index = 0
    typical = cases[cases["case_role"].eq("typical")]
    if len(typical):
        initial_index = int(typical.index[0])
    for case_number, case in cases.reset_index(drop=True).iterrows():
        branch = points[points["branch_id"].eq(case["branch_id"])]
        cnn = branch[branch["method"].eq("CNN")]
        ridge = branch[branch["method"].eq("Ridge")]
        visible = case_number == initial_index
        has_curve = case["true_status"] == "complete"
        true_x = cnn["time_s"] if has_curve else []
        true_y = cnn["pybamm_voltage_V"] if has_curve else []
        traces = [
            go.Scatter(x=true_x, y=true_y, name="PyBaMM", line={"color": INK, "width": 3},
                       mode="lines", visible=visible, legendgroup="truth"),
            go.Scatter(x=cnn["time_s"], y=cnn["predicted_voltage_V"], name="CNN",
                       line={"color": BLUE, "width": 2.5}, mode="lines", visible=visible,
                       legendgroup="cnn"),
            go.Scatter(x=ridge["time_s"], y=ridge["predicted_voltage_V"], name="Ridge",
                       line={"color": ORANGE, "width": 2, "dash": "dash"}, mode="lines",
                       visible=visible, legendgroup="ridge"),
            go.Scatter(x=cnn["time_s"], y=cnn["signed_error_mV"], name="CNN error",
                       line={"color": BLUE, "width": 2}, mode="lines", visible=visible,
                       showlegend=False, legendgroup="cnn"),
            go.Scatter(x=ridge["time_s"], y=ridge["signed_error_mV"], name="Ridge error",
                       line={"color": ORANGE, "width": 1.8, "dash": "dash"}, mode="lines",
                       visible=visible, showlegend=False, legendgroup="ridge"),
            go.Bar(x=["CNN"], y=[case["cnn_cutoff_probability"]], name="CNN cutoff probability",
                   marker_color=BLUE, visible=visible, showlegend=False),
            go.Bar(x=["Ridge"], y=[case["ridge_cutoff_probability"]], name="Ridge cutoff probability",
                   marker_color=ORANGE, visible=visible, showlegend=False),
        ]
        for index, trace in enumerate(traces):
            row = 1 if index < 3 else 2 if index < 5 else 3
            fig.add_trace(trace, row=row, col=1)
        status_note = (
            "full 10 s reference"
            if has_curve
            else "physical cutoff: no full trajectory stored or extrapolated"
        )
        title = (
            f"{case['branch_id']} · {case['case_role'].replace('_', ' ')} · "
            f"SOC {case['initial_soc']:.3f} · {case['current_A']:g} A · {status_note}<br>"
            f"<sup>CNN: {case['cnn_predicted_status']} (p={case['cnn_cutoff_probability']:.3f}) · "
            f"Ridge: {case['ridge_predicted_status']} (p={case['ridge_cutoff_probability']:.3f})</sup>"
        )
        mask = [False] * (len(cases) * traces_per_case)
        start = case_number * traces_per_case
        mask[start : start + traces_per_case] = [True] * traces_per_case
        buttons.append(
            {
                "label": f"{case['history_id']} · {case['current_A']:g} A · {case['true_status']}",
                "method": "update",
                "args": [{"visible": mask}, {"title": {"text": title, "x": 0.02, "xanchor": "left"}}],
            }
        )
    initial_case = cases.reset_index(drop=True).iloc[initial_index]
    initial_status = (
        "full 10 s reference"
        if initial_case["true_status"] == "complete"
        else "physical cutoff: no full trajectory stored or extrapolated"
    )
    fig.update_layout(
        template="plotly_white",
        height=900,
        title={
            "text": (
                f"{initial_case['branch_id']} · {initial_case['case_role'].replace('_', ' ')} · "
                f"SOC {initial_case['initial_soc']:.3f} · {initial_case['current_A']:g} A · {initial_status}<br>"
                f"<sup>CNN: {initial_case['cnn_predicted_status']} "
                f"(p={initial_case['cnn_cutoff_probability']:.3f}) · Ridge: "
                f"{initial_case['ridge_predicted_status']} "
                f"(p={initial_case['ridge_cutoff_probability']:.3f})</sup>"
            ),
            "x": 0.02, "xanchor": "left",
        },
        font={"family": "Arial, sans-serif", "color": INK},
        hovermode="x unified",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.0, "xanchor": "right", "y": 1.19, "yanchor": "top",
                "showactive": True,
            }
        ],
        annotations=list(fig.layout.annotations)
        + [
            {
                "text": (
                    "PRECOMPUTED RESULT BROWSER · select one of three reproducibly chosen test histories "
                    "and its candidate current · no online model inference"
                ),
                "x": 0.0, "y": 1.18, "xref": "paper", "yref": "paper",
                "showarrow": False, "xanchor": "left", "font": {"size": 11, "color": "#61717C"},
            }
        ],
        margin={"l": 80, "r": 45, "t": 175, "b": 70},
    )
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1, gridcolor=GRID)
    fig.update_xaxes(title_text="Time in pulse (s)", row=1, col=1)
    fig.update_yaxes(title_text="Error (mV)", row=2, col=1, zeroline=True, zerolinecolor="#8C99A2")
    fig.update_xaxes(title_text="Time in pulse (s)", row=2, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1.04], row=3, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color=RED, annotation_text="CNN threshold 0.50",
                  row=3, col=1)
    fig.write_html(
        FIGURES / "interactive_test_browser.html",
        include_plotlyjs="cdn",
        full_html=True,
        div_id="battery-pulse-browser",
        config={"displaylogo": False, "responsive": True},
    )


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    static_overview()
    interactive_browser()
    print(FIGURES / "overview.png")
    print(FIGURES / "interactive_test_browser.html")


if __name__ == "__main__":
    main()
