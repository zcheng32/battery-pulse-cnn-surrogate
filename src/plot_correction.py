"""Rebuild the online-correction overview and browser from tracked result tables."""

from __future__ import annotations

import json
import os

from common import LOCAL, RESULTS, ROOT

os.environ.setdefault("MPLCONFIGDIR", str(LOCAL / "matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from plotly.subplots import make_subplots


FIGURES = ROOT / "figures"
BLUE = "#2463A7"
ORANGE = "#E17C35"
GREEN = "#2A9D78"
RED = "#C94C4C"
PURPLE = "#8064A2"
INK = "#24313D"
MUTED = "#60717D"
GRID = "#DCE3E8"
BACKGROUND = "#F7F9FA"


def box(ax, x, y, width, height, text, color, fontsize=9.5):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color,
        edgecolor="none",
        alpha=0.14,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        weight="bold",
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            color="#7E8D98",
            lw=1.5,
            connectionstyle="arc3,rad=0.0",
        )
    )


def method_style(method):
    styles = {
        "Open-loop": (ORANGE, "--", "o"),
        "Linear corrector": (BLUE, "-", "o"),
        "Nonlinear corrector seed 42": (GREEN, ":", "s"),
    }
    return styles[method]


def static_overview():
    display = pd.read_csv(RESULTS / "correction_display_predictions.csv")
    steps = pd.read_csv(RESULTS / "correction_step_metrics.csv")
    metrics = pd.read_csv(RESULTS / "correction_test_metrics.csv")
    summary = json.loads((RESULTS / "correction_summary.json").read_text())
    selection = json.loads((RESULTS / "correction_display_selection.json").read_text())
    clean_steps = steps[steps["measurement_noise_mV"].eq(0.0)]
    shown_methods = ["Open-loop", "Linear corrector", "Nonlinear corrector seed 42"]

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
    fig = plt.figure(figsize=(14, 9), facecolor=BACKGROUND)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.98,
        top=0.87,
        bottom=0.085,
        hspace=0.34,
        wspace=0.24,
    )
    fig.suptitle(
        "Correcting a learned battery state with measured voltage",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=22,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.916,
        "A 10-second CNN forecast is updated after each pulse; the corrected state starts the next forecast.",
        ha="left",
        fontsize=11.5,
        color=MUTED,
    )

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A  Online correction loop", loc="left", weight="bold")
    box(ax, 0.02, 0.57, 0.20, 0.18, "Current state\n$\\hat{s}_k$", BLUE)
    box(ax, 0.31, 0.57, 0.22, 0.18, "Frozen CNN\nforecast", BLUE)
    box(ax, 0.67, 0.70, 0.27, 0.16, "Predicted voltage\n$\\hat{V}_k(t)$", ORANGE)
    box(ax, 0.67, 0.44, 0.27, 0.16, "Predicted next state\n$\\hat{s}_{k+1}^{-}$", GREEN)
    box(ax, 0.31, 0.16, 0.22, 0.17, "Voltage\ninnovation", RED)
    box(ax, 0.67, 0.13, 0.27, 0.20, "Low-rank corrector\n$\\hat{s}_{k+1}^{+}$", PURPLE)
    ax.text(0.02, 0.29, "Measured after pulse", fontsize=9, color=MUTED)
    box(ax, 0.02, 0.08, 0.20, 0.16, "$V_k(t)$", RED, 10.5)
    arrow(ax, (0.22, 0.66), (0.31, 0.66))
    arrow(ax, (0.53, 0.66), (0.67, 0.78))
    arrow(ax, (0.53, 0.66), (0.67, 0.52))
    arrow(ax, (0.22, 0.16), (0.31, 0.245))
    arrow(ax, (0.67, 0.75), (0.53, 0.28))
    arrow(ax, (0.53, 0.245), (0.67, 0.23))
    arrow(ax, (0.80, 0.44), (0.80, 0.33))
    ax.add_patch(
        FancyArrowPatch(
            (0.80, 0.13),
            (0.12, 0.57),
            arrowstyle="-|>",
            mutation_scale=12,
            color="#7E8D98",
            lw=1.5,
            connectionstyle="arc3,rad=-0.31",
        )
    )
    ax.text(
        0.03,
        0.93,
        "The correction changes the next pulse, not the pulse already observed.",
        fontsize=9.5,
        color=MUTED,
    )

    ax = fig.add_subplot(grid[0, 1])
    truth = display[display["method"].eq("Open-loop")]
    ax.plot(
        truth["elapsed_s"],
        truth["pybamm_voltage_V"],
        color=INK,
        lw=2.4,
        label="PyBaMM",
        zorder=4,
    )
    for method in shown_methods:
        subset = display[display["method"].eq(method)]
        color, linestyle, _ = method_style(method)
        ax.plot(
            subset["elapsed_s"],
            subset["predicted_voltage_V"],
            color=color,
            lw=1.7 if method != "Linear corrector" else 2.1,
            ls=linestyle,
            label=method.replace(" seed 42", ""),
        )
    ax.set_title(
        f"B  Precomputed test sequence  ·  {selection['sequence_id']}",
        loc="left",
        weight="bold",
    )
    ax.set_xlabel("Elapsed rollout time (s)")
    ax.set_ylabel("Terminal voltage (V)")
    ax.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    current_by_step = truth.groupby("step", as_index=False)["current_A"].first()
    current_text = "  ".join(
        f"{int(row.step)}:{row.current_A:g}A" for row in current_by_step.itertuples()
    )
    ax.text(
        0.01,
        0.99,
        "pulse:current  " + current_text,
        transform=ax.transAxes,
        va="top",
        fontsize=7.2,
        color=MUTED,
    )

    ax = fig.add_subplot(grid[1, 0])
    for method in shown_methods:
        subset = clean_steps[clean_steps["method"].eq(method)]
        color, linestyle, marker = method_style(method)
        ax.plot(
            subset["elapsed_s"],
            subset["voltage_mae_mV"],
            color=color,
            ls=linestyle,
            marker=marker,
            markersize=4,
            lw=2,
            label=method.replace(" seed 42", ""),
        )
    ax.set_xlabel("Elapsed rollout time (s)")
    ax.set_ylabel("Voltage MAE (mV)")
    ax.set_title("C  Error accumulation across unseen sequences", loc="left", weight="bold")
    ax.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.98,
        0.06,
        f"all complete test steps\n{summary['test_complete_steps']} pulse transitions",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID},
    )

    ax = fig.add_subplot(grid[1, 1])
    for method in shown_methods:
        subset = clean_steps[clean_steps["method"].eq(method)]
        color, linestyle, marker = method_style(method)
        ax.plot(
            subset["elapsed_s"],
            100.0 * subset["posterior_state_rmse_fraction"],
            color=color,
            ls=linestyle,
            marker=marker,
            markersize=4,
            lw=2,
            label=method.replace(" seed 42", ""),
        )
    ax.set_xlabel("Elapsed rollout time (s)")
    ax.set_ylabel("State RMSE (% of concentration range)")
    ax.set_title("D  Voltage feedback limits state drift", loc="left", weight="bold")
    ax.grid(True, color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    clean = metrics[metrics["measurement_noise_mV"].eq(0.0)].set_index("method")
    noisy = metrics[
        metrics["measurement_noise_mV"].eq(5.0)
        & metrics["method"].eq("Linear corrector")
    ].iloc[0]
    open_row = clean.loc["Open-loop"]
    linear_row = clean.loc["Linear corrector"]
    state_reduction = 100.0 * (
        1.0 - linear_row["posterior_state_rmse_fraction"] / open_row["posterior_state_rmse_fraction"]
    )
    ax.text(
        0.98,
        0.94,
        f"Linear corrector selected on validation\n"
        f"Voltage MAE  {open_row['voltage_mae_mV']:.1f} → {linear_row['voltage_mae_mV']:.1f} mV\n"
        f"State RMSE reduced  {state_reduction:.1f}%\n"
        f"With 5 mV sensor noise  {noisy['voltage_mae_mV']:.1f} mV",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID},
    )
    fig.text(
        0.06,
        0.027,
        "36 unseen test sequences · exact simulated state only at initialization · synthetic voltage measurements · no real-cell validation",
        fontsize=9.5,
        color=MUTED,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / "online_correction.png",
        dpi=220,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def interactive_browser():
    display = pd.read_csv(RESULTS / "correction_display_predictions.csv")
    metrics = pd.read_csv(RESULTS / "correction_test_metrics.csv")
    selection = json.loads((RESULTS / "correction_display_selection.json").read_text())
    methods = ["Open-loop", "Linear corrector", "Nonlinear corrector seed 42"]
    colors = {"Open-loop": ORANGE, "Linear corrector": BLUE, "Nonlinear corrector seed 42": GREEN}
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.56, 0.28, 0.16],
        vertical_spacing=0.08,
        subplot_titles=(
            "Terminal voltage",
            "Signed prediction error",
            "Applied current schedule",
        ),
    )
    truth = display[display["method"].eq("Open-loop")]
    fig.add_trace(
        go.Scatter(
            x=truth["elapsed_s"],
            y=truth["pybamm_voltage_V"],
            name="PyBaMM",
            line={"color": INK, "width": 3},
            hovertemplate="%{x:.2f} s<br>%{y:.4f} V<extra>PyBaMM</extra>",
        ),
        row=1,
        col=1,
    )
    for method in methods:
        subset = display[display["method"].eq(method)]
        label = method.replace(" seed 42", "")
        dash = "dash" if method == "Open-loop" else "dot" if method.startswith("Nonlinear") else "solid"
        fig.add_trace(
            go.Scatter(
                x=subset["elapsed_s"],
                y=subset["predicted_voltage_V"],
                name=label,
                line={"color": colors[method], "width": 2, "dash": dash},
                hovertemplate="%{x:.2f} s<br>%{y:.4f} V<extra>" + label + "</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["elapsed_s"],
                y=subset["signed_error_mV"],
                name=label + " error",
                showlegend=False,
                line={"color": colors[method], "width": 1.8, "dash": dash},
                hovertemplate="%{x:.2f} s<br>%{y:.1f} mV<extra>" + label + "</extra>",
            ),
            row=2,
            col=1,
        )
    current = truth.groupby("step", as_index=False)["current_A"].first()
    fig.add_trace(
        go.Bar(
            x=(current["step"] - 0.5) * 10,
            y=current["current_A"],
            width=9.7,
            name="Current",
            marker_color=PURPLE,
            opacity=0.72,
            showlegend=False,
            hovertemplate="Pulse %{customdata}<br>%{y:g} A<extra></extra>",
            customdata=current["step"],
        ),
        row=3,
        col=1,
    )
    for boundary in np.arange(10, 120, 10):
        fig.add_vline(x=boundary, line_width=0.7, line_color=GRID, row="all", col=1)
    clean = metrics[metrics["measurement_noise_mV"].eq(0.0)].set_index("method")
    open_row = clean.loc["Open-loop"]
    linear_row = clean.loc["Linear corrector"]
    fig.update_layout(
        template="plotly_white",
        height=900,
        title={
            "text": (
                f"Precomputed online-correction rollout · {selection['sequence_id']}<br>"
                f"<sup>Full unseen-test MAE: open loop {open_row['voltage_mae_mV']:.2f} mV · "
                f"validation-selected linear corrector {linear_row['voltage_mae_mV']:.2f} mV · "
                "the browser does not run a model</sup>"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        font={"family": "Arial, sans-serif", "color": INK},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0.02},
        margin={"l": 80, "r": 35, "t": 125, "b": 70},
        annotations=[
            {
                "text": (
                    "Synthetic Chen2020 SPM reference. Exact simulated state is used only at sequence initialization; "
                    "each correction uses voltage observed during the preceding completed pulse."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.11,
                "showarrow": False,
                "xanchor": "left",
                "font": {"size": 11, "color": MUTED},
            }
        ],
    )
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Error (mV)", zeroline=True, zerolinecolor="#9AA6AE", row=2, col=1)
    fig.update_yaxes(title_text="Current (A)", row=3, col=1)
    fig.update_xaxes(title_text="Elapsed rollout time (s)", row=3, col=1)
    fig.write_html(
        FIGURES / "online_correction_browser.html",
        include_plotlyjs="cdn",
        full_html=True,
        div_id="online-correction-browser",
        config={"displaylogo": False, "responsive": True},
    )


def main():
    static_overview()
    interactive_browser()
    print(f"Wrote {FIGURES / 'online_correction.png'}")
    print(f"Wrote {FIGURES / 'online_correction_browser.html'}")


if __name__ == "__main__":
    main()
