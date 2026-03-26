from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/rsadve1/PROJECT/Universal_Receiver_NEW")
FIG_DIR = ROOT / "TWC_plots2"
CSV_DIR = FIG_DIR / "csv"

SCENARIOS = OrderedDict(
    [
        ("twc_mild_clean", ("Clean reference", CSV_DIR / "twc_mild_clean_curves.csv")),
        ("twc_mild_main", ("Mild phase distortion", CSV_DIR / "twc_mild_main_curves.csv")),
        (
            "twc_mild_dmrsrich",
            ("Mild phase distortion + additional DMRS", CSV_DIR / "twc_mild_dmrsrich_curves.csv"),
        ),
    ]
)

RECEIVERS_BLER = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]
RECEIVERS_NMSE = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "upair5g_lmmse",
]

LABELS = {
    "baseline_ls_lmmse": "LS",
    "baseline_ls_2dlmmse_lmmse": "LS + 2D LMMSE",
    "baseline_ddcpe_ls_lmmse": "DD-CPE + LS",
    "upair5g_lmmse": "UPAIR-5G",
    "perfect_csi_lmmse": "Perfect CSI",
}

STYLES = {
    "baseline_ls_lmmse": {
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "o",
        "linewidth": 1.9,
    },
    "baseline_ls_2dlmmse_lmmse": {
        "color": "#2ca02c",
        "linestyle": ":",
        "marker": "D",
        "linewidth": 2.0,
    },
    "baseline_ddcpe_ls_lmmse": {
        "color": "#ff7f0e",
        "linestyle": "-.",
        "marker": "s",
        "linewidth": 1.9,
    },
    "upair5g_lmmse": {
        "color": "#d62728",
        "linestyle": "-",
        "marker": "^",
        "linewidth": 2.4,
    },
    "perfect_csi_lmmse": {
        "color": "#9467bd",
        "linestyle": "-",
        "marker": "v",
        "linewidth": 2.0,
    },
}


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)



def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)



def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=250, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)



def _load_frames() -> OrderedDict[str, tuple[str, pd.DataFrame]]:
    frames: OrderedDict[str, tuple[str, pd.DataFrame]] = OrderedDict()
    missing: list[str] = []
    all_rows: list[pd.DataFrame] = []
    for scenario_key, (title, path) in SCENARIOS.items():
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        if "reliable_ber" not in df.columns:
            df["reliable_ber"] = False
        if "reliable_bler" not in df.columns:
            df["reliable_bler"] = False
        if "scenario" not in df.columns:
            df["scenario"] = scenario_key
        frames[scenario_key] = (title, df)
        all_rows.append(df)
    if missing:
        raise FileNotFoundError("Missing TWC_plots2 CSV files:\n" + "\n".join(missing))
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(CSV_DIR / "twc_all_curves.csv", index=False)
    return frames



def _plot_one_receiver(ax: plt.Axes, sub: pd.DataFrame, receiver: str, metric: str) -> None:
    style = STYLES[receiver]
    x = sub["ebno_db"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)

    if metric == "bler":
        line_mask = np.isfinite(y) & (y > 0.0)
        marker_mask = line_mask & sub["reliable_bler"].fillna(False).to_numpy(dtype=bool)
    elif metric == "ber":
        line_mask = np.isfinite(y) & (y > 0.0)
        marker_mask = line_mask & sub["reliable_ber"].fillna(False).to_numpy(dtype=bool)
    else:
        line_mask = np.isfinite(y) & (y > 0.0)
        marker_mask = line_mask

    if not np.any(line_mask):
        return

    ax.semilogy(
        x[line_mask],
        y[line_mask],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        label=LABELS[receiver],
    )
    if np.any(marker_mask):
        ax.semilogy(
            x[marker_mask],
            y[marker_mask],
            linestyle="None",
            marker=style["marker"],
            markersize=6.0 if receiver == "upair5g_lmmse" else 5.2,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.4,
        )



def _format_axis(ax: plt.Axes, ylabel: str, xlim: tuple[float, float]) -> None:
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.grid(True, which="both", alpha=0.25)



def make_fig04_main_bler(frames: OrderedDict[str, tuple[str, pd.DataFrame]]) -> None:
    _, df = frames["twc_mild_main"]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for receiver in RECEIVERS_BLER:
        sub = df[df["receiver"] == receiver].sort_values("ebno_db")
        if not sub.empty:
            _plot_one_receiver(ax, sub, receiver, "bler")
    _format_axis(ax, "BLER", (0.0, 14.0))
    ax.legend(loc="upper right", frameon=True)
    _save(fig, "Fig04_mild_main_bler")



def make_fig05_main_nmse(frames: OrderedDict[str, tuple[str, pd.DataFrame]]) -> None:
    _, df = frames["twc_mild_main"]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for receiver in RECEIVERS_NMSE:
        sub = df[df["receiver"] == receiver].sort_values("ebno_db")
        if not sub.empty:
            _plot_one_receiver(ax, sub, receiver, "nmse")
    _format_axis(ax, "NMSE", (0.0, 14.0))
    ax.legend(loc="upper right", frameon=True)
    _save(fig, "Fig05_mild_main_nmse")



def _crossscenario_panel(
    fig_stem: str,
    frames: OrderedDict[str, tuple[str, pd.DataFrame]],
    metric: str,
    receivers: list[str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), sharey=True)
    legend_handles = None
    legend_labels = None

    for ax, (scenario_key, (title, df)) in zip(axes, frames.items()):
        for receiver in receivers:
            sub = df[df["receiver"] == receiver].sort_values("ebno_db")
            if not sub.empty:
                _plot_one_receiver(ax, sub, receiver, metric)
        ax.set_title(title)
        x_max = 10.0 if scenario_key == "twc_mild_dmrsrich" else 14.0
        _format_axis(ax, "BLER" if metric == "bler" else "NMSE", (0.0, x_max))
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(receivers),
            frameon=True,
            bbox_to_anchor=(0.5, 1.10),
        )
    _save(fig, fig_stem)



def main() -> None:
    _ensure_dirs()
    frames = _load_frames()
    make_fig04_main_bler(frames)
    make_fig05_main_nmse(frames)
    _crossscenario_panel("Fig06_crossscenario_bler", frames, "bler", RECEIVERS_BLER)
    _crossscenario_panel("Fig07_crossscenario_nmse", frames, "nmse", RECEIVERS_NMSE)
    print(f"[TWC_PLOTS2] figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
