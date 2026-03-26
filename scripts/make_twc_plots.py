from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from upair5g.utils import tensor7_to_btfnc

ROOT = Path("/home/rsadve1/PROJECT/Universal_Receiver")
FIG_DIR = ROOT / "TWC_plots"
TABLE_DIR = FIG_DIR / "tables"

MAIN_CURVES = ROOT / "outputs" / "twc_mild_main" / "metrics" / "curves.csv"
MAIN_EXAMPLE = ROOT / "outputs" / "twc_mild_main" / "artifacts" / "channel_example.npz"
CLEAN_CURVES = ROOT / "outputs" / "twc_mild_clean" / "metrics" / "curves.csv"
DMRSRICH_CURVES = ROOT / "outputs" / "twc_mild_dmrsrich" / "metrics" / "curves.csv"

SCENARIOS = {
    "Mild main": MAIN_CURVES,
    "Mild clean": CLEAN_CURVES,
    "Mild DMRS-rich": DMRSRICH_CURVES,
}

LABELS = {
    "baseline_ls_lmmse": "LS+LMMSE",
    "baseline_ddcpe_ls_lmmse": "DD-CPE+LS+LMMSE",
    "paper_cfgres_phaseaware_ls_lmmse": "Phase-aware paper cfg reservoir+LMMSE",
    "upair5g_lmmse": "UPAIR-5G+LMMSE",
    "perfect_csi_lmmse": "Perfect CSI+LMMSE",
}

MAIN_RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
    "upair5g_lmmse",
]

FOCUS_RECEIVERS = [
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]

CLASSICAL_RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
]

METRICS = ["ber", "bler", "nmse"]
LINESTYLES = {
    "baseline_ls_lmmse": "--",
    "baseline_ddcpe_ls_lmmse": "-.",
    "paper_cfgres_phaseaware_ls_lmmse": ":",
    "upair5g_lmmse": "-",
    "perfect_csi_lmmse": "-",
}
MARKERS = {
    "baseline_ls_lmmse": "s",
    "baseline_ddcpe_ls_lmmse": "D",
    "paper_cfgres_phaseaware_ls_lmmse": "^",
    "upair5g_lmmse": "o",
    "perfect_csi_lmmse": "v",
}
LINEWIDTHS = {
    "baseline_ls_lmmse": 1.7,
    "baseline_ddcpe_ls_lmmse": 1.8,
    "paper_cfgres_phaseaware_ls_lmmse": 1.9,
    "upair5g_lmmse": 2.4,
    "perfect_csi_lmmse": 2.0,
}


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    missing = []
    for name, path in SCENARIOS.items():
        if not path.exists():
            missing.append(str(path))
        else:
            frames[name] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError("Missing curves.csv files:\n" + "\n".join(missing))
    for name, df in frames.items():
        # Backward compatibility: older curves may not have reliability columns.
        if "reliable_ber" not in df.columns:
            df["reliable_ber"] = df["ber"] > 0.0
        if "reliable_bler" not in df.columns:
            df["reliable_bler"] = df["bler"] > 0.0
        frames[name] = df
    return frames


def _positive_floor_for_visible_data(frames: dict[str, pd.DataFrame], metric: str, receivers: list[str]) -> float:
    positives: list[float] = []
    reliability_col = {"ber": "reliable_ber", "bler": "reliable_bler"}.get(metric, "")
    for df in frames.values():
        sub = df[df["receiver"].isin(receivers)].copy()
        if reliability_col:
            sub = sub[sub[reliability_col].fillna(False)]
        arr = sub[metric].dropna().to_numpy(dtype=float)
        positives.extend(arr[arr > 0.0].tolist())
    if not positives:
        return 1e-6 if metric in {"ber", "bler"} else 1e-5
    minimum = min(positives)
    lower_bound = 1e-8 if metric in {"ber", "bler"} else 1e-5
    return max(minimum / 5.0, lower_bound)


def _metric_values_for_plot(sub: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    x = sub["ebno_db"].to_numpy(dtype=float)
    raw = sub[metric].to_numpy(dtype=float)
    if metric == "ber":
        reliable = sub["reliable_ber"].fillna(False).to_numpy(dtype=bool)
        raw = np.where(reliable, raw, np.nan)
    elif metric == "bler":
        reliable = sub["reliable_bler"].fillna(False).to_numpy(dtype=bool)
        raw = np.where(reliable, raw, np.nan)
    return x, raw


def _plot_receiver_curve(
    ax: plt.Axes,
    sub: pd.DataFrame,
    receiver: str,
    metric: str,
    floor: float,
) -> None:
    x, raw = _metric_values_for_plot(sub, metric)
    if metric in {"ber", "bler"}:
        positive = np.where(np.isfinite(raw) & (raw > 0.0), raw, np.nan)
        if np.all(np.isnan(positive)):
            return
        y = positive
        ax.semilogy(
            x,
            y,
            linestyle=LINESTYLES[receiver],
            marker=MARKERS[receiver],
            linewidth=LINEWIDTHS[receiver],
            markersize=5.5 if receiver == "upair5g_lmmse" else 4.8,
            label=LABELS[receiver],
        )
    else:
        ax.semilogy(
            x,
            raw,
            linestyle=LINESTYLES[receiver],
            marker=MARKERS[receiver],
            linewidth=LINEWIDTHS[receiver],
            markersize=5.5 if receiver == "upair5g_lmmse" else 4.8,
            label=LABELS[receiver],
        )


def _best_classical(df: pd.DataFrame, ebno_db: float, metric: str, reliable_only: bool = True) -> tuple[str, float] | None:
    classical = df[(df["receiver"].isin(CLASSICAL_RECEIVERS)) & (df["ebno_db"] == ebno_db)].copy()
    if metric == "ber" and reliable_only:
        classical = classical[classical["reliable_ber"].fillna(False)]
    elif metric == "bler" and reliable_only:
        classical = classical[classical["reliable_bler"].fillna(False)]
    classical = classical.dropna(subset=[metric]).sort_values(metric)
    if classical.empty:
        return None
    best = classical.iloc[0]
    return str(best["receiver"]), float(best[metric])


def _gain_ratio(best_value: float, upair_value: float) -> float:
    return best_value / max(upair_value, 1e-15)


def make_fig01_main_curves(frames: dict[str, pd.DataFrame]) -> None:
    df = frames["Mild main"]
    floor_map = {metric: _positive_floor_for_visible_data(frames, metric, MAIN_RECEIVERS) for metric in METRICS}
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.4), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for receiver in MAIN_RECEIVERS:
            sub = df[df["receiver"] == receiver].sort_values("ebno_db")
            if sub.empty:
                continue
            _plot_receiver_curve(ax, sub, receiver, metric, floor_map[metric])
        ax.set_ylabel(metric.upper())
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Eb/N0 [dB]")
    axes[0].legend(loc="best", frameon=False)
    axes[0].text(
        0.98,
        0.05,
        "BER/BLER points are shown only when enough\nobserved errors are available.",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.suptitle("Mild main scenario")
    _save(fig, FIG_DIR / "Fig01_mild_main_curves")


def make_fig02_main_focus(frames: dict[str, pd.DataFrame]) -> None:
    df = frames["Mild main"]
    floor_map = {metric: _positive_floor_for_visible_data(frames, metric, FOCUS_RECEIVERS) for metric in METRICS}
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.4), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for receiver in FOCUS_RECEIVERS:
            sub = df[df["receiver"] == receiver].sort_values("ebno_db")
            if sub.empty:
                continue
            _plot_receiver_curve(ax, sub, receiver, metric, floor_map[metric])
        ax.set_ylabel(metric.upper())
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Eb/N0 [dB]")
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle("Mild main scenario, strongest comparators and upper bound")
    _save(fig, FIG_DIR / "Fig02_mild_main_focus")


def make_fig03_main_gain(frames: dict[str, pd.DataFrame]) -> None:
    df = frames["Mild main"]
    upair = df[df["receiver"] == "upair5g_lmmse"].sort_values("ebno_db")
    rows = []

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.4), sharex=True)
    for ax, metric in zip(axes, METRICS):
        xs = []
        ys = []
        for ebno_db in upair["ebno_db"].tolist():
            best = _best_classical(df, float(ebno_db), metric, reliable_only=(metric in {"ber", "bler"}))
            upair_row = upair[upair["ebno_db"] == ebno_db].iloc[0]
            if metric == "ber" and not bool(upair_row["reliable_ber"]):
                ratio = np.nan
            elif metric == "bler" and not bool(upair_row["reliable_bler"]):
                ratio = np.nan
            elif best is None:
                ratio = np.nan
            else:
                best_name, best_value = best
                upair_value = float(upair_row[metric])
                ratio = _gain_ratio(best_value, upair_value)
                rows.append(
                    {
                        "ebno_db": float(ebno_db),
                        "metric": metric,
                        "best_classical_receiver": best_name,
                        "best_classical_value": best_value,
                        "upair_value": upair_value,
                        "best_classical_over_upair": ratio,
                    }
                )
            xs.append(float(ebno_db))
            ys.append(float(ratio) if np.isfinite(ratio) else np.nan)
        ax.semilogy(xs, ys, marker="o", linewidth=2.2)
        ax.axhline(1.0, linestyle=":", linewidth=1.0)
        ax.set_ylabel("Best classical / UPAIR")
        ax.set_title(metric.upper())
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Eb/N0 [dB]")
    _save(fig, FIG_DIR / "Fig03_mild_main_upair_gain")
    pd.DataFrame(rows).to_csv(TABLE_DIR / "mild_main_gain.csv", index=False)


def _to_magnitude_panel(x: np.ndarray) -> np.ndarray:
    return np.abs(tensor7_to_btfnc(tf.convert_to_tensor(x)).numpy()[0, :, :, 0])


def _to_error_panel(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    ref = tensor7_to_btfnc(tf.convert_to_tensor(reference)).numpy()[0, :, :, 0]
    est = tensor7_to_btfnc(tf.convert_to_tensor(estimate)).numpy()[0, :, :, 0]
    return np.abs(ref - est)


def make_fig04_channel_maps() -> None:
    if not MAIN_EXAMPLE.exists():
        raise FileNotFoundError(str(MAIN_EXAMPLE))
    data = np.load(MAIN_EXAMPLE)
    keys = [
        ("h_ls_linear", "LS+LMMSE"),
        ("h_ddcpe_ls", "DD-CPE+LS+LMMSE"),
        ("h_paper_cfgres_phaseaware", "Phase-aware paper cfg reservoir+LMMSE"),
        ("h_prop", "UPAIR-5G+LMMSE"),
    ]
    keys = [(k, t) for k, t in keys if k in data]
    if not keys:
        return

    mags = [_to_magnitude_panel(data["h_true"])]
    errs = [np.zeros_like(mags[0])]
    titles_top = ["True |H|"]
    titles_bottom = ["Absolute error"]
    for key, title in keys:
        mags.append(_to_magnitude_panel(data[key]))
        errs.append(_to_error_panel(data["h_true"], data[key]))
        titles_top.append(title)
        titles_bottom.append(title)

    fig, axes = plt.subplots(2, len(mags), figsize=(16.0, 6.2))
    vmax_mag = max(np.max(x) for x in mags)
    vmax_err = max(np.max(x) for x in errs[1:]) if len(errs) > 1 else 1.0

    for col, panel in enumerate(mags):
        im0 = axes[0, col].imshow(panel, aspect="auto", origin="upper", vmin=0.0, vmax=vmax_mag)
        axes[0, col].set_title(titles_top[col], fontsize=11)
        axes[0, col].set_xlabel("Subcarrier")
        axes[0, col].set_ylabel("OFDM symbol")
    for col, panel in enumerate(errs):
        im1 = axes[1, col].imshow(panel, aspect="auto", origin="upper", vmin=0.0, vmax=vmax_err)
        axes[1, col].set_title(titles_bottom[col], fontsize=11)
        axes[1, col].set_xlabel("Subcarrier")
        axes[1, col].set_ylabel("OFDM symbol")

    fig.colorbar(im0, ax=axes[0, :].ravel().tolist(), shrink=0.82)
    fig.colorbar(im1, ax=axes[1, :].ravel().tolist(), shrink=0.82)
    _save(fig, FIG_DIR / "Fig04_mild_main_channel_error_maps")


def make_fig05_crossscenario_grid(frames: dict[str, pd.DataFrame]) -> None:
    floor_map = {metric: _positive_floor_for_visible_data(frames, metric, MAIN_RECEIVERS) for metric in METRICS}
    scenarios = list(SCENARIOS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15.4, 11.0), sharex="col")
    for col, scenario in enumerate(scenarios):
        df = frames[scenario]
        axes[0, col].set_title(scenario)
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            for receiver in MAIN_RECEIVERS:
                sub = df[df["receiver"] == receiver].sort_values("ebno_db")
                if sub.empty:
                    continue
                _plot_receiver_curve(ax, sub, receiver, metric, floor_map[metric])
            if col == 0:
                ax.set_ylabel(metric.upper())
            if row == 2:
                ax.set_xlabel("Eb/N0 [dB]")
            ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
    _save(fig, FIG_DIR / "Fig05_mild_crossscenario_metrics_grid")


def make_fig06_crossscenario_gain(frames: dict[str, pd.DataFrame]) -> None:
    scenarios = list(SCENARIOS.keys())
    rows = []
    for scenario in scenarios:
        df = frames[scenario]
        upair = df[df["receiver"] == "upair5g_lmmse"].copy().sort_values("ebno_db")
        for ebno_db in upair["ebno_db"].tolist():
            row = {"scenario": scenario, "ebno_db": float(ebno_db)}
            upair_row = upair[upair["ebno_db"] == ebno_db].iloc[0]
            for metric in METRICS:
                reliable = True
                if metric == "ber":
                    reliable = bool(upair_row["reliable_ber"])
                elif metric == "bler":
                    reliable = bool(upair_row["reliable_bler"])
                best = _best_classical(df, float(ebno_db), metric, reliable_only=(metric in {"ber", "bler"}))
                if (not reliable) or best is None:
                    row[f"best_{metric}_classical_receiver"] = None
                    row[f"best_{metric}_classical"] = np.nan
                    row[f"{metric}_upair"] = np.nan
                    row[f"{metric}_ratio_best_classical_over_upair"] = np.nan
                else:
                    best_name, best_value = best
                    upair_value = float(upair_row[metric])
                    row[f"best_{metric}_classical_receiver"] = best_name
                    row[f"best_{metric}_classical"] = best_value
                    row[f"{metric}_upair"] = upair_value
                    row[f"{metric}_ratio_best_classical_over_upair"] = _gain_ratio(best_value, upair_value)
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "mild_crossscenario_gain.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for scenario in scenarios:
            sub = summary[summary["scenario"] == scenario].copy().sort_values("ebno_db")
            x = sub["ebno_db"].to_numpy(dtype=float)
            y = sub[f"{metric}_ratio_best_classical_over_upair"].to_numpy(dtype=float)
            ax.semilogy(x, y, marker="o", linewidth=2.0, label=scenario)
        ax.axhline(1.0, linestyle=":", linewidth=1.0)
        ax.set_title(metric.upper())
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel("Best classical / UPAIR")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(frameon=False)
    _save(fig, FIG_DIR / "Fig06_mild_crossscenario_upair_gain")


def write_manifest() -> None:
    lines = [
        "Fig01_mild_main_curves.pdf/png",
        "Fig02_mild_main_focus.pdf/png",
        "Fig03_mild_main_upair_gain.pdf/png",
        "Fig04_mild_main_channel_error_maps.pdf/png",
        "Fig05_mild_crossscenario_metrics_grid.pdf/png",
        "Fig06_mild_crossscenario_upair_gain.pdf/png",
        "tables/mild_main_gain.csv",
        "tables/mild_crossscenario_gain.csv",
    ]
    (FIG_DIR / "TWC_plot_manifest.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    _ensure_dirs()
    frames = _load_frames()
    make_fig01_main_curves(frames)
    make_fig02_main_focus(frames)
    make_fig03_main_gain(frames)
    make_fig04_channel_maps()
    make_fig05_crossscenario_grid(frames)
    make_fig06_crossscenario_gain(frames)
    write_manifest()
    print({"twc_plots_dir": str(FIG_DIR)})


if __name__ == "__main__":
    main()
