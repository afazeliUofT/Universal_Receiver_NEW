from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ensure_output_tree
from .utils import tensor7_to_btfnc

PLOT_LABELS = {
    "baseline_ls_lmmse": "LS",
    "baseline_ls_timeavg_lmmse": "LS + time-avg",
    "baseline_ls_2dlmmse_lmmse": "LS + 2D LMMSE",
    "baseline_ddcpe_ls_lmmse": "DD-CPE + LS",
    "paper_cfgres_ls_lmmse": "Configured reservoir",
    "paper_cfgres_phaseaware_ls_lmmse": "Configured reservoir",
    "upair5g_lmmse": "UPAIR-5G",
    "perfect_csi_lmmse": "Perfect CSI",
}

PLOT_ORDER = [
    "baseline_ls_lmmse",
    "baseline_ls_timeavg_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]

METRIC_LABELS = {
    "ber": "BER",
    "bler": "BLER",
    "nmse": "NMSE",
}


class _Style:
    def __init__(self, linewidth: float, marker: str) -> None:
        self.linewidth = linewidth
        self.marker = marker


def _style(receiver: str) -> _Style:
    if receiver == "upair5g_lmmse":
        return _Style(linewidth=2.6, marker="o")
    if receiver == "paper_cfgres_phaseaware_ls_lmmse":
        return _Style(linewidth=2.4, marker="D")
    if receiver == "paper_cfgres_ls_lmmse":
        return _Style(linewidth=2.2, marker="s")
    if receiver == "perfect_csi_lmmse":
        return _Style(linewidth=2.0, marker="^")
    return _Style(linewidth=1.6, marker="o")



def _label(receiver: str) -> str:
    return PLOT_LABELS.get(receiver, receiver)



def _receiver_sort_key(receiver: str) -> tuple[int, str]:
    try:
        return (PLOT_ORDER.index(receiver), receiver)
    except ValueError:
        return (len(PLOT_ORDER), receiver)



def _save_figure(fig: plt.Figure, out_path_no_ext: Path, dpi: int = 250) -> None:
    out_path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_no_ext.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path_no_ext.with_suffix(".pdf"), bbox_inches="tight")



def _ordered_receivers(df: pd.DataFrame) -> list[str]:
    return sorted(df["receiver"].unique().tolist(), key=_receiver_sort_key)



def plot_training_history(history_path: str | Path, out_dir: str | Path) -> None:
    history_path = Path(history_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        return

    with open(history_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("history", [])
    if not rows:
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(df["step"], df["loss"], label="train loss")
    if "val_nmse_prop" in df.columns:
        valid = df["val_nmse_prop"].notna()
        if valid.any():
            ax.plot(df.loc[valid, "step"], df.loc[valid, "val_nmse_prop"], label="val NMSE (UPAIR-5G)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, out_dir / "training_history")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(df["step"], df["nmse_ls"], label="LS NMSE")
    ax.plot(df["step"], df["nmse_prop"], label="UPAIR-5G NMSE")
    ax.set_xlabel("Step")
    ax.set_ylabel("NMSE")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save_figure(fig, out_dir / "training_nmse")
    plt.close(fig)



def _plot_curve(df: pd.DataFrame, metric: str, out_path_no_ext: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for receiver in _ordered_receivers(df):
        sub = df[df["receiver"] == receiver].sort_values("ebno_db")
        if sub.empty or not sub[metric].notna().any():
            continue
        y = np.maximum(sub[metric].to_numpy(dtype=float), 1e-12)
        style = _style(receiver)
        ax.semilogy(sub["ebno_db"], y, marker=style.marker, linewidth=style.linewidth, label=_label(receiver))
    ax.set_xlabel("Eb/N0 [dB]")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric.upper()))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save_figure(fig, out_path_no_ext)
    plt.close(fig)



def plot_curves(curves_path: str | Path, out_dir: str | Path) -> None:
    curves_path = Path(curves_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not curves_path.exists():
        return

    df = pd.read_csv(curves_path)
    for metric in ["ber", "bler", "nmse"]:
        if metric in df.columns and df[metric].notna().any():
            _plot_curve(df, metric, out_dir / f"{metric}_vs_ebno")



def _to_magnitude_panel(tensor_like: np.ndarray) -> np.ndarray:
    h_btfnc = tensor7_to_btfnc(tensor_like).numpy()
    return np.abs(h_btfnc[0, :, :, 0])



def _to_abs_error_panel(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    ref_btfnc = tensor7_to_btfnc(reference).numpy()
    est_btfnc = tensor7_to_btfnc(estimate).numpy()
    err = np.abs(ref_btfnc[0, :, :, 0] - est_btfnc[0, :, :, 0])
    return err



def plot_channel_example(example_npz: str | Path, out_dir: str | Path) -> None:
    example_npz = Path(example_npz)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not example_npz.exists():
        return

    data = np.load(example_npz)
    panels: list[tuple[str, np.ndarray]] = [("True |H|", _to_magnitude_panel(data["h_true"]))]

    optional_keys = [
        ("h_ls_linear", "LS+linear |H|"),
        ("h_ls_timeavg", "LS+time-avg |H|"),
        ("h_ls_2dlmmse", "LS+2D-LMMSE |H|"),
        ("h_ddcpe_ls", "DD-CPE+LS |H|"),
        ("h_paper_cfgres", "Paper cfg reservoir |H|"),
        ("h_paper_cfgres_phaseaware", "Phase-aware paper cfg reservoir |H|"),
        ("h_prop", "UPAIR-5G |H|"),
    ]
    for key, title in optional_keys:
        if key in data:
            panels.append((title, _to_magnitude_panel(data[key])))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, mag) in zip(axes, panels):
        ax.imshow(mag, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Subcarrier")
        ax.set_ylabel("OFDM symbol")

    _save_figure(fig, out_dir / "channel_refinement_example")
    plt.close(fig)



def plot_publication_main_curves(curves_path: str | Path, out_dir: str | Path) -> None:
    curves_path = Path(curves_path)
    out_dir = Path(out_dir)
    if not curves_path.exists():
        return
    df = pd.read_csv(curves_path)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.4), sharex=True)
    metrics = ["ber", "bler", "nmse"]
    for ax, metric in zip(axes, metrics):
        for receiver in _ordered_receivers(df):
            sub = df[df["receiver"] == receiver].sort_values("ebno_db")
            if sub.empty or not sub[metric].notna().any():
                continue
            y = np.maximum(sub[metric].to_numpy(dtype=float), 1e-12)
            style = _style(receiver)
            ax.semilogy(sub["ebno_db"], y, marker=style.marker, linewidth=style.linewidth, label=_label(receiver))
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Eb/N0 [dB]")
    axes[0].legend(loc="best", fontsize=9)
    _save_figure(fig, out_dir / "publication_main_curves")
    plt.close(fig)



def _best_non_upair(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[~df["receiver"].isin(["upair5g_lmmse", "perfect_csi_lmmse"])].copy()
    rows = []
    for ebno_db, grp in sub.groupby("ebno_db"):
        grp = grp.dropna(subset=[metric])
        if grp.empty:
            continue
        best_idx = grp[metric].idxmin()
        rows.append(grp.loc[best_idx])
    return pd.DataFrame(rows)



def plot_upair_gains(curves_path: str | Path, out_dir: str | Path) -> None:
    curves_path = Path(curves_path)
    out_dir = Path(out_dir)
    if not curves_path.exists():
        return
    df = pd.read_csv(curves_path)
    if "upair5g_lmmse" not in df["receiver"].unique():
        return

    upair = df[df["receiver"] == "upair5g_lmmse"].sort_values("ebno_db")
    rows = []
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.4), sharex=True)
    for ax, metric in zip(axes, ["ber", "bler", "nmse"]):
        ref = _best_non_upair(df, metric).sort_values("ebno_db")
        merged = pd.merge(
            upair[["ebno_db", metric]],
            ref[["ebno_db", "receiver", metric]],
            on="ebno_db",
            suffixes=("_upair", "_ref"),
        )
        if merged.empty:
            continue
        gain = np.maximum(merged[f"{metric}_ref"].to_numpy(dtype=float), 1e-12) / np.maximum(merged[f"{metric}_upair"].to_numpy(dtype=float), 1e-12)
        ax.semilogy(merged["ebno_db"], gain, marker="o", linewidth=2.2)
        ax.set_ylabel(f"best non-UPAIR / UPAIR\n{METRIC_LABELS[metric]}")
        ax.grid(True, which="both", alpha=0.3)
        for _, row in merged.iterrows():
            rows.append(
                {
                    "metric": metric,
                    "ebno_db": float(row["ebno_db"]),
                    "best_non_upair_receiver": str(row["receiver"]),
                    "best_non_upair_value": float(row[f"{metric}_ref"]),
                    "upair_value": float(row[f"{metric}_upair"]),
                    "gain_factor_best_non_upair_over_upair": float(
                        max(float(row[f"{metric}_ref"]), 1e-12) / max(float(row[f"{metric}_upair"]), 1e-12)
                    ),
                }
            )
    axes[-1].set_xlabel("Eb/N0 [dB]")
    _save_figure(fig, out_dir / "publication_upair_gain_over_best_nonupair")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "publication_upair_gain_over_best_nonupair.csv", index=False)



def plot_paper_vs_upair(curves_path: str | Path, out_dir: str | Path) -> None:
    curves_path = Path(curves_path)
    out_dir = Path(out_dir)
    if not curves_path.exists():
        return
    df = pd.read_csv(curves_path)
    receivers = set(df["receiver"].unique().tolist())
    if not {"upair5g_lmmse", "paper_cfgres_ls_lmmse"}.issubset(receivers):
        return

    upair = df[df["receiver"] == "upair5g_lmmse"].sort_values("ebno_db")
    paper = df[df["receiver"] == "paper_cfgres_ls_lmmse"].sort_values("ebno_db")
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.4), sharex=True)
    rows = []
    for ax, metric in zip(axes, ["ber", "bler", "nmse"]):
        merged = pd.merge(upair[["ebno_db", metric]], paper[["ebno_db", metric]], on="ebno_db", suffixes=("_upair", "_paper"))
        if merged.empty:
            continue
        ratio = np.maximum(merged[f"{metric}_paper"].to_numpy(dtype=float), 1e-12) / np.maximum(merged[f"{metric}_upair"].to_numpy(dtype=float), 1e-12)
        ax.semilogy(merged["ebno_db"], ratio, marker="s", linewidth=2.1)
        ax.set_ylabel(f"Paper / UPAIR\n{METRIC_LABELS[metric]}")
        ax.grid(True, which="both", alpha=0.3)
        for _, row in merged.iterrows():
            rows.append(
                {
                    "metric": metric,
                    "ebno_db": float(row["ebno_db"]),
                    "paper_value": float(row[f"{metric}_paper"]),
                    "upair_value": float(row[f"{metric}_upair"]),
                    "paper_over_upair": float(max(float(row[f"{metric}_paper"]), 1e-12) / max(float(row[f"{metric}_upair"]), 1e-12)),
                }
            )
    axes[-1].set_xlabel("Eb/N0 [dB]")
    _save_figure(fig, out_dir / "publication_paper_vs_upair")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "publication_paper_vs_upair.csv", index=False)



def plot_publication_channel_errors(example_npz: str | Path, out_dir: str | Path) -> None:
    example_npz = Path(example_npz)
    out_dir = Path(out_dir)
    if not example_npz.exists():
        return
    data = np.load(example_npz)
    if "h_true" not in data:
        return

    order = [
        ("h_ls_linear", "LS+linear"),
        ("h_ddcpe_ls", "DD-CPE+LS"),
        ("h_paper_cfgres", "Paper cfg reservoir"),
        ("h_prop", "UPAIR-5G"),
    ]
    available = [(key, title) for key, title in order if key in data]
    if not available:
        return

    true_mag = _to_magnitude_panel(data["h_true"])
    mags = [true_mag]
    errs = [np.zeros_like(true_mag)]
    titles_top = ["True |H|"]
    titles_bottom = ["Absolute error"]
    for key, title in available:
        mags.append(_to_magnitude_panel(data[key]))
        errs.append(_to_abs_error_panel(data["h_true"], data[key]))
        titles_top.append(f"{title} |H|")
        titles_bottom.append(f"{title} |H-H_hat|")

    vmag = float(max(np.max(panel) for panel in mags)) if mags else 1.0
    verr = float(max(np.max(panel) for panel in errs[1:])) if len(errs) > 1 else 1.0
    fig, axes = plt.subplots(2, len(mags), figsize=(3.6 * len(mags), 6.2))
    for idx in range(len(mags)):
        im0 = axes[0, idx].imshow(mags[idx], aspect="auto", vmin=0.0, vmax=max(vmag, 1e-6))
        axes[0, idx].set_title(titles_top[idx])
        axes[0, idx].set_xlabel("Subcarrier")
        axes[0, idx].set_ylabel("OFDM symbol")
        im1 = axes[1, idx].imshow(errs[idx], aspect="auto", vmin=0.0, vmax=max(verr, 1e-6))
        axes[1, idx].set_title(titles_bottom[idx])
        axes[1, idx].set_xlabel("Subcarrier")
        axes[1, idx].set_ylabel("OFDM symbol")
    fig.colorbar(im0, ax=axes[0, :].ravel().tolist(), shrink=0.82)
    fig.colorbar(im1, ax=axes[1, :].ravel().tolist(), shrink=0.82)
    _save_figure(fig, out_dir / "publication_channel_error_maps")
    plt.close(fig)



def _selected_ebno_points(values: Iterable[float]) -> list[float]:
    values = sorted(float(v) for v in values)
    preferred = [0.0, 8.0, 16.0]
    selected = [v for v in preferred if v in values]
    if selected:
        return selected
    if len(values) <= 3:
        return values
    return [values[0], values[len(values) // 2], values[-1]]



def plot_phasefair_focus(curves_path: str | Path, out_dir: str | Path) -> None:
    curves_path = Path(curves_path)
    out_dir = Path(out_dir)
    if not curves_path.exists():
        return
    df = pd.read_csv(curves_path)
    focus_receivers = [
        "baseline_ddcpe_ls_lmmse",
        "paper_cfgres_ls_lmmse",
        "paper_cfgres_phaseaware_ls_lmmse",
        "upair5g_lmmse",
        "perfect_csi_lmmse",
    ]
    focus_receivers = [r for r in focus_receivers if r in df["receiver"].unique()]
    if len(focus_receivers) < 2:
        return
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.4), sharex=True)
    for ax, metric in zip(axes, ["ber", "bler", "nmse"]):
        for receiver in focus_receivers:
            sub = df[df["receiver"] == receiver].sort_values("ebno_db")
            if sub.empty or not sub[metric].notna().any():
                continue
            y = np.maximum(sub[metric].to_numpy(dtype=float), 1e-12)
            style = _style(receiver)
            ax.semilogy(sub["ebno_db"], y, marker=style.marker, linewidth=style.linewidth, label=_label(receiver))
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Eb/N0 [dB]")
    axes[0].legend(loc="best", fontsize=9)
    _save_figure(fig, out_dir / "publication_paper_phasefair_focus")
    plt.close(fig)


def plot_error_floor_zoom(curves_path: str | Path, out_dir: str | Path) -> None:
    curves_path = Path(curves_path)
    out_dir = Path(out_dir)
    if not curves_path.exists():
        return
    df = pd.read_csv(curves_path)
    ebnos = sorted(float(v) for v in df["ebno_db"].unique().tolist())
    if len(ebnos) < 3:
        return
    zoom_start = ebnos[len(ebnos) // 2]
    df = df[df["ebno_db"] >= zoom_start].copy()
    focus_receivers = [
        "baseline_ddcpe_ls_lmmse",
        "paper_cfgres_phaseaware_ls_lmmse",
        "upair5g_lmmse",
        "perfect_csi_lmmse",
    ]
    focus_receivers = [r for r in focus_receivers if r in df["receiver"].unique()]
    if len(focus_receivers) < 2:
        return
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.4), sharex=True)
    for ax, metric in zip(axes, ["ber", "bler", "nmse"]):
        for receiver in focus_receivers:
            sub = df[df["receiver"] == receiver].sort_values("ebno_db")
            if sub.empty or not sub[metric].notna().any():
                continue
            y = np.maximum(sub[metric].to_numpy(dtype=float), 1e-12)
            style = _style(receiver)
            ax.semilogy(sub["ebno_db"], y, marker=style.marker, linewidth=style.linewidth, label=_label(receiver))
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Eb/N0 [dB]")
    axes[0].legend(loc="best", fontsize=9)
    _save_figure(fig, out_dir / "publication_error_floor_zoom")
    plt.close(fig)


def write_publication_tables(curves_path: str | Path, out_dir: str | Path) -> None:
    curves_path = Path(curves_path)
    out_dir = Path(out_dir)
    if not curves_path.exists():
        return
    df = pd.read_csv(curves_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_points = _selected_ebno_points(df["ebno_db"].unique().tolist())
    selected_df = df[df["ebno_db"].isin(selected_points)].copy()
    selected_df = selected_df.sort_values(["ebno_db", "receiver"], key=lambda s: s.map({r: i for i, r in enumerate(PLOT_ORDER)}) if s.name == "receiver" else s)
    selected_df.to_csv(out_dir / "publication_selected_points.csv", index=False)

    ranking_rows = []
    for metric in ["ber", "bler", "nmse"]:
        for ebno_db, grp in df.groupby("ebno_db"):
            grp = grp.dropna(subset=[metric]).sort_values(metric)
            for rank, (_, row) in enumerate(grp.iterrows(), start=1):
                ranking_rows.append(
                    {
                        "metric": metric,
                        "ebno_db": float(ebno_db),
                        "rank": rank,
                        "receiver": str(row["receiver"]),
                        "value": float(row[metric]),
                    }
                )
    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(out_dir / "publication_receiver_ranking.csv", index=False)

    latex_lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Receiver @ Eb/N0 & BER & BLER & NMSE \\\\",
        "\\midrule",
    ]
    for ebno_db in selected_points:
        grp = selected_df[selected_df["ebno_db"] == ebno_db].copy()
        for _, row in grp.iterrows():
            latex_lines.append(
                f"{_label(str(row['receiver']))} @ {ebno_db:.0f} dB & {float(row['ber']):.3e} & {float(row['bler']):.3e} & {float(row['nmse']):.3e} \\\\"
            )
        latex_lines.append("\\midrule")
    if latex_lines[-1] == "\\midrule":
        latex_lines = latex_lines[:-1]
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (out_dir / "publication_selected_points.tex").write_text("\n".join(latex_lines), encoding="utf-8")



def make_all_plots(cfg: dict) -> dict[str, str]:
    paths = ensure_output_tree(cfg)
    plot_training_history(paths["metrics"] / "history.json", paths["plots"])
    plot_curves(paths["metrics"] / "curves.csv", paths["plots"])
    plot_channel_example(paths["artifacts"] / "channel_example.npz", paths["plots"])
    plot_publication_main_curves(paths["metrics"] / "curves.csv", paths["plots"])
    plot_upair_gains(paths["metrics"] / "curves.csv", paths["plots"])
    plot_paper_vs_upair(paths["metrics"] / "curves.csv", paths["plots"])
    plot_phasefair_focus(paths["metrics"] / "curves.csv", paths["plots"])
    plot_error_floor_zoom(paths["metrics"] / "curves.csv", paths["plots"])
    plot_publication_channel_errors(paths["artifacts"] / "channel_example.npz", paths["plots"])
    write_publication_tables(paths["metrics"] / "curves.csv", paths["plots"])
    return {"plots_dir": str(paths["plots"])}
