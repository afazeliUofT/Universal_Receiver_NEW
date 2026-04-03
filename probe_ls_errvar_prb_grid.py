#!/usr/bin/env python3
"""Probe LS estimates and LS error variances on the first PRB.

Designed for the Universal_Receiver_NEW repo on Narval.

This script follows the repo's own runtime path:
  transmitter -> Eb/N0 to No -> channel -> optional symbol-wise phase impairment -> LS estimator

It saves annotated one-PRB figures for:
  - transmitted slot grid (same style as the modulation probe)
  - LS channel estimate on every RE for one selected RX antenna
  - LS error variance on every RE for one selected RX antenna
  - antenna-averaged LS error variance (the quantity used in the feature tensor F)
  - CSV / NPZ dumps with exact numeric values per RE

Usage:
  source /home/rsadve1/PROJECT/Universal_Receiver_NEW/venv_universal_receiver/bin/activate
  cd /home/rsadve1/PROJECT/Universal_Receiver_NEW
  python /mnt/data/probe_ls_errvar_prb_grid.py \
      --repo-root /home/rsadve1/PROJECT/Universal_Receiver_NEW

Optional:
  python /mnt/data/probe_ls_errvar_prb_grid.py \
      --repo-root /home/rsadve1/PROJECT/Universal_Receiver_NEW \
      --ebno-db 7 \
      --rx-ant 0
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ------------------------------
# Small formatting helpers
# ------------------------------
def fmt_c(z: complex, digits: int = 2) -> str:
    a = float(np.real(z))
    b = float(np.imag(z))
    if abs(a) < 0.5 * 10 ** (-digits):
        a = 0.0
    if abs(b) < 0.5 * 10 ** (-digits):
        b = 0.0
    sign = "+" if b >= 0 else "-"
    return f"{a:.{digits}f}{sign}j{abs(b):.{digits}f}"


def fmt_r(x: float, digits: int = 4) -> str:
    x = float(x)
    ax = abs(x)
    if ax == 0.0:
        return "0"
    if ax >= 1000 or ax < 1e-3:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def qam_name(num_bits_per_symbol: int) -> str:
    mapping = {
        1: "pi/2-BPSK or BPSK-like",
        2: "QPSK",
        4: "16-QAM",
        6: "64-QAM",
        8: "256-QAM",
        10: "1024-QAM",
    }
    return mapping.get(int(num_bits_per_symbol), f"unknown({num_bits_per_symbol})")


# ------------------------------
# Repo import helpers
# ------------------------------
def add_repo_to_path(repo_root: Path) -> None:
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_cfg(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def load_repo_modules(repo_root: Path) -> dict[str, Any]:
    add_repo_to_path(repo_root)
    builders = importlib.import_module("upair5g.builders")
    utils = importlib.import_module("upair5g.utils")
    impairments = importlib.import_module("upair5g.impairments")
    return {
        "build_pusch_transmitter": builders.build_pusch_transmitter,
        "build_channel": builders.build_channel,
        "build_ls_estimator": builders.build_ls_estimator,
        "get_resource_grid": builders.get_resource_grid,
        "extract_pilot_mask": builders.extract_pilot_mask,
        "call_transmitter": utils.call_transmitter,
        "call_channel": utils.call_channel,
        "ebno_db_to_no": utils.ebno_db_to_no,
        "tensor7_to_btfnc": utils.tensor7_to_btfnc,
        "y_to_btfnc": utils.y_to_btfnc,
        "set_global_seed": utils.set_global_seed,
        "safe_call_variants": importlib.import_module("upair5g.compat").safe_call_variants,
        "apply_symbol_phase_impairment": impairments.apply_symbol_phase_impairment,
    }


# ------------------------------
# Plotting helpers
# ------------------------------
def category_colors(cell_kind: str) -> tuple[str, str, str]:
    if cell_kind == "dmrs_nonzero":
        return "#4F8BFF", "#1F4FB2", "white"
    if cell_kind == "masked_zero":
        return "#D5F5F6", "#4AA7A9", "#205A5C"
    return "#FDE7C8", "#A27A2C", "#5F4516"


def classify_cell(mask_val: bool, dmrs_nonzero: bool) -> str:
    if dmrs_nonzero:
        return "dmrs_nonzero"
    if mask_val:
        return "masked_zero"
    return "data"


def plot_annotated_grid(
    values_sf: np.ndarray,
    mask_sf: np.ndarray,
    dmrs_nonzero_sf: np.ndarray,
    out_path: Path,
    title: str,
    value_kind: str,
    digits: int = 2,
    subtitle: str | None = None,
) -> None:
    """Plot one PRB grid with per-cell annotations.

    Args:
        values_sf: [12, T] values, complex or real.
        mask_sf: [12, T] bool mask used in F.
        dmrs_nonzero_sf: [12, T] bool of nonzero transmitted DMRS on the selected port.
        value_kind: one of {"complex", "real"}
    """
    num_sc, num_sym = values_sf.shape
    fig = plt.figure(figsize=(max(14.5, 0.95 * num_sym), 8.6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, num_sym)
    ax.set_ylim(0, num_sc)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(num_sym) + 0.5)
    ax.set_xticklabels([str(i) for i in range(num_sym)], fontsize=10)
    ax.set_yticks(np.arange(num_sc) + 0.5)
    ax.set_yticklabels([str(i) for i in range(num_sc)], fontsize=10)
    ax.set_xlabel("OFDM symbol index s")
    ax.set_ylabel("Subcarrier index f within first PRB")
    ax.set_title(title)

    for f in range(num_sc):
        for s in range(num_sym):
            kind = classify_cell(bool(mask_sf[f, s]), bool(dmrs_nonzero_sf[f, s]))
            face, edge, text_color = category_colors(kind)
            rect = Rectangle((float(s), float(f)), 1, 1, facecolor=face, edgecolor=edge, linewidth=0.9)
            ax.add_patch(rect)

            if value_kind == "complex":
                value_text = fmt_c(complex(values_sf[f, s]), digits=digits)
            elif value_kind == "real":
                value_text = fmt_r(float(values_sf[f, s]), digits=digits)
            else:
                raise ValueError(f"Unsupported value_kind: {value_kind}")

            if kind == "dmrs_nonzero":
                top = "DMRS"
            elif kind == "masked_zero":
                top = "mask"
            else:
                top = "data"

            ax.text(
                s + 0.5,
                f + 0.53,
                f"{top}\n{value_text}",
                ha="center",
                va="center",
                fontsize=7.0,
                color=text_color,
            )

    for s in range(num_sym + 1):
        ax.plot([s, s], [0, num_sc], color="0.75", linewidth=0.45)
    for f in range(num_sc + 1):
        ax.plot([0, num_sym], [f, f], color="0.75", linewidth=0.45)

    note = (
        "Blue: nonzero DMRS on the selected port.  "
        "Cyan: REs masked in F but zero on this port.  "
        "Orange: data REs."
    )
    ax.text(0.0, num_sc + 0.62, note, fontsize=9, ha="left", va="bottom")
    if subtitle:
        ax.text(0.0, num_sc + 1.02, subtitle, fontsize=9, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overview_panel(
    tx_sf: np.ndarray,
    hls_sf: np.ndarray,
    err_sf: np.ndarray,
    mask_sf: np.ndarray,
    dmrs_nonzero_sf: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    num_sc, num_sym = tx_sf.shape
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 7.4), constrained_layout=True)
    panels = [
        (tx_sf, "complex", "Transmitted x on first PRB"),
        (hls_sf, "complex", r"LS estimate $\hat h^{\mathrm{LS}}$ on first PRB"),
        (err_sf, "real", r"LS error variance $V^{\mathrm{LS}}$ on first PRB"),
    ]

    for ax, (values_sf, value_kind, panel_title) in zip(axes, panels):
        ax.set_xlim(0, num_sym)
        ax.set_ylim(0, num_sc)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(num_sym) + 0.5)
        ax.set_xticklabels([str(i) for i in range(num_sym)], fontsize=8)
        ax.set_yticks(np.arange(num_sc) + 0.5)
        ax.set_yticklabels([str(i) for i in range(num_sc)], fontsize=8)
        ax.set_xlabel("s")
        ax.set_ylabel("f")
        ax.set_title(panel_title, fontsize=10)

        for f in range(num_sc):
            for s in range(num_sym):
                kind = classify_cell(bool(mask_sf[f, s]), bool(dmrs_nonzero_sf[f, s]))
                face, edge, text_color = category_colors(kind)
                rect = Rectangle((float(s), float(f)), 1, 1, facecolor=face, edgecolor=edge, linewidth=0.7)
                ax.add_patch(rect)
                if value_kind == "complex":
                    text = fmt_c(complex(values_sf[f, s]), digits=1)
                else:
                    text = fmt_r(float(values_sf[f, s]), digits=3)
                ax.text(s + 0.5, f + 0.52, text, ha="center", va="center", fontsize=5.6, color=text_color)

        for s in range(num_sym + 1):
            ax.plot([s, s], [0, num_sc], color="0.78", linewidth=0.35)
        for f in range(num_sc + 1):
            ax.plot([0, num_sym], [f, f], color="0.78", linewidth=0.35)

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------
# Data extraction helpers
# ------------------------------
def resolve_eval_ebno_from_cfg(cfg: dict[str, Any]) -> float:
    grid = list(cfg.get("system", {}).get("ebno_db_eval", [7.0]))
    if not grid:
        return 7.0
    return float(grid[min(len(grid) // 2, len(grid) - 1)])


def infer_ls_output(ls_out: Any) -> tuple[Any, Any]:
    if isinstance(ls_out, (tuple, list)) and len(ls_out) >= 2:
        return ls_out[0], ls_out[1]
    raise ValueError("LS estimator output does not contain (h_hat, err_var).")


def get_prb0_slices(
    x: np.ndarray,
    y_btfnc: np.ndarray,
    h_btfnc: np.ndarray,
    h_ls_btfnc: np.ndarray,
    err_btfnc: np.ndarray,
    mask_tf: np.ndarray,
    dmrs_grid: np.ndarray,
    rx_ant: int,
) -> dict[str, np.ndarray]:
    """Return first-PRB [12, T] slices with axes [f, s]."""
    tx_sf = x[0, 0, 0].T[:12, :]                  # [12, T]
    y_sf = y_btfnc[0, :, :12, rx_ant].T          # [12, T]
    h_true_sf = h_btfnc[0, :, :12, rx_ant].T     # [12, T]
    h_ls_sf = h_ls_btfnc[0, :, :12, rx_ant].T    # [12, T]
    err_sf = err_btfnc[0, :, :12, rx_ant].T      # [12, T]
    err_mean_sf = np.mean(err_btfnc[0, :, :12, :], axis=-1).T  # [12, T]
    mask_sf = mask_tf[:, :12, 0].T.astype(bool)  # [12, T]

    dmrs_grid = np.asarray(dmrs_grid)
    if dmrs_grid.ndim == 3:
        dmrs_nonzero_sf = np.abs(dmrs_grid[0, :12, :]) > 1e-12
    elif dmrs_grid.ndim == 2:
        dmrs_nonzero_sf = np.abs(dmrs_grid[:12, :]) > 1e-12
    else:
        raise ValueError(f"Unexpected DMRS grid rank: {dmrs_grid.ndim}")

    return {
        "tx_sf": tx_sf,
        "y_sf": y_sf,
        "h_true_sf": h_true_sf,
        "h_ls_sf": h_ls_sf,
        "err_sf": err_sf,
        "err_mean_sf": err_mean_sf,
        "mask_sf": mask_sf,
        "dmrs_nonzero_sf": dmrs_nonzero_sf,
    }


def save_re_csv(
    out_path: Path,
    case_name: str,
    ebno_db: float,
    no: float,
    rx_ant: int,
    prb: dict[str, np.ndarray],
) -> None:
    fields = [
        "case",
        "ebno_db",
        "noise_variance_no",
        "rx_ant",
        "subcarrier_f",
        "symbol_s",
        "pilot_mask_F",
        "nonzero_dmrs_on_selected_port",
        "tx_real",
        "tx_imag",
        "y_real",
        "y_imag",
        "h_true_real",
        "h_true_imag",
        "h_ls_real",
        "h_ls_imag",
        "err_var_rx_ant",
        "err_var_mean_over_rx",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for f_idx in range(prb["tx_sf"].shape[0]):
            for s_idx in range(prb["tx_sf"].shape[1]):
                tx_val = complex(prb["tx_sf"][f_idx, s_idx])
                y_val = complex(prb["y_sf"][f_idx, s_idx])
                h_true_val = complex(prb["h_true_sf"][f_idx, s_idx])
                h_ls_val = complex(prb["h_ls_sf"][f_idx, s_idx])
                writer.writerow({
                    "case": case_name,
                    "ebno_db": ebno_db,
                    "noise_variance_no": no,
                    "rx_ant": rx_ant,
                    "subcarrier_f": f_idx,
                    "symbol_s": s_idx,
                    "pilot_mask_F": int(bool(prb["mask_sf"][f_idx, s_idx])),
                    "nonzero_dmrs_on_selected_port": int(bool(prb["dmrs_nonzero_sf"][f_idx, s_idx])),
                    "tx_real": float(np.real(tx_val)),
                    "tx_imag": float(np.imag(tx_val)),
                    "y_real": float(np.real(y_val)),
                    "y_imag": float(np.imag(y_val)),
                    "h_true_real": float(np.real(h_true_val)),
                    "h_true_imag": float(np.imag(h_true_val)),
                    "h_ls_real": float(np.real(h_ls_val)),
                    "h_ls_imag": float(np.imag(h_ls_val)),
                    "err_var_rx_ant": float(prb["err_sf"][f_idx, s_idx]),
                    "err_var_mean_over_rx": float(prb["err_mean_sf"][f_idx, s_idx]),
                })


# ------------------------------
# Main case runner
# ------------------------------
def run_case(
    repo_root: Path,
    config_rel: str,
    out_dir: Path,
    rx_ant: int,
    ebno_db_override: float | None,
) -> dict[str, Any]:
    mods = load_repo_modules(repo_root)
    cfg_path = repo_root / config_rel
    cfg = load_cfg(cfg_path)

    # Reproduce the repo's own deterministic seed behavior.
    mods["set_global_seed"](int(cfg["system"]["seed"]))

    tx, pc = mods["build_pusch_transmitter"](cfg)
    channel = mods["build_channel"](cfg, tx)
    ls_estimator = mods["build_ls_estimator"](tx, cfg)
    resource_grid = mods["get_resource_grid"](tx)

    ebno_db = float(resolve_eval_ebno_from_cfg(cfg) if ebno_db_override is None else ebno_db_override)
    no = mods["ebno_db_to_no"](ebno_db, tx=tx, resource_grid=resource_grid)

    x, bits = mods["call_transmitter"](tx, 1)
    y, h = mods["call_channel"](channel, x, no)
    y, h = mods["apply_symbol_phase_impairment"](y, h, cfg, training=False)

    ls_out = mods["safe_call_variants"](ls_estimator, y, no)
    h_ls, err_var = infer_ls_output(ls_out)

    # Convert to repo's B-T-F-Nr view used before feature stacking.
    y_btfnc = extract_numpy(mods["y_to_btfnc"](y))
    h_btfnc = extract_numpy(mods["tensor7_to_btfnc"](h))
    h_ls_btfnc = extract_numpy(mods["tensor7_to_btfnc"](h_ls))
    err_btfnc = extract_numpy(mods["tensor7_to_btfnc"](err_var))
    mask_tf = extract_numpy(mods["extract_pilot_mask"](resource_grid))
    x_np = extract_numpy(x)

    num_rx_ant = h_ls_btfnc.shape[-1]
    if not (0 <= rx_ant < num_rx_ant):
        raise ValueError(f"rx_ant must be in [0, {num_rx_ant-1}], got {rx_ant}")

    prb = get_prb0_slices(
        x=x_np,
        y_btfnc=y_btfnc,
        h_btfnc=h_btfnc,
        h_ls_btfnc=h_ls_btfnc,
        err_btfnc=err_btfnc,
        mask_tf=mask_tf,
        dmrs_grid=getattr(pc, "dmrs_grid"),
        rx_ant=rx_ant,
    )

    stem = Path(config_rel).stem
    qmbits = int(getattr(pc.tb, "num_bits_per_symbol"))
    dmrs_symbols = [int(v) for v in list(getattr(pc, "dmrs_symbol_indices"))]
    no_scalar = float(np.asarray(extract_numpy(no)).reshape(-1)[0])

    common_subtitle = (
        f"Eb/N0={ebno_db:.2f} dB, No={fmt_r(no_scalar, digits=5)}, "
        f"effective modulation={qam_name(qmbits)} (Qm={qmbits}), DMRS symbols={dmrs_symbols}, RX antenna={rx_ant}"
    )

    # Same transmit grid style as the earlier modulation probe.
    plot_annotated_grid(
        values_sf=prb["tx_sf"],
        mask_sf=prb["mask_sf"],
        dmrs_nonzero_sf=prb["dmrs_nonzero_sf"],
        out_path=out_dir / f"{stem}_prb0_tx_grid.png",
        title=f"{stem}: transmitted symbols on first PRB",
        value_kind="complex",
        digits=2,
        subtitle=common_subtitle,
    )

    plot_annotated_grid(
        values_sf=prb["h_ls_sf"],
        mask_sf=prb["mask_sf"],
        dmrs_nonzero_sf=prb["dmrs_nonzero_sf"],
        out_path=out_dir / f"{stem}_prb0_hls_rx{rx_ant}.png",
        title=f"{stem}: LS channel estimate on first PRB (RX antenna {rx_ant})",
        value_kind="complex",
        digits=2,
        subtitle=common_subtitle,
    )

    plot_annotated_grid(
        values_sf=prb["err_sf"],
        mask_sf=prb["mask_sf"],
        dmrs_nonzero_sf=prb["dmrs_nonzero_sf"],
        out_path=out_dir / f"{stem}_prb0_errvar_rx{rx_ant}.png",
        title=f"{stem}: LS error variance on first PRB (RX antenna {rx_ant})",
        value_kind="real",
        digits=4,
        subtitle=common_subtitle,
    )

    plot_annotated_grid(
        values_sf=prb["err_mean_sf"],
        mask_sf=prb["mask_sf"],
        dmrs_nonzero_sf=prb["dmrs_nonzero_sf"],
        out_path=out_dir / f"{stem}_prb0_errvar_mean.png",
        title=f"{stem}: antenna-averaged LS error variance on first PRB (matches F variance feature)",
        value_kind="real",
        digits=4,
        subtitle=common_subtitle,
    )

    plot_overview_panel(
        tx_sf=prb["tx_sf"],
        hls_sf=prb["h_ls_sf"],
        err_sf=prb["err_sf"],
        mask_sf=prb["mask_sf"],
        dmrs_nonzero_sf=prb["dmrs_nonzero_sf"],
        out_path=out_dir / f"{stem}_prb0_overview_rx{rx_ant}.png",
        title=f"{stem}: transmitted symbols, LS estimate, and LS error variance on first PRB",
    )

    npz_path = out_dir / f"{stem}_prb0_values_rx{rx_ant}.npz"
    np.savez_compressed(
        npz_path,
        ebno_db=np.array([ebno_db], dtype=np.float32),
        noise_variance_no=np.array([no_scalar], dtype=np.float32),
        tx_sf=prb["tx_sf"],
        y_sf=prb["y_sf"],
        h_true_sf=prb["h_true_sf"],
        h_ls_sf=prb["h_ls_sf"],
        err_sf=prb["err_sf"],
        err_mean_sf=prb["err_mean_sf"],
        mask_sf=prb["mask_sf"].astype(np.int8),
        dmrs_nonzero_sf=prb["dmrs_nonzero_sf"].astype(np.int8),
        dmrs_symbol_indices=np.array(dmrs_symbols, dtype=np.int32),
    )

    csv_path = out_dir / f"{stem}_prb0_values_rx{rx_ant}.csv"
    save_re_csv(
        out_path=csv_path,
        case_name=stem,
        ebno_db=ebno_db,
        no=no_scalar,
        rx_ant=rx_ant,
        prb=prb,
    )

    summary = {
        "config": str(cfg_path),
        "ebno_db": ebno_db,
        "noise_variance_no": no_scalar,
        "rx_ant": rx_ant,
        "effective_tb_mcs_index": int(pc.tb.mcs_index),
        "effective_tb_mcs_table": int(pc.tb.mcs_table),
        "effective_num_bits_per_symbol": qmbits,
        "effective_modulation_name": qam_name(qmbits),
        "effective_target_coderate": float(pc.tb.target_coderate),
        "num_resource_blocks": int(pc.num_resource_blocks),
        "num_subcarriers": int(pc.num_subcarriers),
        "dmrs_symbol_indices": dmrs_symbols,
        "tx_grid_png": str(out_dir / f"{stem}_prb0_tx_grid.png"),
        "hls_grid_png": str(out_dir / f"{stem}_prb0_hls_rx{rx_ant}.png"),
        "errvar_grid_png": str(out_dir / f"{stem}_prb0_errvar_rx{rx_ant}.png"),
        "errvar_mean_grid_png": str(out_dir / f"{stem}_prb0_errvar_mean.png"),
        "overview_png": str(out_dir / f"{stem}_prb0_overview_rx{rx_ant}.png"),
        "values_npz": str(npz_path),
        "values_csv": str(csv_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/home/rsadve1/PROJECT/Universal_Receiver_NEW"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rx-ant", type=int, default=0, help="RX antenna index to visualize for h_ls and err_var.")
    parser.add_argument(
        "--ebno-db",
        type=float,
        default=None,
        help="If omitted, use the same midpoint evaluation Eb/N0 choice as the repo validation path.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["configs/twc_mild_clean.yaml", "configs/twc_mild_dmrsrich.yaml"],
        help="Config files relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir or (repo_root / "probe_outputs" / "modulation_prb_grid")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import sionna
        sionna_version = getattr(sionna, "__version__", "unknown")
    except Exception:
        sionna_version = "import_failed"

    all_summaries: dict[str, Any] = {
        "repo_root": str(repo_root),
        "out_dir": str(out_dir),
        "sionna_version": sionna_version,
        "rx_ant": args.rx_ant,
        "ebno_db_override": args.ebno_db,
        "cases": [],
    }

    for case in args.cases:
        summary = run_case(
            repo_root=repo_root,
            config_rel=case,
            out_dir=out_dir,
            rx_ant=args.rx_ant,
            ebno_db_override=args.ebno_db,
        )
        all_summaries["cases"].append(summary)

    json_path = out_dir / "ls_errvar_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("=" * 92)
    print(f"Repo root          : {repo_root}")
    print(f"Output dir         : {out_dir}")
    print(f"Sionna version     : {sionna_version}")
    print(f"RX antenna shown   : {args.rx_ant}")
    if args.ebno_db is None:
        print("Eb/N0 source       : midpoint of each config's eval grid (same logic as repo validation)")
    else:
        print(f"Eb/N0 source       : command-line override = {args.ebno_db:.2f} dB")
    print("=" * 92)
    for case in all_summaries["cases"]:
        print(f"Config             : {case['config']}")
        print(f"Eb/N0              : {case['ebno_db']:.2f} dB")
        print(f"No                 : {case['noise_variance_no']:.6g}")
        print(f"Qm                 : {case['effective_num_bits_per_symbol']}")
        print(f"Modulation         : {case['effective_modulation_name']}")
        print(f"DMRS symbols       : {case['dmrs_symbol_indices']}")
        print(f"TX grid PNG        : {case['tx_grid_png']}")
        print(f"H_LS grid PNG      : {case['hls_grid_png']}")
        print(f"ErrVar grid PNG    : {case['errvar_grid_png']}")
        print(f"ErrVar mean PNG    : {case['errvar_mean_grid_png']}")
        print(f"Overview PNG       : {case['overview_png']}")
        print(f"Values CSV         : {case['values_csv']}")
        print(f"Values NPZ         : {case['values_npz']}")
        print("-" * 92)
    print(f"Summary JSON       : {json_path}")


if __name__ == "__main__":
    main()
