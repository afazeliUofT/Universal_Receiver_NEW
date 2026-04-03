#!/usr/bin/env python3
"""Probe the actual modulation and draw one-PRB transmit grids.

Designed for the Universal_Receiver_NEW repo on Narval.

Usage:
  source /home/rsadve1/PROJECT/Universal_Receiver_NEW/venv_universal_receiver/bin/activate
  cd /home/rsadve1/PROJECT/Universal_Receiver_NEW
  python /mnt/data/probe_modulation_prb_grid.py \
      --repo-root /home/rsadve1/PROJECT/Universal_Receiver_NEW

It will:
  1) build the exact PUSCH transmitter used by the repo,
  2) print the effective mcs_table, mcs_index, num_bits_per_symbol, coderate,
  3) save one-PRB resource-grid figures for twc_mild_clean and twc_mild_dmrsrich,
  4) save constellation scatter plots of all transmitted data REs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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


def fmt_c(z: complex, digits: int = 2) -> str:
    a = float(np.real(z))
    b = float(np.imag(z))
    if abs(a) < 0.5 * 10 ** (-digits):
        a = 0.0
    if abs(b) < 0.5 * 10 ** (-digits):
        b = 0.0
    sign = "+" if b >= 0 else "-"
    return f"{a:.{digits}f}{sign}j{abs(b):.{digits}f}"


def load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_repo_to_path(repo_root: Path) -> None:
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_builder(repo_root: Path):
    add_repo_to_path(repo_root)
    mod = importlib.import_module("upair5g.builders")
    return mod.build_pusch_transmitter


def extract_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def save_constellation(data_vals: np.ndarray, out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(6.2, 6.0))
    ax = fig.add_subplot(111)
    ax.scatter(np.real(data_vals), np.imag(data_vals), s=14, alpha=0.65)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_prb_grid(
    x_grid_sf: np.ndarray,
    dmrs_mask_sf: np.ndarray,
    dmrs_grid_sf: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """All arrays are [12, 14] with axes [subcarrier, symbol]."""
    fig = plt.figure(figsize=(13.8, 7.3))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(14) + 0.5)
    ax.set_xticklabels([str(i) for i in range(14)], fontsize=10)
    ax.set_yticks(np.arange(12) + 0.5)
    ax.set_yticklabels([str(i) for i in range(12)], fontsize=10)
    ax.set_xlabel("OFDM symbol index s")
    ax.set_ylabel("Subcarrier index f within first PRB")
    ax.set_title(title)

    for f in range(12):
        for s in range(14):
            x = float(s)
            y = float(f)
            is_dmrs = abs(dmrs_grid_sf[f, s]) > 1e-12
            is_mask = bool(dmrs_mask_sf[f, s])
            is_mask_zero = is_mask and not is_dmrs
            if is_dmrs:
                face = "#4F8BFF"  # DMRS
                edge = "#1F4FB2"
                label = f"DMRS\n{fmt_c(dmrs_grid_sf[f, s])}"
                text_color = "white"
            elif is_mask_zero:
                face = "#D5F5F6"  # masked but zero for this port
                edge = "#4AA7A9"
                label = "mask\n0"
                text_color = "#205A5C"
            else:
                face = "#FDE7C8"  # data
                edge = "#A27A2C"
                label = f"data\n{fmt_c(x_grid_sf[f, s])}"
                text_color = "#5F4516"

            rect = Rectangle((x, y), 1, 1, facecolor=face, edgecolor=edge, linewidth=0.9)
            ax.add_patch(rect)
            ax.text(x + 0.5, y + 0.52, label, ha="center", va="center", fontsize=7.3, color=text_color)

    # Grid lines
    for s in range(15):
        ax.plot([s, s], [0, 12], color="0.75", linewidth=0.45)
    for f in range(13):
        ax.plot([0, 14], [f, f], color="0.75", linewidth=0.45)

    # Legend note
    note = (
        "Blue: nonzero DMRS symbols actually transmitted on this port.  "
        "Cyan: REs masked from data but zero on this port.  "
        "Orange: data REs carrying the selected QAM constellation."
    )
    ax.text(0.0, 12.75, note, fontsize=9, ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_case(repo_root: Path, config_rel: str, out_dir: Path) -> dict:
    build_pusch_transmitter = load_builder(repo_root)
    cfg_path = repo_root / config_rel
    cfg = load_cfg(cfg_path)

    tx, pc = build_pusch_transmitter(cfg)
    x, b = tx(1)

    # Effective transport-block parameters after repo + Sionna handling
    tb = pc.tb
    mcs_index = int(tb.mcs_index)
    mcs_table_effective = int(tb.mcs_table)
    num_bits_per_symbol = int(tb.num_bits_per_symbol)
    target_coderate = float(tb.target_coderate)

    # Public docs: x is [batch, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
    x = extract_numpy(x)
    x_grid = x[0, 0, 0]  # [14, 192]
    x_grid_sf = x_grid.T[:12, :]  # first PRB, [12, 14]

    dmrs_mask_sf = np.asarray(pc.dmrs_mask[:12, :], dtype=bool)
    dmrs_grid_sf = np.asarray(pc.dmrs_grid[0, :12, :])
    is_data = ~dmrs_mask_sf
    data_vals = x_grid_sf[is_data]

    stem = Path(config_rel).stem
    save_prb_grid(
        x_grid_sf=x_grid_sf,
        dmrs_mask_sf=dmrs_mask_sf,
        dmrs_grid_sf=dmrs_grid_sf,
        out_path=out_dir / f"{stem}_prb0_grid.png",
        title=(
            f"{stem}: first PRB transmit grid | effective modulation = {qam_name(num_bits_per_symbol)} "
            f"(Qm={num_bits_per_symbol}), DMRS symbols = {list(pc.dmrs_symbol_indices)}"
        ),
    )
    save_constellation(
        data_vals=data_vals,
        out_path=out_dir / f"{stem}_data_constellation.png",
        title=f"{stem}: all data RE symbols in one slot | {qam_name(num_bits_per_symbol)} (Qm={num_bits_per_symbol})",
    )

    summary = {
        "config": str(cfg_path),
        "mcs_index_from_yaml": cfg["pusch"]["mcs_index"],
        "mcs_table_from_yaml": cfg["pusch"]["mcs_table"],
        "effective_tb_mcs_index": mcs_index,
        "effective_tb_mcs_table": mcs_table_effective,
        "effective_num_bits_per_symbol": num_bits_per_symbol,
        "effective_modulation_name": qam_name(num_bits_per_symbol),
        "effective_target_coderate": target_coderate,
        "num_resource_blocks": int(pc.num_resource_blocks),
        "num_subcarriers": int(pc.num_subcarriers),
        "dmrs_symbol_indices": [int(v) for v in pc.dmrs_symbol_indices],
        "prb0_grid_png": str(out_dir / f"{stem}_prb0_grid.png"),
        "constellation_png": str(out_dir / f"{stem}_data_constellation.png"),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/home/rsadve1/PROJECT/Universal_Receiver_NEW"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir or (repo_root / "probe_outputs" / "modulation_prb_grid")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import sionna
        sionna_version = getattr(sionna, "__version__", "unknown")
    except Exception:
        sionna_version = "import_failed"

    cases = [
        "configs/twc_mild_clean.yaml",
        "configs/twc_mild_dmrsrich.yaml",
    ]

    all_summaries = {
        "repo_root": str(repo_root),
        "sionna_version": sionna_version,
        "cases": [],
    }

    for case in cases:
        summary = run_case(repo_root, case, out_dir)
        all_summaries["cases"].append(summary)

    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("=" * 88)
    print(f"Repo root       : {repo_root}")
    print(f"Sionna version  : {sionna_version}")
    print(f"Output dir      : {out_dir}")
    print("=" * 88)
    for case in all_summaries["cases"]:
        print(f"Config                 : {case['config']}")
        print(f"YAML mcs_table         : {case['mcs_table_from_yaml']}")
        print(f"YAML mcs_index         : {case['mcs_index_from_yaml']}")
        print(f"Effective tb.mcs_table : {case['effective_tb_mcs_table']}")
        print(f"Effective tb.mcs_index : {case['effective_tb_mcs_index']}")
        print(f"Qm                     : {case['effective_num_bits_per_symbol']}")
        print(f"Modulation             : {case['effective_modulation_name']}")
        print(f"Target coderate        : {case['effective_target_coderate']:.6f}")
        print(f"DMRS symbols           : {case['dmrs_symbol_indices']}")
        print(f"PRB grid PNG           : {case['prb0_grid_png']}")
        print(f"Constellation PNG      : {case['constellation_png']}")
        print("-" * 88)

    print(f"JSON summary saved to  : {json_path}")


if __name__ == "__main__":
    main()
