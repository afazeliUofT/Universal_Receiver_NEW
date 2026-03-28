from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upair5g.config import get_cfg, load_config, set_cfg  # noqa: E402
from upair5g.evaluation import evaluate_model  # noqa: E402

KEEP_RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]
ADDED_RECEIVER = "baseline_ls_2dlmmse_lmmse"
FINAL_RECEIVER_ORDER = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)



def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            if column in {"reliable_ber", "reliable_bler"}:
                result[column] = False
            else:
                result[column] = np.nan
    return result



def _scenario_existing_curves_path(cfg: dict) -> Path:
    output_root = Path(str(get_cfg(cfg, "experiment.output_root", "outputs")))
    scenario_name = str(get_cfg(cfg, "experiment.name", "experiment"))
    return PROJECT_ROOT / output_root / scenario_name / "metrics" / "curves.csv"



def _load_preserved_rows(cfg: dict) -> tuple[str, pd.DataFrame, Path]:
    scenario_name = str(get_cfg(cfg, "experiment.name", "experiment"))
    curves_path = _scenario_existing_curves_path(cfg)
    df = _read_csv(curves_path)
    found = set(df["receiver"].dropna().astype(str).unique().tolist())
    missing = [receiver for receiver in KEEP_RECEIVERS if receiver not in found]
    if missing:
        raise RuntimeError(
            "Existing curves are missing required preserved receivers for "
            f"{scenario_name}: {missing}\nChecked file: {curves_path}"
        )
    df = df[df["receiver"].isin(KEEP_RECEIVERS)].copy()
    df["scenario"] = scenario_name
    return scenario_name, df, curves_path



def _evaluate_only_2dlmmse(cfg: dict, scenario_name: str) -> pd.DataFrame:
    cfg_eval = copy.deepcopy(cfg)
    set_cfg(cfg_eval, "experiment.output_root", "TWC_plots2/_tmp_eval")
    set_cfg(cfg_eval, "experiment.name", scenario_name)
    set_cfg(cfg_eval, "baselines.enabled_receivers", [ADDED_RECEIVER])
    set_cfg(cfg_eval, "evaluation.stopping_receivers", [ADDED_RECEIVER])
    set_cfg(cfg_eval, "evaluation.save_example_batch", False)

    result = evaluate_model(cfg_eval, checkpoint_path=None)
    curves_path = Path(result["curves_path"])
    df = _read_csv(curves_path)
    df = df[df["receiver"] == ADDED_RECEIVER].copy()
    if df.empty:
        raise RuntimeError(f"2D-LMMSE evaluation produced no rows for {scenario_name}")
    df["scenario"] = scenario_name

    tmp_root = PROJECT_ROOT / "TWC_plots2" / "_tmp_eval" / scenario_name
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    parent_tmp = PROJECT_ROOT / "TWC_plots2" / "_tmp_eval"
    if parent_tmp.exists() and not any(parent_tmp.iterdir()):
        parent_tmp.rmdir()
    return df



def _merge_rows(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    columns = list(existing_df.columns)
    for column in new_df.columns:
        if column not in columns:
            columns.append(column)

    existing_df = _ensure_columns(existing_df, columns)
    new_df = _ensure_columns(new_df, columns)

    merged = pd.concat([existing_df[columns], new_df[columns]], ignore_index=True)
    order_map = {name: idx for idx, name in enumerate(FINAL_RECEIVER_ORDER)}
    merged["_receiver_order"] = merged["receiver"].map(lambda x: order_map.get(str(x), len(order_map)))
    merged = merged.sort_values(["_receiver_order", "ebno_db"], kind="mergesort").drop(columns=["_receiver_order"])
    return merged.reset_index(drop=True)



def main() -> None:
    parser = argparse.ArgumentParser(description="Preserve existing TWC curves and add only LS+2D-LMMSE results.")
    parser.add_argument("--config", required=True, help="Path to the scenario YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scenario_name, existing_df, existing_curves_path = _load_preserved_rows(cfg)
    new_df = _evaluate_only_2dlmmse(cfg, scenario_name)
    merged_df = _merge_rows(existing_df, new_df)

    out_dir = PROJECT_ROOT / "TWC_plots2" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{scenario_name}_curves.csv"
    merged_df.to_csv(out_csv, index=False)

    manifest = {
        "scenario": scenario_name,
        "preserved_receivers": KEEP_RECEIVERS,
        "added_receiver": ADDED_RECEIVER,
        "source_existing_curves": str(existing_curves_path),
        "output_csv": str(out_csv),
        "num_rows": int(len(merged_df)),
    }
    with open(out_dir / f"{scenario_name}_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[TWC_PLOTS2] wrote {out_csv}")


if __name__ == "__main__":
    main()
