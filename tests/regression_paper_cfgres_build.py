from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from upair5g.baselines import BASELINE_PAPER_CFGRES_LS_LMMSE, build_classical_baseline_suite
from upair5g.builders import build_channel, build_pusch_transmitter
from upair5g.config import ensure_output_tree, load_config
from upair5g.impairments import apply_symbol_phase_impairment
from upair5g.utils import call_channel, call_receiver, call_transmitter, ebno_db_to_no, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression for paper-style configured-reservoir comparator.")
    parser.add_argument("--config", default="configs/smoke_paper_comparison.yaml", type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg["system"]["seed"]))
    paths = ensure_output_tree(cfg)

    tx, _ = build_pusch_transmitter(cfg)
    channel = build_channel(cfg, tx)
    classical_receivers, classical_estimators, artifacts = build_classical_baseline_suite(cfg=cfg, tx=tx, channel=channel, paths=paths)

    if BASELINE_PAPER_CFGRES_LS_LMMSE not in classical_estimators:
        raise AssertionError("Paper configured-reservoir estimator was not built.")
    if BASELINE_PAPER_CFGRES_LS_LMMSE not in classical_receivers:
        raise AssertionError("Paper configured-reservoir receiver was not built.")

    basis_path = artifacts.get("paper_cfgres_basis")
    cov_path = artifacts.get("paper_cfgres_covariances")
    if basis_path is None or cov_path is None:
        raise AssertionError("Paper configured-reservoir artifacts were not registered.")
    if not Path(basis_path).exists():
        raise AssertionError(f"Missing paper configured-reservoir basis artifact: {basis_path}")
    if not Path(cov_path).exists():
        raise AssertionError(f"Missing paper configured-reservoir covariance artifact: {cov_path}")

    batch_size = min(2, int(cfg["system"]["batch_size_eval"]))
    ebno_db = float(cfg["system"]["ebno_db_eval"][0])
    x, _ = call_transmitter(tx, batch_size)
    no = ebno_db_to_no(ebno_db, tx=tx)
    y, h = call_channel(channel, x, no)
    y, h = apply_symbol_phase_impairment(y, h, cfg, training=False)

    estimator = classical_estimators[BASELINE_PAPER_CFGRES_LS_LMMSE]
    h_hat, err_hat = estimator(y, no)
    if h_hat.shape != h.shape:
        raise AssertionError(f"Paper configured-reservoir output shape mismatch: {h_hat.shape} vs {h.shape}")
    if err_hat.shape != h.shape:
        raise AssertionError(f"Paper configured-reservoir err_var shape mismatch: {err_hat.shape} vs {h.shape}")
    if not bool(tf.reduce_all(tf.math.is_finite(tf.math.real(h_hat))).numpy()):
        raise AssertionError("Paper configured-reservoir h_hat contains non-finite real values.")
    if not bool(tf.reduce_all(tf.math.is_finite(tf.math.imag(h_hat))).numpy()):
        raise AssertionError("Paper configured-reservoir h_hat contains non-finite imaginary values.")
    if not bool(tf.reduce_all(tf.math.is_finite(tf.cast(err_hat, tf.float32))).numpy()):
        raise AssertionError("Paper configured-reservoir err_hat contains non-finite values.")

    receiver = classical_receivers[BASELINE_PAPER_CFGRES_LS_LMMSE]
    call_receiver(receiver, y, no)
    print("Paper configured-reservoir regression passed.")


if __name__ == "__main__":
    main()
