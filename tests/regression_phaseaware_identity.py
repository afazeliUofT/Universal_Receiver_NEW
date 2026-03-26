from __future__ import annotations

import argparse
from copy import deepcopy

import tensorflow as tf

from upair5g.builders import build_channel, build_ls_estimator, build_pusch_transmitter, get_resource_grid
from upair5g.compat import safe_call_variants
from upair5g.config import load_config, set_cfg
from upair5g.impairments import apply_symbol_phase_impairment
from upair5g.phase_aware import DecisionDirectedCPEEstimator
from upair5g.utils import call_channel, call_transmitter, ebno_db_to_no, infer_num_bits_per_symbol, set_global_seed


def _call_estimator(estimator, y: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    out = safe_call_variants(estimator, y, no)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError("Estimator must return (h_hat, err_var).")
    return tf.convert_to_tensor(out[0]), tf.convert_to_tensor(out[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression for phase-aware baseline identity/build.")
    parser.add_argument("--config", default="configs/smoke_phaseaware_baselines.yaml", type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg["system"]["seed"]))

    tx, _ = build_pusch_transmitter(cfg)
    channel = build_channel(cfg, tx)
    base_estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")

    batch_size = min(2, int(cfg["system"]["batch_size_eval"]))
    ebno_db = float(cfg["system"]["ebno_db_eval"][0])

    x, _ = call_transmitter(tx, batch_size)
    no = ebno_db_to_no(ebno_db, tx=tx, resource_grid=get_resource_grid(tx))
    y, _ = call_channel(channel, x, no)
    y, _ = apply_symbol_phase_impairment(y, None, cfg, training=False)

    h_base, err_base = _call_estimator(base_estimator, y, no)

    cfg_identity = deepcopy(cfg)
    set_cfg(cfg_identity, "baselines.phase_aware_ddcpe.num_iterations", 0)
    phase_identity = DecisionDirectedCPEEstimator(
        base_estimator=base_estimator,
        resource_grid=get_resource_grid(tx),
        bits_per_symbol=infer_num_bits_per_symbol(tx, default=6),
        cfg=cfg_identity,
    )
    h_identity, err_identity = phase_identity(y, no)

    max_h_diff = float(tf.reduce_max(tf.abs(h_identity - h_base)).numpy())
    max_err_diff = float(tf.reduce_max(tf.abs(tf.cast(err_identity, tf.float32) - tf.cast(err_base, tf.float32))).numpy())
    if max_h_diff > 1e-6 or max_err_diff > 1e-6:
        raise AssertionError(
            f"Phase-aware identity regression failed: max_h_diff={max_h_diff:.3e}, max_err_diff={max_err_diff:.3e}"
        )

    phase_runtime = DecisionDirectedCPEEstimator(
        base_estimator=base_estimator,
        resource_grid=get_resource_grid(tx),
        bits_per_symbol=infer_num_bits_per_symbol(tx, default=6),
        cfg=cfg,
    )
    h_phase, err_phase = phase_runtime(y, no)
    if h_phase.shape != h_base.shape or err_phase.shape != err_base.shape:
        raise AssertionError(
            f"Unexpected phase-aware output shapes: h={h_phase.shape} vs {h_base.shape}, "
            f"err={err_phase.shape} vs {err_base.shape}"
        )

    finite_real = bool(tf.reduce_all(tf.math.is_finite(tf.math.real(h_phase))).numpy())
    finite_imag = bool(tf.reduce_all(tf.math.is_finite(tf.math.imag(h_phase))).numpy())
    if (not finite_real) or (not finite_imag):
        raise AssertionError("Phase-aware channel estimate contains non-finite values.")

    print("Phase-aware identity/build regression passed.")


if __name__ == "__main__":
    main()
