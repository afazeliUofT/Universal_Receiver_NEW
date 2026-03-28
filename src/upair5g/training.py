from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, get_resource_grid
from .config import ensure_output_tree, get_cfg
from .estimator import UPAIRChannelEstimator
from .impairments import apply_symbol_phase_impairment
from .utils import (
    call_channel,
    call_transmitter,
    complex_sq_abs,
    compute_nmse,
    ebno_db_to_no,
    save_json,
    save_yaml,
    set_global_seed,
    tf_float,
)


def _make_batch(
    tx: Any,
    channel: Any,
    cfg: dict[str, Any],
    batch_size: int,
    training: bool,
    fixed_ebno_db: float | None = None,
) -> dict[str, tf.Tensor]:
    x, bits = call_transmitter(tx, batch_size)

    if fixed_ebno_db is None:
        ebno_db = tf.random.uniform(
            [],
            minval=float(get_cfg(cfg, "system.ebno_db_train_min", 0.0)),
            maxval=float(get_cfg(cfg, "system.ebno_db_train_max", 16.0)),
            dtype=tf.float32,
        )
    else:
        ebno_db = tf.constant(float(fixed_ebno_db), tf.float32)

    no = ebno_db_to_no(ebno_db, tx=tx, resource_grid=get_resource_grid(tx))
    y, h = call_channel(channel, x, no)
    y, h = apply_symbol_phase_impairment(y, h, cfg, training=training)

    return {
        "x": x,
        "b": bits,
        "y": y,
        "h": h,
        "no": no,
        "ebno_db": ebno_db,
    }


def _make_optimizer(cfg: dict[str, Any]) -> tf.keras.optimizers.Optimizer:
    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"]["weight_decay"])
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    except Exception:
        return tf.keras.optimizers.Adam(learning_rate=lr)


def _train_step(
    estimator: UPAIRChannelEstimator,
    optimizer: tf.keras.optimizers.Optimizer,
    batch: dict[str, tf.Tensor],
    nmse_loss_weight: float,
    grad_clip_norm: float,
) -> dict[str, tf.Tensor]:
    with tf.GradientTape() as tape:
        h_hat, err_hat, h_ls, _ = estimator.estimate_with_ls(batch["y"], batch["no"], training=True)
        target = tf.convert_to_tensor(batch["h"])

        residual = target - h_hat
        residual_ls = target - h_ls
        sq_err = complex_sq_abs(residual)
        power = tf.reduce_mean(complex_sq_abs(target)) + 1e-9

        loss_nll = tf.reduce_mean(sq_err / (err_hat + 1e-6) + tf.math.log(err_hat + 1e-6))
        nmse_prop = tf.reduce_mean(sq_err) / power
        nmse_ls = tf.reduce_mean(complex_sq_abs(residual_ls)) / power
        loss = loss_nll + float(nmse_loss_weight) * nmse_prop

    grads = tape.gradient(loss, estimator.trainable_variables)
    grad_var_pairs = [(g, v) for g, v in zip(grads, estimator.trainable_variables) if g is not None]
    if grad_var_pairs:
        grad_tensors = [g for g, _ in grad_var_pairs]
        clipped_grads, _ = tf.clip_by_global_norm(grad_tensors, float(grad_clip_norm))
        optimizer.apply_gradients(zip(clipped_grads, [v for _, v in grad_var_pairs]))

    return {
        "loss": tf.cast(loss, tf.float32),
        "loss_nll": tf.cast(loss_nll, tf.float32),
        "nmse_prop": tf.cast(nmse_prop, tf.float32),
        "nmse_ls": tf.cast(nmse_ls, tf.float32),
    }


def _validate(
    estimator: UPAIRChannelEstimator,
    tx: Any,
    channel: Any,
    cfg: dict[str, Any],
) -> dict[str, float]:
    val_steps = int(cfg["training"]["val_steps"])
    eval_grid = list(get_cfg(cfg, "system.ebno_db_eval", [10]))
    ebno_for_val = float(eval_grid[min(len(eval_grid) // 2, len(eval_grid) - 1)])

    nmse_prop = []
    nmse_ls = []

    for _ in range(val_steps):
        batch = _make_batch(
            tx=tx,
            channel=channel,
            cfg=cfg,
            batch_size=int(cfg["system"]["batch_size_eval"]),
            training=False,
            fixed_ebno_db=ebno_for_val,
        )
        h_hat, _, h_ls, _ = estimator.estimate_with_ls(batch["y"], batch["no"], training=False)
        nmse_prop.append(float(compute_nmse(batch["h"], h_hat).numpy()))
        nmse_ls.append(float(compute_nmse(batch["h"], h_ls).numpy()))

    return {
        "val_nmse_prop": float(np.mean(nmse_prop)),
        "val_nmse_ls": float(np.mean(nmse_ls)),
        "val_ebno_db": ebno_for_val,
    }


def train_model(cfg: dict[str, Any]) -> dict[str, Any]:
    set_global_seed(int(cfg["system"]["seed"]))
    paths = ensure_output_tree(cfg)

    tx, _ = build_pusch_transmitter(cfg)
    channel = build_channel(cfg, tx)
    ls_estimator = build_ls_estimator(tx, cfg)
    estimator = UPAIRChannelEstimator(ls_estimator=ls_estimator, resource_grid=get_resource_grid(tx), cfg=cfg)
    optimizer = _make_optimizer(cfg)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    ckpt_path = paths["checkpoints"] / str(cfg["training"]["checkpoint_name"])

    total_steps = int(cfg["training"]["steps"])
    log_every = int(cfg["training"]["log_every"])
    eval_every = int(cfg["training"]["eval_every"])
    nmse_loss_weight = float(cfg["training"]["nmse_loss_weight"])
    grad_clip_norm = float(cfg["training"]["grad_clip_norm"])

    for step in range(1, total_steps + 1):
        batch = _make_batch(
            tx=tx,
            channel=channel,
            cfg=cfg,
            batch_size=int(cfg["system"]["batch_size_train"]),
            training=True,
        )
        metrics = _train_step(
            estimator=estimator,
            optimizer=optimizer,
            batch=batch,
            nmse_loss_weight=nmse_loss_weight,
            grad_clip_norm=grad_clip_norm,
        )
        row = {
            "step": step,
            "ebno_db": float(batch["ebno_db"].numpy()),
            "loss": float(metrics["loss"].numpy()),
            "loss_nll": float(metrics["loss_nll"].numpy()),
            "nmse_prop": float(metrics["nmse_prop"].numpy()),
            "nmse_ls": float(metrics["nmse_ls"].numpy()),
        }

        if step % eval_every == 0 or step == total_steps:
            val_metrics = _validate(estimator, tx, channel, cfg)
            row.update(val_metrics)
            if val_metrics["val_nmse_prop"] < best_val:
                best_val = val_metrics["val_nmse_prop"]
                estimator.save_weights(str(ckpt_path))

        history.append(row)

        if step % log_every == 0 or step == 1 or step == total_steps:
            print(
                f"[TRAIN] step={step:05d} "
                f"loss={row['loss']:.5f} "
                f"nmse_prop={row['nmse_prop']:.5f} "
                f"nmse_ls={row['nmse_ls']:.5f}"
            )

    if not ckpt_path.exists():
        estimator.save_weights(str(ckpt_path))

    save_json({"history": history}, paths["metrics"] / "history.json")
    save_yaml(cfg, paths["artifacts"] / "resolved_config.yaml")

    return {
        "output_dir": str(paths["root"]),
        "checkpoint_path": str(ckpt_path),
        "history_path": str(paths["metrics"] / "history.json"),
    }
