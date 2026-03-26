from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from upair5g.estimator import UPAIRChannelEstimator


class _DummyPilotPattern:
    def __init__(self, mask: tf.Tensor) -> None:
        self.mask = mask


class _DummyResourceGrid:
    def __init__(self, mask: tf.Tensor) -> None:
        self.pilot_pattern = _DummyPilotPattern(mask)


class _DummyLSEstimator:
    def __call__(self, y: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        y = tf.convert_to_tensor(y)
        batch_size = tf.shape(y)[0]
        num_rx_ant = tf.shape(y)[2]
        num_symbols = tf.shape(y)[3]
        num_subcarriers = tf.shape(y)[4]

        h_ls = tf.complex(
            tf.zeros([batch_size, 1, num_rx_ant, 1, 1, num_symbols, num_subcarriers], dtype=tf.float32),
            tf.zeros([batch_size, 1, num_rx_ant, 1, 1, num_symbols, num_subcarriers], dtype=tf.float32),
        )
        err_var = tf.fill([batch_size, 1, num_rx_ant, 1, 1, num_symbols, num_subcarriers], 0.1)
        return h_ls, err_var


def _make_cfg(num_rx_ant: int) -> dict[str, dict[str, float | int | bool]]:
    return {
        "channel": {
            "num_rx_ant": num_rx_ant,
        },
        "model": {
            "d_model": 8,
            "num_blocks": 1,
            "num_heads": 1,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "residual_scale": 0.25,
            "use_noise_feature": True,
            "use_pilot_mask_feature": True,
        },
    }


def _make_inputs(batch_size: int, num_rx_ant: int, num_symbols: int, num_subcarriers: int) -> tuple[tf.Tensor, tf.Tensor]:
    real = tf.random.normal([batch_size, 1, num_rx_ant, num_symbols, num_subcarriers], dtype=tf.float32)
    imag = tf.random.normal([batch_size, 1, num_rx_ant, num_symbols, num_subcarriers], dtype=tf.float32)
    y = tf.complex(real, imag)
    no = tf.constant(0.05, dtype=tf.float32)
    return y, no


def main() -> None:
    tf.random.set_seed(123)
    np.random.seed(123)

    batch_size = 2
    num_rx_ant = 2
    num_symbols = 4
    num_subcarriers = 8
    pilot_mask = tf.constant(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=tf.float32,
    )
    resource_grid = _DummyResourceGrid(pilot_mask)
    cfg = _make_cfg(num_rx_ant=num_rx_ant)
    ls_estimator = _DummyLSEstimator()
    y, no = _make_inputs(
        batch_size=batch_size,
        num_rx_ant=num_rx_ant,
        num_symbols=num_symbols,
        num_subcarriers=num_subcarriers,
    )

    estimator = UPAIRChannelEstimator(ls_estimator=ls_estimator, resource_grid=resource_grid, cfg=cfg)
    estimator.estimate_with_ls(y, no, training=False)
    if not estimator.built:
        raise RuntimeError("Estimator should be marked built after direct estimate_with_ls() forward pass.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "checkpoint.weights.h5"
        estimator.save_weights(str(checkpoint_path))

        restored = UPAIRChannelEstimator(ls_estimator=ls_estimator, resource_grid=resource_grid, cfg=cfg)
        restored.estimate_with_ls(y, no, training=False)
        restored.load_weights(str(checkpoint_path))

        if len(estimator.weights) != len(restored.weights):
            raise RuntimeError("Weight count mismatch after checkpoint reload.")
        for ref_var, got_var in zip(estimator.weights, restored.weights):
            np.testing.assert_allclose(ref_var.numpy(), got_var.numpy(), rtol=1e-6, atol=1e-6)

    print("Checkpoint build/save/load regression passed.")


if __name__ == "__main__":
    main()
