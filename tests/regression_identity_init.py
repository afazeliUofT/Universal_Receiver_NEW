from __future__ import annotations

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

        real = tf.random.normal(
            [batch_size, 1, num_rx_ant, 1, 1, num_symbols, num_subcarriers],
            dtype=tf.float32,
        )
        imag = tf.random.normal(
            [batch_size, 1, num_rx_ant, 1, 1, num_symbols, num_subcarriers],
            dtype=tf.float32,
        )
        h_ls = tf.complex(real, imag)
        err_var = tf.fill(
            [batch_size, 1, num_rx_ant, 1, 1, num_symbols, num_subcarriers],
            0.1,
        )
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
    h_hat, err_hat, h_ls, err_ls = estimator.estimate_with_ls(y, no, training=False)

    np.testing.assert_allclose(h_hat.numpy(), h_ls.numpy(), rtol=0.0, atol=1e-7)
    if not tf.reduce_all(err_hat >= err_ls):
        raise RuntimeError("Refined error variance should not be smaller than LS at initialization.")

    print("Identity-init LS regression passed.")


if __name__ == "__main__":
    main()
