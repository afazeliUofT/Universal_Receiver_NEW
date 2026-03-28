from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from .compat import first_present_attr, instantiate_filtered, resolve_attr, set_if_present
from .config import get_cfg


def build_pusch_config(cfg: dict[str, Any]) -> Any:
    PUSCHConfig = resolve_attr(["sionna.phy.nr", "sionna.nr"], "PUSCHConfig")
    pc = PUSCHConfig()

    pusch_cfg = cfg["pusch"]
    dmrs_cfg = pusch_cfg["dmrs"]

    carrier = getattr(pc, "carrier", None)
    tb = getattr(pc, "tb", None)
    dmrs = getattr(pc, "dmrs", None)

    if carrier is not None:
        set_if_present(carrier, "n_size_grid", int(pusch_cfg["n_size_grid"]))
        set_if_present(carrier, "subcarrier_spacing", int(pusch_cfg["subcarrier_spacing_khz"]))
        set_if_present(carrier, "cyclic_prefix", str(pusch_cfg["cyclic_prefix"]))

    set_if_present(pc, "n_size_bwp", int(pusch_cfg["n_size_bwp"]))
    set_if_present(pc, "mapping_type", str(pusch_cfg["mapping_type"]))
    set_if_present(pc, "symbol_allocation", list(pusch_cfg["symbol_allocation"]))
    set_if_present(pc, "num_layers", int(pusch_cfg["num_layers"]))
    set_if_present(pc, "num_antenna_ports", int(pusch_cfg["num_antenna_ports"]))
    set_if_present(pc, "precoding", str(pusch_cfg["precoding"]))
    set_if_present(pc, "transform_precoding", bool(pusch_cfg["transform_precoding"]))

    if tb is not None:
        try:
            set_if_present(tb, "mcs_index", int(pusch_cfg["mcs_index"]))
        except Exception:
            pass
        try:
            set_if_present(tb, "mcs_table", pusch_cfg["mcs_table"])
        except Exception:
            pass

    if dmrs is not None:
        set_if_present(dmrs, "config_type", int(dmrs_cfg["config_type"]))
        set_if_present(dmrs, "length", int(dmrs_cfg["length"]))
        set_if_present(dmrs, "additional_position", int(dmrs_cfg["additional_position"]))
        set_if_present(dmrs, "type_a_position", int(dmrs_cfg["type_a_position"]))
        set_if_present(dmrs, "num_cdm_groups_without_data", int(dmrs_cfg["num_cdm_groups_without_data"]))

    return pc


def build_pusch_transmitter(cfg: dict[str, Any]) -> tuple[Any, Any]:
    PUSCHTransmitter = resolve_attr(["sionna.phy.nr", "sionna.nr"], "PUSCHTransmitter")
    pc = build_pusch_config(cfg)

    kwargs = {
        "pusch_configs": [pc],
        "output_domain": "freq",
        "return_bits": True,
        "precision": get_cfg(cfg, "system.precision", "single"),
    }

    try:
        tx = instantiate_filtered(PUSCHTransmitter, **kwargs)
    except Exception:
        try:
            kwargs["pusch_configs"] = pc
            tx = instantiate_filtered(PUSCHTransmitter, **kwargs)
        except Exception:
            try:
                tx = PUSCHTransmitter([pc], output_domain="freq", return_bits=True, precision=get_cfg(cfg, "system.precision", "single"))
            except Exception:
                tx = PUSCHTransmitter(pc, output_domain="freq", return_bits=True, precision=get_cfg(cfg, "system.precision", "single"))

    return tx, pc


def get_resource_grid(tx: Any) -> Any:
    rg = first_present_attr(tx, ["resource_grid", "_resource_grid"], None)
    if rg is None:
        raise AttributeError("Could not locate resource_grid in PUSCH transmitter.")
    return rg


def build_ls_estimator(
    tx: Any,
    cfg: dict[str, Any],
    interpolation_type: str = "lin",
    interpolator: Any | None = None,
) -> Any:
    PUSCHLSChannelEstimator = resolve_attr(["sionna.phy.nr", "sionna.nr"], "PUSCHLSChannelEstimator")
    rg = get_resource_grid(tx)
    kwargs = {
        "resource_grid": rg,
        "dmrs_length": first_present_attr(tx, ["_dmrs_length"], 1),
        "dmrs_additional_position": first_present_attr(tx, ["_dmrs_additional_position"], 0),
        "num_cdm_groups_without_data": first_present_attr(tx, ["_num_cdm_groups_without_data"], 2),
        "precision": get_cfg(cfg, "system.precision", "single"),
    }
    if interpolator is not None:
        kwargs["interpolator"] = interpolator
    else:
        kwargs["interpolation_type"] = str(interpolation_type)
    try:
        return instantiate_filtered(PUSCHLSChannelEstimator, **kwargs)
    except Exception:
        if interpolator is not None:
            return PUSCHLSChannelEstimator(
                rg,
                first_present_attr(tx, ["_dmrs_length"], 1),
                first_present_attr(tx, ["_dmrs_additional_position"], 0),
                first_present_attr(tx, ["_num_cdm_groups_without_data"], 2),
                interpolator=interpolator,
                precision=get_cfg(cfg, "system.precision", "single"),
            )
        return PUSCHLSChannelEstimator(
            rg,
            first_present_attr(tx, ["_dmrs_length"], 1),
            first_present_attr(tx, ["_dmrs_additional_position"], 0),
            first_present_attr(tx, ["_num_cdm_groups_without_data"], 2),
            interpolation_type=str(interpolation_type),
            precision=get_cfg(cfg, "system.precision", "single"),
        )


def _build_single_antenna(num_ant: int, carrier_frequency: float) -> Any:
    try:
        Antenna = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "Antenna")
        return instantiate_filtered(
            Antenna,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
        )
    except Exception:
        AntennaArray = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "AntennaArray")
        return instantiate_filtered(
            AntennaArray,
            num_rows=1,
            num_cols=num_ant,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
        )


def _build_bs_array(num_ant: int, carrier_frequency: float) -> Any:
    AntennaArray = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "AntennaArray")
    return instantiate_filtered(
        AntennaArray,
        num_rows=1,
        num_cols=num_ant,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
    )


def build_channel(cfg: dict[str, Any], tx: Any) -> Any:
    OFDMChannel = resolve_attr(["sionna.phy.channel", "sionna.channel"], "OFDMChannel")
    CDL = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "CDL")

    channel_cfg = cfg["channel"]
    pusch_cfg = cfg["pusch"]
    carrier_frequency = float(pusch_cfg["carrier_frequency_hz"])

    ut_array = _build_single_antenna(int(channel_cfg["num_tx_ant"]), carrier_frequency)
    bs_array = _build_bs_array(int(channel_cfg["num_rx_ant"]), carrier_frequency)

    cdl_kwargs = {
        "model": str(channel_cfg["model"]),
        "delay_spread": float(channel_cfg["delay_spread_s"]),
        "carrier_frequency": carrier_frequency,
        "ut_array": ut_array,
        "bs_array": bs_array,
        "direction": "uplink",
        "min_speed": float(channel_cfg["min_speed_mps"]),
        "max_speed": float(channel_cfg["max_speed_mps"]),
        "dtype": tf.complex64,
    }
    try:
        channel_model = instantiate_filtered(CDL, **cdl_kwargs)
    except Exception:
        channel_model = CDL(
            str(channel_cfg["model"]),
            float(channel_cfg["delay_spread_s"]),
            carrier_frequency,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            min_speed=float(channel_cfg["min_speed_mps"]),
            max_speed=float(channel_cfg["max_speed_mps"]),
        )

    ofdm_kwargs = {
        "channel_model": channel_model,
        "resource_grid": get_resource_grid(tx),
        "add_awgn": True,
        "normalize_channel": bool(channel_cfg["normalize_channel"]),
        "return_channel": True,
    }
    try:
        return instantiate_filtered(OFDMChannel, **ofdm_kwargs)
    except Exception:
        return OFDMChannel(
            channel_model,
            get_resource_grid(tx),
            add_awgn=True,
            normalize_channel=bool(channel_cfg["normalize_channel"]),
            return_channel=True,
        )


def build_receiver(tx: Any, cfg: dict[str, Any], channel_estimator: Any | None = None, perfect_csi: bool = False) -> Any:
    PUSCHReceiver = resolve_attr(["sionna.phy.nr", "sionna.nr"], "PUSCHReceiver")
    kwargs = {
        "pusch_transmitter": tx,
        "return_tb_crc_status": True,
        "input_domain": "freq",
        "precision": get_cfg(cfg, "system.precision", "single"),
    }
    if perfect_csi:
        kwargs["channel_estimator"] = "perfect"
    elif channel_estimator is not None:
        kwargs["channel_estimator"] = channel_estimator
    try:
        return instantiate_filtered(PUSCHReceiver, **kwargs)
    except Exception:
        return PUSCHReceiver(tx, **{k: v for k, v in kwargs.items() if k != "pusch_transmitter"})


def extract_pilot_mask(resource_grid: Any) -> tf.Tensor:
    pilot_pattern = first_present_attr(resource_grid, ["pilot_pattern", "_pilot_pattern"], None)
    if pilot_pattern is None:
        raise AttributeError("Could not locate pilot pattern in resource_grid.")
    mask = first_present_attr(pilot_pattern, ["mask", "_mask"], None)
    if mask is None:
        raise AttributeError("Could not locate pilot mask in pilot pattern.")

    mask = tf.cast(tf.convert_to_tensor(mask), tf.float32)

    # Collapse leading singleton dimensions across Sionna versions
    while mask.shape.rank is not None and mask.shape.rank > 2:
        squeezed = False
        for axis, dim in enumerate(mask.shape.as_list()):
            if dim == 1:
                mask = tf.squeeze(mask, axis=axis)
                squeezed = True
                break
        if not squeezed:
            break

    if mask.shape.rank != 2:
        raise ValueError(f"Unexpected pilot-mask rank after squeezing: {mask.shape.rank}")

    # Some versions store [F, T] instead of [T, F]
    static_shape = mask.shape.as_list()
    if static_shape is not None and len(static_shape) == 2:
        t, f = static_shape
        if t is not None and f is not None and t > f:
            mask = tf.transpose(mask, [1, 0])

    return tf.expand_dims(mask, axis=-1)
