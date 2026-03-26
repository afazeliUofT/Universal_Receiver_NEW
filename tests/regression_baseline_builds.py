from __future__ import annotations

import argparse

from upair5g.baselines import PERFECT_RECEIVER, PROPOSED_RECEIVER, build_classical_baseline_suite, wants_receiver
from upair5g.builders import build_channel, build_ls_estimator, build_pusch_transmitter, build_receiver, get_resource_grid
from upair5g.config import ensure_output_tree, load_config
from upair5g.estimator import UPAIRChannelEstimator
from upair5g.impairments import apply_symbol_phase_impairment
from upair5g.utils import call_channel, call_receiver, call_transmitter, ebno_db_to_no, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression for richer baselines build/forward.")
    parser.add_argument("--config", default="configs/smoke_richer_baselines.yaml", type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg["system"]["seed"]))
    paths = ensure_output_tree(cfg)

    tx, _ = build_pusch_transmitter(cfg)
    channel = build_channel(cfg, tx)
    classical_receivers, _, _ = build_classical_baseline_suite(cfg=cfg, tx=tx, channel=channel, paths=paths)

    ls_estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")
    proposed_estimator = UPAIRChannelEstimator(ls_estimator=ls_estimator, resource_grid=get_resource_grid(tx), cfg=cfg)
    proposed_rx = build_receiver(tx, cfg, channel_estimator=proposed_estimator, perfect_csi=False) if wants_receiver(cfg, PROPOSED_RECEIVER) else None
    perfect_rx = build_receiver(tx, cfg, channel_estimator=None, perfect_csi=True) if wants_receiver(cfg, PERFECT_RECEIVER) else None

    batch_size = min(2, int(cfg["system"]["batch_size_eval"]))
    ebno_db = float(cfg["system"]["ebno_db_eval"][0])
    x, _ = call_transmitter(tx, batch_size)
    no = ebno_db_to_no(ebno_db, tx=tx, resource_grid=get_resource_grid(tx))
    y, h = call_channel(channel, x, no)
    y, h = apply_symbol_phase_impairment(y, h, cfg, training=False)

    proposed_estimator.estimate_with_ls(y, no, training=False)

    for receiver_name, receiver_block in classical_receivers.items():
        call_receiver(receiver_block, y, no)
        print(f"Baseline forward passed: {receiver_name}")

    if proposed_rx is not None:
        call_receiver(proposed_rx, y, no)
        print("Baseline forward passed: upair5g_lmmse")

    if perfect_rx is not None:
        call_receiver(perfect_rx, y, no, h=h)
        print("Baseline forward passed: perfect_csi_lmmse")

    print("Richer-baselines regression passed.")


if __name__ == "__main__":
    main()
