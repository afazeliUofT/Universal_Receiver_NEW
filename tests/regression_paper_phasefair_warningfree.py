from __future__ import annotations

import argparse
import warnings

import tensorflow as tf

from upair5g.baselines import (
    BASELINE_PAPER_CFGRES_PHASEAWARE_LS_LMMSE,
    build_classical_baseline_suite,
)
from upair5g.builders import build_channel, build_pusch_transmitter
from upair5g.config import ensure_output_tree, load_config
from upair5g.impairments import apply_symbol_phase_impairment
from upair5g.utils import call_channel, call_receiver, call_transmitter, ebno_db_to_no, set_global_seed


BLOCKED_WARNING_SNIPPETS = (
    "looks like it has unbuilt state",
    "does not have a `build()` method",
    "Could not parse estimator inputs.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression that the phase-aware paper comparator is warning-free.")
    parser.add_argument("--config", default="configs/target_cdlc_highmobility_paper_phasefair_comparison.yaml", type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg["system"]["seed"]))
    paths = ensure_output_tree(cfg)
    tx, _ = build_pusch_transmitter(cfg)
    channel = build_channel(cfg, tx)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        classical_receivers, classical_estimators, _ = build_classical_baseline_suite(
            cfg=cfg,
            tx=tx,
            channel=channel,
            paths=paths,
        )

        estimator = classical_estimators[BASELINE_PAPER_CFGRES_PHASEAWARE_LS_LMMSE]
        receiver = classical_receivers[BASELINE_PAPER_CFGRES_PHASEAWARE_LS_LMMSE]

        batch_size = min(2, int(cfg["system"]["batch_size_eval"]))
        ebno_db = float(cfg["system"]["ebno_db_eval"][0])

        x, _ = call_transmitter(tx, batch_size)
        no = ebno_db_to_no(ebno_db, tx=tx)
        y, h = call_channel(channel, x, no)
        y, h = apply_symbol_phase_impairment(y, h, cfg, training=False)

        h_hat, err_hat = estimator(y, no)
        if h_hat.shape != h.shape:
            raise AssertionError(f"Phase-aware paper output shape mismatch: {h_hat.shape} vs {h.shape}")
        if err_hat.shape != h.shape:
            raise AssertionError(f"Phase-aware paper err_var shape mismatch: {err_hat.shape} vs {h.shape}")

        call_receiver(receiver, y, no)

    blocked = [
        str(w.message)
        for w in caught
        if any(snippet in str(w.message) for snippet in BLOCKED_WARNING_SNIPPETS)
    ]
    if blocked:
        raise AssertionError(
            "Unexpected phase-aware paper build warnings were emitted:\n" + "\n".join(blocked)
        )

    print("Phase-aware paper comparator warning-free regression passed.")


if __name__ == "__main__":
    main()
