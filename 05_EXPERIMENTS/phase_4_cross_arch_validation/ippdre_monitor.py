# ==============================================================================
# ACC EXPERIMENTAL HARNESS (UAI 2026)
# ==============================================================================
# PROVENANCE: i-ppDRE Density Chasm Monitor (HP Z4 Workstation)
# AUTHOR: Krishnamurthi (Lead Researcher)
# DATE: 2026-02-14
#
# DESCRIPTION:
# Real-time i-ppDRE monitoring for student/teacher divergence. This script:
#   1) Streams student states from shared memory (GPU student).
#   2) Uses teacher activations to set the "safe" manifold.
#   3) Tracks density ratio drift (i-ppDRE) at <5ms/token.
#   4) Estimates KL(p||q) to flag the Density Chasm (threshold ~20 nats).
#   5) Calibrates a conformal safety gate for interventions.
# ==============================================================================

import argparse
import os
import sys
import time
from collections import deque
from typing import Deque

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT, "02_SRC")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from acc_core.detector.ipp_dre import IncrementalDriftTracker
from acc_core.control.conformal import ConformalSafetyGate
from acc_core.system.ring_buffer import ACCOpsBridge


def _load_teacher_states(path: str, input_dim: int, max_samples: int | None) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Teacher states not found: {path}")
    states = np.load(path)
    if states.ndim == 1:
        states = states[None, :]
    if states.shape[1] < input_dim:
        pad = np.zeros((states.shape[0], input_dim - states.shape[1]), dtype=states.dtype)
        states = np.concatenate([states, pad], axis=1)
    elif states.shape[1] > input_dim:
        states = states[:, :input_dim]
    if max_samples is not None and states.shape[0] > max_samples:
        states = states[:max_samples]
    return states.astype(np.float32, copy=False)


def _estimate_kl_from_teacher(
    tracker: IncrementalDriftTracker,
    teacher_states: np.ndarray,
    sample_count: int,
) -> float:
    if teacher_states.size == 0:
        return 0.0
    n = min(sample_count, teacher_states.shape[0])
    idx = np.random.choice(teacher_states.shape[0], size=n, replace=teacher_states.shape[0] < n)
    samples = teacher_states[idx]
    log_ratios = []
    for z in samples:
        w_x = tracker.score(z)
        log_ratios.append(np.log(max(w_x, 1e-8)))
    return float(np.mean(log_ratios))


def main() -> None:
    parser = argparse.ArgumentParser(description="i-ppDRE Density Chasm Monitor")
    parser.add_argument("--teacher-states", required=True, help=".npy file with teacher activations")
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--rff-dim", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--kl-threshold", type=float, default=20.0)
    parser.add_argument("--kl-sample-count", type=int, default=256)
    parser.add_argument("--latency-budget-ms", type=float, default=5.0)
    parser.add_argument("--bridge-name", default="acc_bridge")
    parser.add_argument("--log-file", default="../../06_RESULTS/phase_4_logs/ippdre_monitor.log")
    parser.add_argument("--max-teacher-samples", type=int, default=5000)
    args = parser.parse_args()

    teacher_states = _load_teacher_states(args.teacher_states, args.input_dim, args.max_teacher_samples)

    tracker = IncrementalDriftTracker(
        input_dim=args.input_dim,
        rff_dim=args.rff_dim,
        alpha_lambda=0.99,
        device="cpu",
    )

    gate = ConformalSafetyGate(epsilon=args.epsilon, min_threshold=0.005)

    tracker.update_teacher_baseline(torch.from_numpy(teacher_states))

    for z in teacher_states[: args.calibration_samples]:
        gate.add_calibration_score(tracker.score(z))
    gate.calibrate()

    bridge = ACCOpsBridge(name=args.bridge_name, create=True)

    kl_window: Deque[float] = deque(maxlen=50)
    latency_window: Deque[float] = deque(maxlen=200)

    last_step = -1
    with open(args.log_file, "a") as log:
        log.write("--- i-ppDRE MONITOR SESSION START ---\n")
        log.write(f"TEACHER_STATES={args.teacher_states}\n")
        log.write(f"EPSILON={args.epsilon} KL_THRESHOLD={args.kl_threshold}\n")

    try:
        while True:
            timestamp, _, step = bridge.read_latest_state()
            if step <= 0 or step == last_step:
                time.sleep(0.001)
                continue

            raw_bytes = bridge.buffer[16 : 16 + (args.input_dim * 4)]
            z_t = np.frombuffer(raw_bytes, dtype=np.float32).copy()

            start = time.perf_counter()
            w_x = tracker.score(z_t)
            tracker.update(z_t)
            latency_ms = (time.perf_counter() - start) * 1000.0
            latency_window.append(latency_ms)

            if gate.check(w_x):
                bridge.trigger_intervention()

            kl_est = _estimate_kl_from_teacher(tracker, teacher_states, args.kl_sample_count)
            kl_window.append(kl_est)
            kl_ema = float(np.mean(kl_window))

            density_chasm = kl_ema >= args.kl_threshold
            latency_ok = latency_ms <= args.latency_budget_ms

            with open(args.log_file, "a") as log:
                log.write(
                    f"step={step} t={timestamp:.3f} w_x={w_x:.5f} "
                    f"kl={kl_ema:.2f} latency_ms={latency_ms:.3f} "
                    f"gate={'1' if gate.check(w_x) else '0'} "
                    f"chasm={'1' if density_chasm else '0'} "
                    f"latency_ok={'1' if latency_ok else '0'}\n"
                )

            if density_chasm:
                print(
                    f"[DENSITY CHASM] KL~{kl_ema:.2f} >= {args.kl_threshold} "
                    f"(step {step})"
                )

            last_step = step
            time.sleep(0.001)

    except KeyboardInterrupt:
        bridge.close()
        bridge.shm.unlink()


if __name__ == "__main__":
    main()
