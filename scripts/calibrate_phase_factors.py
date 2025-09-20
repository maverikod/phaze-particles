#!/usr/bin/env python3
"""
Phase factors and envelope calibration with progress indicator.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import argparse
import itertools
import sys
import time
from typing import Iterable, List, Tuple

from phaze_particles.models.proton_integrated import ModelConfig, ProtonModel


def to_float(x):
    return float(x.get() if hasattr(x, "get") else x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate phase density/velocity factors and envelope params with progress"
    )
    p.add_argument("--mode", choices=["phase", "envelope"], default="phase")
    p.add_argument("--grid-size", type=int, default=48)
    p.add_argument("--box-size", type=float, default=4.0)
    p.add_argument("--torus", choices=["120deg", "clover", "cartesian"], default="120deg")
    p.add_argument("--r-scale", type=float, default=0.6)

    # Phase factor ranges
    p.add_argument("--dens", type=str, default="0.8,1.0,1.2,1.5",
                   help="Comma-separated phase_density_factor values")
    p.add_argument("--vel", type=str, default="0.6,0.8,1.0,1.2",
                   help="Comma-separated phase_velocity_factor values")

    # Envelope ranges
    p.add_argument("--shell", type=str, default="0.8,1.0,1.4,2.0,2.4",
                   help="Comma-separated env_shell_fraction values")
    p.add_argument("--width", type=str, default="0.6,0.8,1.0,1.2,1.6",
                   help="Comma-separated env_width_fraction values")
    p.add_argument("--pos", type=str, default="0.85,0.9,0.95,0.98,1.0",
                   help="Comma-separated env_pos_weight values")
    p.add_argument("--neg", type=str, default="0.0,0.05,0.1,0.15,0.2",
                   help="Comma-separated env_neg_weight values")

    p.add_argument("--quiet", action="store_true", help="Less verbose output")
    return p.parse_args()


def progress(i: int, n: int, msg: str) -> None:
    pct = 100.0 * (i + 1) / max(1, n)
    sys.stdout.write(f"[{i+1:4d}/{n:4d}] {pct:6.2f}% | {msg}\n")
    sys.stdout.flush()


def score_phase(model: ProtonModel) -> Tuple[float, float, float, float]:
    r = model.analyze_phase_tails()
    modes = r.circumferential_modes or []
    mismatch = 0.0
    for cm in modes:
        m_obs = cm.get("m_fft") or cm.get("m") or 0
        m_pred = cm.get("m_pred") or 0
        mismatch += abs(int(m_obs) - int(m_pred))
    pq = model.physical_quantities
    mu = to_float(pq.magnetic_moment)
    q = to_float(pq.electric_charge)
    b = to_float(pq.baryon_number)
    # Composite: prioritize B,Q → then μ → then mode mismatch
    score = 50.0 * abs(q - 1.0) + 50.0 * abs(b - 1.0) + 5.0 * abs(mu - 2.793) + mismatch
    return score, mu, q, b


def run_phase(args: argparse.Namespace) -> None:
    dens_vals = [float(x) for x in args.dens.split(",") if x]
    vel_vals = [float(x) for x in args.vel.split(",") if x]
    grid = list(itertools.product(dens_vals, vel_vals))
    total = len(grid)
    best = None
    best_score = 1e18

    t0 = time.time()
    for i, (d, v) in enumerate(grid):
        cfg = ModelConfig(
            grid_size=args.grid_size,
            box_size=args.box_size,
            torus_config=args.torus,
            r_scale=args.r_scale,
            phase_density_factor=d,
            phase_velocity_factor=v,
        )
        m = ProtonModel(cfg)
        m.create_geometry()
        m.build_fields()
        m.calculate_energy()
        m.calculate_physics()

        score, mu, q, b = score_phase(m)
        msg = f"phase_density={d:.3f} phase_velocity={v:.3f} | mu={mu:.3f} Q={q:.3f} B={b:.3f} score={score:.3f}"
        progress(i, total, msg)

        if score < best_score:
            best_score = score
            best = (d, v, mu, q, b, score)

    dt = time.time() - t0
    print("\nBest phase factors:")
    print(
        f"density={best[0]:.3f} velocity={best[1]:.3f} | mu={best[2]:.3f} Q={best[3]:.3f} B={best[4]:.3f} score={best[5]:.3f}"
    )
    print(f"Elapsed: {dt:.2f}s\n")


def score_env(model: ProtonModel) -> Tuple[float, float, float, float]:
    pq = model.physical_quantities
    mu = to_float(pq.magnetic_moment)
    q = to_float(pq.electric_charge)
    b = to_float(pq.baryon_number)
    score = abs(mu - 2.793) + 10.0 * abs(q - 1.0) + 10.0 * abs(b - 1.0)
    return score, mu, q, b


def run_envelope(args: argparse.Namespace) -> None:
    dens_vals = [float(x) for x in args.dens.split(",") if x]
    vel_vals = [float(x) for x in args.vel.split(",") if x]
    # Fix phase factors at the best rough guess (choose middle)
    d = dens_vals[len(dens_vals) // 2]
    v = vel_vals[len(vel_vals) // 2]

    shell_vals = [float(x) for x in args.shell.split(",") if x]
    width_vals = [float(x) for x in args.width.split(",") if x]
    pos_vals = [float(x) for x in args.pos.split(",") if x]
    neg_vals = [float(x) for x in args.neg.split(",") if x]

    grid = list(itertools.product(shell_vals, width_vals, pos_vals, neg_vals))
    total = len(grid)
    best = None
    best_score = 1e18

    t0 = time.time()
    for i, (sf, wf, pw, nw) in enumerate(grid):
        cfg = ModelConfig(
            grid_size=args.grid_size,
            box_size=args.box_size,
            torus_config=args.torus,
            r_scale=args.r_scale,
            phase_density_factor=d,
            phase_velocity_factor=v,
        )
        m = ProtonModel(cfg)
        m.create_geometry()
        m.build_fields()
        m.calculate_energy()
        mm = m.physics_calculator.magnetic_calculator
        mm.env_shell_fraction = sf
        mm.env_width_fraction = wf
        mm.env_pos_weight = pw
        mm.env_neg_weight = nw
        m.calculate_physics()

        score, mu, q, b = score_env(m)
        msg = (
            f"shell={sf:.2f} width={wf:.2f} pos={pw:.2f} neg={nw:.2f} | "
            f"mu={mu:.3f} Q={q:.3f} B={b:.3f} score={score:.3f}"
        )
        progress(i, total, msg)

        if score < best_score:
            best_score = score
            best = (sf, wf, pw, nw, mu, q, b, score)

    dt = time.time() - t0
    print("\nBest envelope params:")
    print(
        f"shell={best[0]:.2f} width={best[1]:.2f} pos={best[2]:.2f} neg={best[3]:.2f} | "
        f"mu={best[4]:.3f} Q={best[5]:.3f} B={best[6]:.3f} score={best[7]:.3f}"
    )
    print(f"Elapsed: {dt:.2f}s\n")


def main() -> None:
    args = parse_args()
    if args.mode == "phase":
        run_phase(args)
    else:
        run_envelope(args)


if __name__ == "__main__":
    main()


