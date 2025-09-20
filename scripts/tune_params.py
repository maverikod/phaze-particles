#!/usr/bin/env python3
"""
Parameter tuning utility for ProtonModel.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com

This script sweeps over r_scale (and optionally Skyrme constants later)
to minimize a composite objective built from:
 - E2/E_total balance target ~ 0.5
 - Virial residual target ~ 0
 - Baryon number target ~ 1.0
 - Charge radius target ~ 0.841 fm
 - Magnetic moment target ~ 2.793 μN
 - Mass target ~ 938.272 MeV

Outputs a CSV with all trials and prints the best configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

import numpy as np

# Ensure package import path is correct when run directly
try:
    from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Import error. Run with PYTHONPATH=. Active venv may be required. Details: {e}")


def _to_host(x: Any) -> Any:
    """Return NumPy array or Python scalar from possible CuPy/NumPy array."""
    if hasattr(x, "get"):
        return x.get()
    return x


def evaluate_once(
    grid: int,
    box: float,
    config_type: str,
    r_scale: float,
    c2: float,
    c4: float,
    c6: float,
    skip_optimize: bool = True,
) -> Optional[Dict[str, float]]:
    """Build model, compute energy and physics, return metrics or None on failure."""
    try:
        cfg = ModelConfig(
            grid_size=grid,
            box_size=box,
            torus_config=config_type,
            r_scale=float(r_scale),
            c2=float(c2),
            c4=float(c4),
            c6=float(c6),
            auto_tune_baryon=False,
            validation_enabled=False,
            max_iterations=200,
        )
        model = ProtonModel(cfg)
        if not (model.create_geometry() and model.build_fields() and model.calculate_energy()):
            return None
        # energy integrals
        dx = box / grid
        c2_term = _to_host(model.energy_density.c2_term)
        c4_term = _to_host(model.energy_density.c4_term)
        c6_term = _to_host(model.energy_density.c6_term)
        e2 = float(np.sum(c2_term) * dx ** 3)
        e4 = float(np.sum(c4_term) * dx ** 3)
        e6 = float(np.sum(c6_term) * dx ** 3)
        et = e2 + e4 + e6
        e2_ratio = (e2 / et) if et > 0 else 0.0
        e4_ratio = (e4 / et) if et > 0 else 0.0
        virial = ((-e2 + e4 + 3.0 * e6) / et) if et > 0 else 0.0

        if not model.calculate_physics():
            return None

        mass = float(model.physical_quantities.mass)
        rE = float(model.physical_quantities.charge_radius)
        mu = float(model.physical_quantities.magnetic_moment)
        B = float(model.physical_quantities.baryon_number)

        return {
            "r_scale": float(r_scale),
            "c2": float(c2),
            "c4": float(c4),
            "c6": float(c6),
            "E2": e2,
            "E4": e4,
            "E6": e6,
            "E_total": et,
            "E2_ratio": e2_ratio,
            "E4_ratio": e4_ratio,
            "virial": virial,
            "mass": mass,
            "rE": rE,
            "mu": mu,
            "B": B,
        }
    except Exception:
        return None


def objective(
    m: Dict[str, float],
    w_e: float,
    w_v: float,
    w_B: float,
    w_r: float,
    w_mu: float,
    w_M: float,
) -> float:
    """Compute scalar objective (lower is better)."""
    # Targets
    t_rE = 0.841
    t_mu = 2.793
    t_M = 938.272
    # Terms
    e_term = abs(m["E2_ratio"] - 0.5) + abs(m["E4_ratio"] - 0.5)
    v_term = abs(m["virial"])  # normalized by Et already
    B_term = abs(m["B"] - 1.0)
    r_term = abs(m["rE"] - t_rE) / t_rE if t_rE > 0 else 0.0
    mu_term = abs(m["mu"] - t_mu) / t_mu if t_mu > 0 else 0.0
    M_term = abs(m["mass"] - t_M) / t_M if t_M > 0 else 0.0
    return (
        w_e * e_term + w_v * v_term + w_B * B_term + w_r * r_term + w_mu * mu_term + w_M * M_term
    )


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_csv(rows: list[Dict[str, float]], out_dir: str, short: str) -> str:
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    filename = f"{short}-{ts}.csv"
    path = os.path.join(out_dir, filename)
    fieldnames = [
        "r_scale",
        "c2",
        "c4",
        "c6",
        "E2",
        "E4",
        "E6",
        "E_total",
        "E2_ratio",
        "E4_ratio",
        "virial",
        "mass",
        "rE",
        "mu",
        "B",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Tune r_scale (and later c2/c4) for ProtonModel")
    p.add_argument("--grid-size", type=int, default=48, help="Grid size (N)")
    p.add_argument("--box-size", type=float, default=4.0, help="Box size (fm)")
    p.add_argument("--config-type", type=str, default="120deg", choices=["120deg", "clover", "cartesian"], help="Torus configuration")
    p.add_argument("--r-scale-start", type=float, default=0.6, help="Start r_scale")
    p.add_argument("--r-scale-end", type=float, default=1.6, help="End r_scale")
    p.add_argument("--r-scale-steps", type=int, default=11, help="Number of r_scale steps")
    p.add_argument("--c2", type=float, default=1.0, help="Skyrme c2")
    p.add_argument("--c4", type=float, default=1.0, help="Skyrme c4")
    p.add_argument("--c6", type=float, default=1.0, help="Skyrme c6")
    p.add_argument("--out-dir", type=str, default="results/proton/tuning/", help="Output directory for CSV")
    p.add_argument("--weights", type=str, default="{\"e\":1.0,\"v\":0.5,\"B\":0.8,\"r\":0.6,\"mu\":0.3,\"M\":0.2}", help="JSON dict of weights: e,v,B,r,mu,M")
    args = p.parse_args()

    try:
        w = json.loads(args.weights)
        w_e = float(w.get("e", 1.0))
        w_v = float(w.get("v", 0.5))
        w_B = float(w.get("B", 0.8))
        w_r = float(w.get("r", 0.6))
        w_mu = float(w.get("mu", 0.3))
        w_M = float(w.get("M", 0.2))
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Invalid --weights JSON: {exc}")

    r_values = np.linspace(args.r_scale_start, args.r_scale_end, args.r_scale_steps)
    rows: list[Dict[str, float]] = []
    best: Optional[Tuple[float, Dict[str, float]]] = None

    for r in r_values:
        res = evaluate_once(
            grid=args.grid_size,
            box=args.box_size,
            config_type=args.config_type,
            r_scale=float(r),
            c2=args.c2,
            c4=args.c4,
            c6=args.c6,
            skip_optimize=True,
        )
        if res is None:
            continue
        rows.append(res)
        score = objective(res, w_e, w_v, w_B, w_r, w_mu, w_M)
        if best is None or score < best[0]:
            best = (score, res)
        print({
            "r_scale": round(res["r_scale"], 4),
            "E2_ratio": round(res["E2_ratio"], 4),
            "virial": round(res["virial"], 4),
            "mass": round(res["mass"], 3),
            "rE": round(res["rE"], 4),
            "mu": round(res["mu"], 4),
            "B": round(res["B"], 4),
            "score": round(score, 6),
        })

    if not rows:
        raise SystemExit("No successful evaluations. Check configuration or environment.")

    short = f"-grid{args.grid_size}-box{args.box_size}-{args.config_type}"
    csv_path = save_csv(rows, args.out_dir, short)

    assert best is not None
    print("\nBEST by objective:")
    print(json.dumps(best[1], indent=2))
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":  # pragma: no cover
    main()


