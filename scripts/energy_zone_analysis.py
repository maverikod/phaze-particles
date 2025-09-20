#!/usr/bin/env python3
"""
Energy zone analysis (core / transition / tail) with virial per zone

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com

This script builds the proton model, determines a core radius from the
radial profile of phase energy, splits the space into zones:
  - core:      r < R_core
  - transition:r in [R_core, R_core + Δ]
  - tail:      r >= R_core + Δ

and reports E2, E4, E6, totals, fractions and virial residual per zone.

Usage:
  python scripts/energy_zone_analysis.py --grid-size 48 --box-size 4.0 \
      --torus-config 120deg --delta-factor 0.3
"""

from __future__ import annotations

import argparse
import numpy as np


def to_np(a):
    return a.get() if hasattr(a, "get") else a


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy zone analysis with virial per zone")
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--box-size", type=float, default=4.0)
    parser.add_argument("--torus-config", type=str, default="120deg")
    parser.add_argument("--delta-factor", type=float, default=0.3, help="Δ as fraction of R_core")
    args = parser.parse_args()

    # Lazy import to use project environment
    from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig

    cfg = ModelConfig(
        grid_size=args.grid_size,
        box_size=args.box_size,
        torus_config=args.torus_config,
    )

    model = ProtonModel(cfg)
    assert model.create_geometry(), "Failed to create geometry"
    assert model.build_fields(), "Failed to build fields"
    assert model.calculate_energy(), "Failed to calculate energy"
    model.calculate_physics()

    # Energy density components (NumPy)
    c2 = to_np(model.energy_density.c2_term)
    c4 = to_np(model.energy_density.c4_term)
    c6 = to_np(model.energy_density.c6_term)

    # Coordinates and radii
    N, L = cfg.grid_size, cfg.box_size
    dx = L / N
    axis = np.linspace(-L / 2 + 0.5 * dx, L / 2 - 0.5 * dx, N)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)

    # Phase variables to estimate core radius via phase energy |∇f|^2
    U = model.su2_field
    u00 = to_np(U.u_00)
    u11 = to_np(U.u_11)
    a0 = np.clip(np.real(0.5 * (u00 + u11)), -1.0, 1.0)
    f = np.arccos(a0)
    fx = np.gradient(f, dx, axis=0)
    fy = np.gradient(f, dx, axis=1)
    fz = np.gradient(f, dx, axis=2)
    e_phi = fx * fx + fy * fy + fz * fz

    # Radial profile of phase energy (bin-averaged)
    r_flat = R.ravel()
    e_flat = e_phi.ravel()
    nbins = 64
    bins = np.linspace(0.0, R.max() + 1e-12, nbins + 1)
    idx = np.digitize(r_flat, bins) - 1
    idx = np.clip(idx, 0, nbins - 1)
    rad = 0.5 * (bins[:-1] + bins[1:])
    e_of_r = np.zeros(nbins)
    cnt = np.zeros(nbins)
    np.add.at(e_of_r, idx, e_flat)
    np.add.at(cnt, idx, 1)
    e_of_r = np.where(cnt > 0, e_of_r / cnt, 0.0)
    core_bin = int(np.argmax(e_of_r))
    r_core = float(rad[core_bin])
    delta = max(1e-6, float(args.delta_factor * r_core))

    core_mask = R < r_core
    trans_mask = (R >= r_core) & (R < r_core + delta)
    tail_mask = R >= (r_core + delta)

    vol = dx ** 3
    e2_tot = float(c2.sum() * vol)
    e4_tot = float(c4.sum() * vol)
    e6_tot = float(c6.sum() * vol)
    etot = e2_tot + e4_tot + e6_tot

    def zone_report(mask: np.ndarray) -> dict:
        e2 = float(c2[mask].sum() * vol)
        e4 = float(c4[mask].sum() * vol)
        e6 = float(c6[mask].sum() * vol)
        et = e2 + e4 + e6
        vir = (-e2 + e4 + 3.0 * e6) / et if et > 0 else float("nan")
        frac = et / etot if etot > 0 else 0.0
        return {"E2": e2, "E4": e4, "E6": e6, "Et": et, "virial": vir, "frac": frac}

    zones = {
        "core": zone_report(core_mask),
        "transition": zone_report(trans_mask),
        "tail": zone_report(tail_mask),
    }

    out = {
        "grid": N,
        "box": L,
        "R_core": r_core,
        "Delta": delta,
        "global": {"E2": e2_tot, "E4": e4_tot, "E6": e6_tot, "Etot": etot},
        "zones": zones,
        "baryon_number": float(model.physical_quantities.baryon_number),
        "charge_radius": float(model.physical_quantities.charge_radius),
        "mass": float(model.physical_quantities.mass),
    }

    # Pretty print
    import json

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


