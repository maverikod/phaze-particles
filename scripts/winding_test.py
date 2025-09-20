#!/usr/bin/env python3
"""
Winding (Baryon Number) Test Script

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com

This script constructs a hedgehog SU(2) configuration using phase variables
U = cos(f) I + i sin(f) (n̂ · σ), with n̂ = r̂ and f(r) monotonic: f(0)=π, f(∞)=0.

It then computes the baryon number B in two ways:
1) Phase-based density on a 3D grid:
   b0(x) = (1/(2π²)) sin²(f) · ε^{ijk} (∂_i f) [ n̂ · (∂_j n̂ × ∂_k n̂) ]
   B = ∫ b0 d³x ≈ sum(b0) · dx³

2) 1D radial identity check for the hedgehog ansatz:
   B = (1/π) [ f - (1/2) sin(2f) ]_{r=0}^{∞}

If B ≈ 1 on the grid, the winding is correct and can be integrated into the model.
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import numpy as np


def build_grid(grid_size: int, box_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build a cubic grid and return coordinate arrays and spacing.

    Args:
        grid_size: Number of grid points along each axis (N)
        box_size: Physical box size (L) in the same units (e.g., fm)

    Returns:
        (X, Y, Z, dx): coordinate arrays centered at 0 and grid spacing
    """
    # Coordinates centered at zero, uniform spacing
    dx = box_size / grid_size
    half = box_size / 2.0
    axis = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, grid_size, dtype=np.float64)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    return X, Y, Z, dx


def hedgehog_profile(r: np.ndarray, r0: float, power: float = 2.0) -> np.ndarray:
    """Hedgehog profile f(r) = 2 arctan((r0/r)^power) with f(0)=π, f(∞)=0.

    Args:
        r: radial distances
        r0: scale parameter (core size)
        power: power in the rational ansatz (default 2)

    Returns:
        f(r) array
    """
    eps = 1e-12
    ratio = (r0 / np.maximum(r, eps)) ** power
    return 2.0 * np.arctan(ratio)


def compute_phase_density(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, f: np.ndarray, dx: float
) -> Tuple[np.ndarray, float]:
    """Compute phase-based baryon density b0 and its integral B on a 3D grid.

    b0(x) = (1/(2π²)) sin²(f) · ε^{ijk} (∂_i f) [ n̂ · (∂_j n̂ × ∂_k n̂) ]

    Args:
        X, Y, Z: coordinate arrays
        f: phase profile array
        dx: grid spacing

    Returns:
        (b0, B): density array and integrated baryon number
    """
    # Radial unit vector n̂ = r̂
    r = np.sqrt(X * X + Y * Y + Z * Z)
    eps = 1e-14
    inv_r = 1.0 / np.maximum(r, eps)
    n_x = np.where(r < eps, 0.0, X * inv_r)
    n_y = np.where(r < eps, 0.0, Y * inv_r)
    n_z = np.where(r < eps, 1.0, Z * inv_r)

    # Gradients
    dfx, dfy, dfz = np.gradient(f, dx, dx, dx, edge_order=2)
    dnxdx, dnxdy, dnxdz = np.gradient(n_x, dx, dx, dx, edge_order=2)
    dnydx, dnydd, dnydz = np.gradient(n_y, dx, dx, dx, edge_order=2)
    dnzdx, dnzdy, dnzdz = np.gradient(n_z, dx, dx, dx, edge_order=2)

    # For each (j,k) compute n · (∂_j n × ∂_k n)
    # j,k in {x(0), y(1), z(2)}
    # Build arrays for derivatives
    dn = {
        0: (dnxdx, dnydx, dnzdx),  # ∂_x n = (∂_x n_x, ∂_x n_y, ∂_x n_z)
        1: (dnxdy, dnydd, dnzdy),  # ∂_y n
        2: (dnxdz, dnydz, dnzdz),  # ∂_z n
    }
    n_vec = (n_x, n_y, n_z)

    def scalar_triple(a: Tuple[np.ndarray, np.ndarray, np.ndarray],
                      b: Tuple[np.ndarray, np.ndarray, np.ndarray],
                      c: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Compute a · (b × c) for 3D vector fields."""
        bx_cy = b[1] * c[2] - b[2] * c[1]
        by_cz = b[2] * c[0] - b[0] * c[2]
        bz_cx = b[0] * c[1] - b[1] * c[0]
        return a[0] * bx_cy + a[1] * by_cz + a[2] * bz_cx

    # ε^{ijk} (∂_i f) [ n · (∂_j n × ∂_k n) ]
    df = {0: dfx, 1: dfy, 2: dfz}
    eps_ijk_sum = (
        + df[0] * scalar_triple(n_vec, dn[1], dn[2])
        + df[1] * scalar_triple(n_vec, dn[2], dn[0])
        + df[2] * scalar_triple(n_vec, dn[0], dn[1])
    )

    # Sign convention: choose orientation so that hedgehog with f(0)=π -> f(∞)=0 gives B=+1
    pref = -1.0 / (2.0 * math.pi * math.pi)
    b0 = pref * (np.sin(f) ** 2) * eps_ijk_sum

    B = float(np.sum(b0) * (dx ** 3))
    return b0, B


def radial_identity_B(f0: float, finf: float) -> float:
    """Radial identity for hedgehog: B = (1/π) [ f - (1/2) sin(2f) ]_{0}^{∞}.

    Args:
        f0: f(0)
        finf: f(∞)

    Returns:
        B value
    """
    def F(val: float) -> float:
        return val - 0.5 * math.sin(2.0 * val)

    return (F(f0) - F(finf)) / math.pi


def main() -> None:
    parser = argparse.ArgumentParser(description="Winding test (baryon number) for hedgehog configuration")
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size N")
    parser.add_argument("--box-size", type=float, default=6.0, help="Box size L")
    parser.add_argument("--r0", type=float, default=1.2, help="Core scale r0")
    parser.add_argument("--power", type=float, default=2.0, help="Profile power in ansatz")
    args = parser.parse_args()

    X, Y, Z, dx = build_grid(args.grid_size, args.box_size)
    r = np.sqrt(X * X + Y * Y + Z * Z)

    # Build hedgehog phase
    f = hedgehog_profile(r, r0=args.r0, power=args.power)

    # 3D phase-based density integration
    b0, B_grid = compute_phase_density(X, Y, Z, f, dx)

    # Radial identity check
    B_radial = radial_identity_B(f0=math.pi, finf=0.0)

    # Report
    print("=== Winding Test (Hedgehog) ===")
    print(f"Grid: {args.grid_size}^3, Box: {args.box_size}, dx: {dx:.4f}")
    print(f"Profile: f(r)=2 arctan((r0/r)^p), r0={args.r0}, p={args.power}")
    print("")
    print("Phase-density integration (3D grid):")
    print(f"  B_grid = {B_grid:.6f}")
    print(f"  b0 stats: min={b0.min():.3e}, max={b0.max():.3e}, mean={b0.mean():.3e}")
    print("")
    print("Radial identity (analytic for hedgehog):")
    print(f"  B_radial = {B_radial:.6f} (expected 1.000000)")

    # Simple success heuristic
    if abs(B_grid - 1.0) < 0.05:
        print("\nResult: SUCCESS (correct winding on grid)")
    else:
        print("\nResult: NEEDS TUNING (increase grid, adjust r0/box, or gradients)")


if __name__ == "__main__":
    main()


