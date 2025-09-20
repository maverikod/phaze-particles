#!/usr/bin/env python3
"""
Numerical shell-by-shell magnetic moment analysis (no zoning assumptions).

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import argparse
import csv
import datetime as dt
from typing import Tuple

import numpy as np

from phaze_particles.models.proton_integrated import ModelConfig, ProtonModel


def to_float(x):
    return float(x.get() if hasattr(x, "get") else x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shell-wise μ analysis with circumferential FFT")
    p.add_argument("--grid-size", type=int, default=48)
    p.add_argument("--box-size", type=float, default=4.0)
    p.add_argument("--torus", choices=["120deg", "clover", "cartesian"], default="120deg")
    p.add_argument("--r-scale", type=float, default=0.6)
    # Phase circumferential analysis factors (optional)
    p.add_argument("--phase-density-factor", type=float, default=1.0)
    p.add_argument("--phase-velocity-factor", type=float, default=1.0)
    # Envelope parameters (used by μ model), can be tuned externally
    p.add_argument("--env-shell-frac", type=float, default=0.8)
    p.add_argument("--env-width-frac", type=float, default=0.5)
    p.add_argument("--env-pos-weight", type=float, default=0.75)
    p.add_argument("--env-neg-weight", type=float, default=0.25)
    p.add_argument("--bins", type=int, default=64)
    p.add_argument("--out", type=str, default="results/proton/static")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def compute_phase_vars(xp, field) -> Tuple[object, object]:
    u00 = xp.asarray(field.u_00)
    u11 = xp.asarray(field.u_11)
    a0 = xp.clip(xp.real(0.5 * (u00 + u11)), -1.0, 1.0)
    f = xp.arccos(a0)
    return f, xp.abs(xp.imag(u00))


def main() -> None:
    args = parse_args()

    cfg = ModelConfig(
        grid_size=args.grid_size,
        box_size=args.box_size,
        torus_config=args.torus,
        r_scale=args.r_scale,
        phase_density_factor=args.phase_density_factor,
        phase_velocity_factor=args.phase_velocity_factor,
    )
    model = ProtonModel(cfg)
    model.create_geometry()
    model.build_fields()
    model.calculate_energy()

    # Set envelope params for μ
    mm = model.physics_calculator.magnetic_calculator
    mm.env_shell_fraction = args.env_shell_frac
    mm.env_width_fraction = args.env_width_frac
    mm.env_pos_weight = args.env_pos_weight
    mm.env_neg_weight = args.env_neg_weight

    model.calculate_physics()
    pq = model.physical_quantities

    # Backend and coordinates
    backend = model.physics_calculator.magnetic_calculator.backend
    xp = backend.get_array_module() if backend else np
    N = model.config.grid_size
    L = model.config.box_size
    dx = L / N
    lin = backend.linspace(-L / 2 + 0.5 * dx, L / 2 - 0.5 * dx, N) if backend else np.linspace(-L/2+0.5*dx, L/2-0.5*dx, N)
    X, Y, Z = (backend.meshgrid(lin, lin, lin, indexing="ij") if backend else np.meshgrid(lin, lin, lin, indexing="ij"))
    R = xp.sqrt(X * X + Y * Y + Z * Z)

    # Phase variables and energy
    f, amp = compute_phase_vars(xp, model.su2_field)
    df_dx = backend.gradient(f, dx, axis=0) if backend else np.gradient(f, dx, axis=0)
    df_dy = backend.gradient(f, dx, axis=1) if backend else np.gradient(f, dx, axis=1)
    df_dz = backend.gradient(f, dx, axis=2) if backend else np.gradient(f, dx, axis=2)
    phase_energy = df_dx * df_dx + df_dy * df_dy + df_dz * df_dz

    # Robust r_core from phase_energy(r) profile
    R_np = R.get() if hasattr(R, "get") else R
    pe_np = phase_energy.get() if hasattr(phase_energy, "get") else phase_energy
    r_flat = R_np.ravel(); e_flat = pe_np.ravel()
    nb = args.bins
    r_edges = np.linspace(0.0, float(R_np.max()) + 1e-12, nb + 1)
    idx = np.digitize(r_flat, r_edges) - 1; idx = np.clip(idx, 0, nb - 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    e_of_r = np.zeros(nb); cnt = np.zeros(nb)
    np.add.at(e_of_r, idx, e_flat); np.add.at(cnt, idx, 1)
    e_of_r = np.where(cnt > 0, e_of_r / cnt, 0.0)
    core_bin = int(np.argmax(e_of_r)); r_core = float(r_centers[core_bin])

    # Envelope density ρ (same construction as in physics.MagneticMomentCalculator)
    shell_mask = (R >= r_core) & (R <= r_core + float(args.env_shell_frac) * r_core)
    shell_pe = pe_np[(R_np >= r_core) & (R_np <= r_core + 0.6 * r_core)]
    shell_med = float(np.median(shell_pe)) if shell_pe.size > 0 else float(np.median(pe_np))
    thr_eng = 0.6 * shell_med
    fluct = phase_energy - thr_eng
    pos = xp.maximum(fluct, 0.0)
    neg = xp.maximum(-fluct, 0.0)
    rho = xp.where(shell_mask, float(args.env_pos_weight) * pos - float(args.env_neg_weight) * neg, xp.float64(0.0))
    rho = rho * (1.0 + 0.5 * xp.tanh((amp - xp.mean(amp)) / (amp.std() + 1e-12)))
    width = xp.float64(max(float(args.env_width_frac) * r_core, 1e-12))
    damp = xp.exp(-((R - r_core) / width) ** 2)
    rho = rho * damp

    # Tangential velocity (axis z)
    Rxy = xp.sqrt(X * X + Y * Y) + 1e-12
    vx, vy = -Y / Rxy, X / Rxy
    jx, jy = rho * vx, rho * vy
    mudz = 0.5 * (X * jy - Y * jx)

    # Convert to numpy for binning and integration
    mudz_np = mudz.get() if hasattr(mudz, "get") else mudz
    vol = dx ** 3

    # μ contribution per shell
    mu_shell = np.zeros(nb)
    for i in range(nb):
        m = (R_np >= r_edges[i]) & (R_np < r_edges[i + 1])
        mu_shell[i] = float(np.sum(mudz_np[m]) * vol)

    # Physical units scaling (same factor as in calculator)
    from phaze_particles.utils.mathematical_foundations import PhysicalConstants
    factor = 2.0 * PhysicalConstants.PROTON_MASS_MEV / PhysicalConstants.HBAR_C
    mu_shell *= factor
    mu_cum = np.cumsum(mu_shell)

    # Alternative honest variants (no assumptions):
    # 1) plain: rho_plain ∝ phase_energy (no thresholds/damping)
    jx_p, jy_p = phase_energy * vx, phase_energy * vy
    mudz_p = 0.5 * (X * jy_p - Y * jx_p)
    mudz_p_np = mudz_p.get() if hasattr(mudz_p, "get") else mudz_p
    mu_shell_plain = np.zeros(nb)
    for i in range(nb):
        m = (R_np >= r_edges[i]) & (R_np < r_edges[i + 1])
        mu_shell_plain[i] = float(np.sum(mudz_p_np[m]) * vol)
    mu_shell_plain *= factor
    mu_plain_total = float(np.sum(mu_shell_plain))

    # 2) compression-driven: rho_comp ∝ phase_energy where Laplacian(f)<0
    # Compute Laplacian of f
    d2f_x = backend.gradient(df_dx, dx, axis=0) if backend else np.gradient(df_dx, dx, axis=0)
    d2f_y = backend.gradient(df_dy, dx, axis=1) if backend else np.gradient(df_dy, dx, axis=1)
    d2f_z = backend.gradient(df_dz, dx, axis=2) if backend else np.gradient(df_dz, dx, axis=2)
    lap_f = d2f_x + d2f_y + d2f_z
    comp_mask = (lap_f < 0)
    rho_comp = phase_energy * comp_mask
    jx_c, jy_c = rho_comp * vx, rho_comp * vy
    mudz_c = 0.5 * (X * jy_c - Y * jx_c)
    mudz_c_np = mudz_c.get() if hasattr(mudz_c, "get") else mudz_c
    mu_shell_comp = np.zeros(nb)
    for i in range(nb):
        m = (R_np >= r_edges[i]) & (R_np < r_edges[i + 1])
        mu_shell_comp[i] = float(np.sum(mudz_c_np[m]) * vol)
    mu_shell_comp *= factor
    mu_comp_total = float(np.sum(mu_shell_comp))

    # Equatorial circumferential FFT and interference metrics per ring (|Z|<dx)
    plane = np.abs((Z.get() if hasattr(Z, "get") else Z)) <= (dx * 0.6)
    u00 = (model.su2_field.u_00.get() if hasattr(model.su2_field.u_00, "get") else model.su2_field.u_00)
    u11 = (model.su2_field.u_11.get() if hasattr(model.su2_field.u_11, "get") else model.su2_field.u_11)
    a0_np = np.clip(np.real(0.5 * (u00 + u11)), -1.0, 1.0)
    phase_angle_np = np.arccos(a0_np)
    m_fft = np.zeros(nb, dtype=int)
    interfer_std = np.zeros(nb)
    S_balance = np.zeros(nb)

    for i in range(nb):
        r_lo, r_hi = r_edges[i], r_edges[i + 1]
        ring = plane & (R_np >= r_lo) & (R_np < r_hi)
        if np.count_nonzero(ring) < 50:
            continue
        phi = np.arctan2((Y.get() if hasattr(Y, "get") else Y)[ring], (X.get() if hasattr(X, "get") else X)[ring])
        sig = phase_angle_np[ring]
        order = np.argsort(phi)
        phi_sorted = phi[order]
        sig_sorted = sig[order]
        bins = 256
        phi_bins = np.linspace(-np.pi, np.pi, bins + 1)
        idxs = np.digitize(phi_sorted, phi_bins) - 1
        idxs = np.clip(idxs, 0, bins - 1)
        sig_binned = np.zeros(bins)
        cntb = np.zeros(bins)
        for k, s in zip(idxs, sig_sorted):
            sig_binned[k] += float(s)
            cntb[k] += 1
        valid = cntb > 0
        if not np.any(valid):
            continue
        sig_binned[valid] /= cntb[valid]
        sig_binned = np.interp(np.arange(bins), np.where(valid)[0], sig_binned[valid])
        sig_zero = sig_binned - np.mean(sig_binned)
        spec = np.fft.rfft(sig_zero)
        mag = np.abs(spec)
        if mag.size > 1:
            m_fft[i] = int(1 + np.argmax(mag[1:]))
        # Interference strength: std along φ
        interfer_std[i] = float(np.std(sig_zero))
        # Curvature balance S = w_pos - w_neg (peaks vs troughs)
        dphi = phi_bins[1] - phi_bins[0]
        dsig = np.gradient(sig_binned, dphi)
        d2sig = np.gradient(dsig, dphi)
        neg_curv = d2sig < 0
        pos_curv = d2sig > 0
        w_pos = float(np.mean(neg_curv))
        w_neg = float(np.mean(pos_curv))
        S_balance[i] = w_pos - w_neg

    # Write CSV (UTF-8 with BOM)
    ts = dt.datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    short = f"-grid{args.grid_size}-box{args.box_size}-ring-mu"
    out_path = f"{args.out}/{short}-{ts}.csv"

    import os
    os.makedirs(args.out, exist_ok=True)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp", "grid_size", "box_size", "torus", "r_scale",
            "phase_density_factor", "phase_velocity_factor",
            "env_shell_frac", "env_width_frac", "env_pos_weight", "env_neg_weight",
        ])
        w.writerow([
            ts, args.grid_size, args.box_size, args.torus, args.r_scale,
            args.phase_density_factor, args.phase_velocity_factor,
            args.env_shell_frac, args.env_width_frac, args.env_pos_weight, args.env_neg_weight,
        ])
        w.writerow(["r_center", "mu_shell", "mu_cumulative", "m_fft_equatorial", "interference_std_phi", "S_balance"])
        for i in range(nb):
            w.writerow([r_centers[i], mu_shell[i], mu_cum[i], m_fft[i], interfer_std[i], S_balance[i]])

    # Print summary
    print("=== MU RING ANALYSIS SUMMARY ===")
    print(f"Output CSV: {out_path}")
    print(f"Model μ (physics calc): {to_float(pq.magnetic_moment):.6f} μN")
    print(f"Shell-summed μ:         {mu_cum[-1]:.6f} μN")
    print(f"Q={to_float(pq.electric_charge):.6f}, B={to_float(pq.baryon_number):.6f}")
    print(f"Plain(ρ∝E_phase) μ:     {mu_plain_total:.6f} μN | Compression(Δf<0) μ: {mu_comp_total:.6f} μN")


if __name__ == "__main__":
    main()


