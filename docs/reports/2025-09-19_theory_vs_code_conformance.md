Author: Vasiliy Zdanovskiy  
Email: vasilyvz@gmail.com

# Theory vs Implementation Conformance Report (7d-00-15.md → phaze-particles)

Date: 2025-09-19

## Scope
Deep review of the last third of `7d-00-15.md` (sections on (2,4,6)-terms, virial relations, boundary-value formulations, energy identities, and rigorous clarifications) against current code in:
- `phaze_particles/utils/energy_densities.py`
- `phaze_particles/utils/su2_fields.py`
- `phaze_particles/utils/physics.py`
- `phaze_particles/models/proton_integrated.py`

Focus: energy construction (c₂, c₄, c₆), left currents and commutators, baryon density b₀, virial balance, positivity, physical outputs (B, Q, r_E, μ_p, M), and boundary/variational assumptions.

## Executive Summary
- The implementation partially reflects the (2,4,6) energy structure but has critical mismatches:
  - c₄ (commutator) is a placeholder; c₆ (baryon density) is inconsistent across modules.
  - Virial balance is not implemented per theory (−E₂ + E₄ + 3E₆ = 0); current check only tests E₂≈E₄.
  - Energy can become negative (observed), violating positivity required by the theoretical functional.
  - Baryon number (B) path is closer to theory in one place, but c₆ uses a simplified b₀ elsewhere → inconsistency.
  - Magnetic moment is a rough heuristic and not aligned with the theoretical current-based definition.
  - No time dynamics and no boundary/weak-form framework (theory provides rigorous PDE + BC), current code is static.

Priority: unify b₀ definition, implement true commutator traces, fix signs/normalization for positivity, implement virial diagnostics (with c₆), then refine observables.

## Mapping Theory → Code

1) Energy functional (theory):
- E = E₂ + E₄ + E₆ with
  - E₂ ∝ ∫ Tr(Lᵢ Lᵢ)
  - E₄ ∝ ∫ Tr([Lᵢ, Lⱼ]²)
  - E₆ ∝ ∫ b₀², with b₀ = −(1/(24π²)) εⁱʲᵏ Tr(Lᵢ Lⱼ Lₖ)
- Virial: −E₂ + E₄ + 3E₆ = 0 (stationary scale)
- Positivity and coercivity are required.

2) Implementation status:
- E₂: `energy_densities.py` accumulates an l_squared term from Lᵢ components. OK in principle, but sign/normalization must be checked.
- E₄: placeholder `comm_squared = l_squared * 0.1` (incorrect; must be built from true commutators [Lᵢ, Lⱼ]).
- E₆: uses b₀ from `_compute_baryon_density` as a simplified function of l_squared, not the triple-trace expression. In contrast, `BaryonNumberCalculator` implements triple-trace for B. This is inconsistent.
- Virial: `EnergyDensity.check_virial_condition` compares only E₂ vs E₄ (valid only when c₆=0), while default c₆=1.0. Not aligned with theory (must use −E₂ + E₄ + 3E₆≈0).
- Positivity: observed negative total energy and negative mass; this contradicts theory. Likely due to sign conventions in Tr(LᵢLᵢ) accumulation and/or lack of absolute-square structure in traces.
- Boundaries/weak forms: theory formulates PDEs with clamped/Navier and Robin/impedance BCs and energy identities; code uses periodic/implicit grid without BC controls or weak forms. Static only, no δ damping.
- Physical observables: Q≈1 and r_E are reasonable; μ_p≈0 (not OK), M negative (not OK), B close to 0 with small imaginary artifacts in some runs (should be topological and real).

## Detailed Findings and Gaps

1) c₄ (commutator) term
- Theory requires Tr([Lᵢ, Lⱼ]²). Code currently sets `comm_squared = l_squared * 0.1`.
- Impact: wrong energy composition, wrong virial, wrong stability/scale diagnostics.
- Action: implement commutators [Lᵢ, Lⱼ] using existing `SU2FieldOperations._compute_commutator` (already present in `su2_fields.py`) and then compute Tr([Lᵢ, Lⱼ]²) by contracting commutator components similarly to l_squared, with proper real-part extraction and dtype handling.

2) c₆ (baryon density) term
- Theory: b₀ = −(1/(24π²)) ε Tr(Lᵢ Lⱼ Lₖ), then E₆ ∝ ∫ b₀².
- Code: `EnergyDensityCalculator._compute_baryon_density` uses a placeholder aggregate from l_squared; however `BaryonNumberCalculator` correctly computes the triple-trace (matrix product path). 
- Impact: mismatch between c₆ used in energy and B used in observables; energy and virial invalid.
- Action: unify b₀ computation by reusing the triple-trace route for b₀ at the density level (per grid point) and square it for c₆. Ensure real part extraction before squaring and stability with CuPy/NumPy.

3) Virial balance
- Theory: −E₂ + E₄ + 3E₆ = 0 at stationary scale. Current code: `check_virial_condition()` only checks E₂≈E₄ and ignores E₆.
- Impact: the reported “energy balance 0.5” is a mock and not meaningful when c₆≠0.
- Action: implement a virial diagnostic: compute E₂, E₄, E₆ explicitly and report (−E₂ + E₄ + 3E₆)/E_total as the virial residual; expose in reports/CSV.

4) Energy positivity and sign conventions
- Observed negative total energy (and negative “mass”). The theoretical construction requires a positive-definite energy density. Signs inside Tr(LᵢLᵢ) can differ by convention; in Skyrme-like conventions energy density uses positive quadratic forms in currents.
- Action: audit the trace formulas and ensure the energy density is built from positive-definite quadratic combinations (use |·|² where appropriate, consistent with SU(2) algebra), apply Re(·) where needed, and add asserts/statistics to detect negative densities. After fixes, mass should be ≥0.

5) Baryon number, reality, and consistency
- B should be topological and real. We saw small complex parts or values drifting to ~0. Ensure consistent use of Re() and float64 casting in all triple-trace paths. Align b₀ used for c₆ with B computation to eliminate discrepancies.

6) Magnetic moment μ_p
- Current model in `physics.py` is highly simplified (current density heuristic). The theory expects physically grounded currents from SU(2) structures.
- Action: derive μ from SU(2) left currents and appropriate Noether currents (or a validated approximation) consistent with the field direction and profile; at minimum, calibrate a consistent current from the same Lᵢ objects rather than ad-hoc cross products.

7) Boundary conditions and weak formulation
- Theory provides rigorous BCs (clamped/Navier/Robin) and energy identities. Implementation uses a periodic cubic lattice implicitly with no BC control, and no time dynamics (δ-damping absent). This is acceptable for a static prototype, but should be documented and reflected in diagnostics (periodic ≈ torus case in theory).
- Action: state periodic/torus assumption explicitly in logs/reports; consider adding an option to mimic clamped/Navier energetics by suitable padding or derivative handling if later needed.

8) Virial scaling (E(λ))
- Theory uses scaling arguments to determine natural radius/scale. Implementation lacks a scaling diagnostic.
- Action: add a numerical scaling probe: resample the SU(2) field at scaled coordinates (λx), recompute (E₂,E₄,E₆) to estimate dE/dλ|₁ and d²E/dλ²|₁; report stationary/convexity conditions.

## Concrete Action Plan (Code-Level)

Phase 1 — Correctness of energy (highest priority)
1. Implement commutator-based c₄:
   - Use `su2_fields.SU2FieldOperations._compute_commutator` to build [Lᵢ,Lⱼ].
   - Compute Tr([Lᵢ,Lⱼ]²) per point with real-part extraction and float64 dtype.
2. Unify b₀ for c₆:
   - Implement b₀ via triple-trace at density level (reuse logic from `BaryonNumberCalculator._compute_triple_trace`).
   - Set c₆ = c6_coeff * b₀².
3. Fix signs/positivity:
   - Review c₂ accumulation; ensure positive-definite quadratic structure; adopt Re(·) and |·|² where appropriate.
   - Add runtime checks and summary stats for any negative density; fail-fast in DEBUG.

Phase 2 — Virial and diagnostics
4. Implement virial diagnostic: compute residual R_v = (−E₂ + E₄ + 3E₆)/E_total; expose in `EnergyDensity.get_energy_balance()` and in CSV/JSON.
5. Add scaling probe E(λ) around λ=1 for stationary check and convexity.

Phase 3 — Physical observables
6. μ_p from Lᵢ-based currents (or an interim consistent approximation) and unit calibration; target |μ_p−2.793|/2.793 ≤ 5%.
7. Tighten B computation to ensure reality and closeness to 1 within tolerance; validate numerically with grid refinement.

Phase 4 — Documentation and tests
8. Document periodic (torus) assumption mapping to theory’s torus setting; clarify deviations from clamped/Navier.
9. Unit tests: 
   - new tests for commutator traces, b₀ density, virial residual, positivity checks;
   - regression tests for E terms and for μ_p/B tolerances.

## File/Function Pointers (where to implement)
- `phaze_particles/utils/energy_densities.py`
  - Replace placeholder `comm_squared` with true Tr([Lᵢ,Lⱼ]²).
  - Replace `_compute_baryon_density` with triple-trace-based density (consistent with `BaryonNumberCalculator`).
  - Add a virial-residual calculator and expose components (E₂,E₄,E₆) for reporting.
- `phaze_particles/utils/su2_fields.py`
  - Reuse `SU2FieldOperations._compute_commutator` and consider a helper to return Tr([Lᵢ,Lⱼ]²).
- `phaze_particles/utils/physics.py`
  - Rework magnetic moment to use currents derived from Lᵢ (or provide a coherent interim formula and calibration).
- `phaze_particles/models/proton_integrated.py`
  - Report virial residual and energy components; fail/flag when positivity/virial deviates.

## Observed Runtime Evidence from Current Runs
- Q ≈ 1.000 (OK), r_E ≈ 0.881 fm (close), but μ_p ≈ 0.000 μN (not OK).
- Total energy and “mass” negative (−87.5 MeV) — violates positivity/coercivity.
- Complex values appeared in validation until JSON conversion was extended; now converted, but underlying complex traces should be consistently brought to Re(·) for energy and B.

## Acceptance Targets (aligned with DoD)
- After Phase 1–2:
  - Energy ≥ 0 and stable over runs; virial residual |−E₂ + E₄ + 3E₆|/E_total ≤ 1–5% (as configured).
  - c₄ and c₆ implemented per definitions; b₀ density matches triple-trace path.
- After Phase 3:
  - |B−1| ≤ 0.02, |Q−1| ≤ 0.005, |r_E−0.841|/0.841 ≤ 3%, |μ_p−2.793|/2.793 ≤ 5%, |M−938.272|/938.272 ≤ 2% (subject to scale calibration).

## Risks and Notes
- Sign conventions for Lᵢ and traces must be handled carefully to avoid negative energy; test against known analytic profiles.
- Commutator computation is heavier; ensure CuPy support and memory safety (use existing CUDA manager and pools).
- Magnetic moment requires clear current definition; an interim “consistent” approximation may be acceptable before a full Noether current derivation.

## Conclusion
The current implementation captures the high-level structure but diverges on the crucial c₄/c₆ constructions, virial identity, and positivity. The outlined steps will bring the code in line with the rigorous theoretical sections and the project’s DoD (energy balance, physical tolerances), while maintaining the CUDA-aware execution pathway.


