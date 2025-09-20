Author: Vasiliy Zdanovskiy  
Email: vasilyvz@gmail.com  
Date: 2025-09-19

## Theory vs Code Discrepancy Report

### Scope
- Focus on Skyrme model core: E₂, E₄, E₆, virial condition, baryon density, EM charge density (charge radius), magnetic moment.
- Emphasis on the refined equations stated in the second half of 7d-00-15.md (commutator form of E₄, triple-trace baryon density, virial −E₂+E₄+3E₆=0, correct EM current-based charge density, Noether current for μ).

### Agreements (Code matches theory)
- E₄ term implemented via true commutators: Tr([Lᵢ,Lⱼ]²) (see `phaze_particles/utils/energy_densities.py::_compute_commutator_traces`).
- E₆ term uses triple-trace baryon density with correct prefactor: b₀ = −(1/(24π²)) ε^{ijk} Tr(Lᵢ Lⱼ Lₖ).
- Virial condition used in diagnostics: −E₂ + E₄ + 3E₆ = 0, normalized residual reported.
- Positivity: quadratic and commutator forms enforced as positive-definite (abs² of components), avoiding negative total energy.

### Discrepancies (Code diverges from theory)
- E₂ normalization constants: Theory typically uses physical constants (Fπ, e) with standard prefactors (e.g., Fπ²/16), while code uses dimensionless `c2` free parameter. Impact: physical scale must be captured by `energy_scale` or tuned constants; otherwise, mass and radii are not on absolute scale.
- E₄ normalization constants: Similarly, theoretical 1/(32 e²) is replaced with tunable `c4`. Same impact as above.
- Charge density (for charge radius):
  - Theory: Proton EM charge density should be derived from the Noether electromagnetic current; common hedgehog-level relation involves baryon density and isovector current contributions (isoscalar/isovector mix), not just a heuristic function of f(r).
  - Code: Uses ρ_E ∝ sin²(f(r)) normalized to Q=1 (`ChargeDensity.compute_charge_density`). Earlier attempts to switch to b₀-proportional formula were partial and not coupled to left currents. Impact: charge radius can deviate; sensitive to profile choice rather than currents.
- Magnetic moment:
  - Theory: Should be computed from the proper EM current (Noether) with SU(2) structure.
  - Code: Provides an Lᵢ-based surrogate, but the mapping `j_i ~ Tr(σ_i L_j)` is heuristic. Impact: μ_p accuracy limited and may not meet tolerance targets.
- Left current traces for E₂: Code computes Tr(LᵢLᵢ) via sum of abs² matrix elements, which enforces positivity but may differ from exact Hermitian trace conventions by constant factors. Impact: absorbed by `c2` tuning, but should be documented.

### Recommendations
- Keep tunable (`c2,c4,c6`) but document mapping to physical (Fπ,e) and expose a physical scaling to align absolute mass and radius without ad-hoc tuning.
- For charge radius, allow a current-based path: compute EM charge density from left currents (baryon density plus isovector correction) when available; fallback to sin²(f) only for preview modes.
- For magnetic moment, adopt a Noether-current-based expression consistent with SU(2) hedgehog ansatz; deprecate heuristic mapping once verified.

### Implemented Fixes in This Session
- Added an optional current-based charge density path (uses left currents to compute b₀ and uses it as ρ_E baseline) to improve theory conformance for r_E while maintaining the previous fallback.
- Retained verified E₄, E₆, virial and positivity implementations.

### Follow-ups
- Derive and implement the full EM current for hedgehog ansatz; include isovector term weight consistent with proton (p) vs neutron (n).
- Calibrate (`energy_scale`, `c2,c4,c6`) to physical (Fπ,e) using test cases and record mappings.

