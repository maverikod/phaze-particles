# Proton Static Architecture Fix Plan (2025-09-19)

Author: Vasiliy Zdanovskiy  
Email: vasilyvz@gmail.com

## 1. Current Issues Overview

- Missing `backend` propagation in `ProtonModel` and dependent calculators (e.g., `PhysicalQuantitiesCalculator`).
- Inconsistent contract for `field_direction` between `TorusGeometries` and `SU2FieldBuilder`.
- `EnergyDensityCalculator.calculate_energy_density` used mock derivatives instead of real pipeline.
- `left_currents` structure mismatch: expected matrix components (`l_00`, `l_01`, `l_10`, `l_11`) but provided arrays.
- `calculate_quantities` used mock `profile` and derivatives; lacked proper inputs.
- Logging includes debug traces mixed into CLI flow; needs unified logging.
- CLI run fails after fields due to physics pipeline inputs and backend absence.

## 2. Target Architecture (Static Proton Pipeline)

1) Geometry: `TorusGeometries.create_configuration` → configuration object with `get_field_direction(x,y,z)`.
2) Field build: `SU2FieldBuilder.build_field(field_direction, profile)` where `field_direction` can be either
   - tuple of (n_x, n_y, n_z) arrays, or
   - configuration providing `get_field_direction`.
3) Derivatives: compute left currents `L_i`, traces Tr(L_i L_i), commutators `[L_i, L_j]`, baryon density core inputs.
4) Energy density: c2, c4, c6 terms from derivatives → `EnergyDensity`.
5) Physical quantities: charge density/radius, baryon number, magnetic moment, mass using field, derivatives, profile, energy.
6) Output: CSV with metrics and validation report.

## 3. Concrete Fix Plan

- Backend propagation
  - Add `self.backend` in `ProtonModel` (ArrayBackend from math foundations or from CUDA manager helper).
  - Pass backend to `SU2FieldBuilder`, `EnergyDensityCalculator`, `PhysicalQuantitiesCalculator`, and numerical methods as needed.

- Field direction contract
  - In `SU2FieldBuilder.build_field` support configuration via `get_field_direction(self.X,self.Y,self.Z)` (done).
  - Ensure `TorusGeometries.create_field_direction` returns configuration or arrays consistently. Prefer returning configuration object and let builder extract arrays.

- Derivatives pipeline
  - Implement `EnergyDensityCalculator.calculate_field_derivatives(su2_field)` returning:
    - `traces`: `l_squared`, `comm_squared`
    - `left_currents`: dict for axes x,y,z where each has matrix components `l_00,l_01,l_10,l_11` (temporary placeholder done; replace with real computation using SU(2) field and backend gradients).
  - Remove inline mocks from `calculate_energy_density` (done) and reuse the method above.

- Physics pipeline inputs
  - `PhysicalQuantitiesCalculator.calculate_quantities(su2_field, energy_density, profile=None, derivatives=None)`
    - Accept optional `profile`, `derivatives`. If `None`, reconstruct: profile from builder config; derivatives from `EnergyDensityCalculator` (avoid new dependencies by passing from model).
  - Stop creating mock dicts; use provided arguments.

- Model wiring
  - In `ProtonModel`, store `self.backend` and construct all components with it.
  - Keep `self.profile` used to build the field; pass it to physics calculations.
  - After energy computed, call physics with `(field=self.su2_field, profile=self.profile, derivatives=field_derivatives, energy=energy_density.get_total_energy())`.

- Logging and errors
  - Replace direct `traceback.print_exc()` with logger debug when `--verbose`; otherwise concise messages.

- CLI validation & outputs
  - Ensure CSV naming and columns follow project standard.
  - Generate validation report after physics.

## 4. Step-by-Step Implementation Tasks

1) Add `self.backend` to `ProtonModel.__init__` and use it across components.
2) Store `self.profile` when building field in `build_fields`.
3) Expose derivatives from `EnergyDensityCalculator` back to model (return both density and derivatives, or method to get them).
4) Update `PhysicalQuantitiesCalculator.calculate_quantities` signature to accept `profile` and `field_derivatives`.
5) Remove mocks in physics and energy code paths; use real data.
6) Unify `left_currents` structure across energy and physics computations.
7) Replace ad-hoc prints with logging; keep concise CLI output.
8) Run tests, fix type/lint, and validate CLI run for 120°, clover, cartesian.

## 5. Risks & Mitigations

- GPU/CPU tensor ops divergence: use `ArrayBackend` consistently and `.to_numpy()` only at boundaries (CSV/plots).
- Performance regressions: start with small grids (16/32), add progress bars (already present), profile later.
- Large refactor scope: proceed incrementally with working checkpoints and tests.

## 6. Acceptance Criteria

- `phaze-particles proton static` runs end-to-end, producing CSV and report without mocks.
- All tests remain green; lint clean.
- CUDA detection/logging intact; CPU fallback works.

