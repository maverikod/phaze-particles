#!/usr/bin/env python3
"""
Energy densities for proton model.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class EnergyDensity:
    """Energy density of proton model."""

    # Energy density components
    c2_term: np.ndarray  # c₂ Tr(Lᵢ Lᵢ)
    c4_term: np.ndarray  # c₄ Tr([Lᵢ, Lⱼ]²)
    c6_term: np.ndarray  # c₆ b₀²

    # Total energy density
    total_density: np.ndarray  # ℰ = c₂ + c₄ + c₆

    # Metadata
    grid_size: int
    box_size: float
    dx: float

    # Skyrme constants
    c2: float
    c4: float
    c6: float

    def get_total_energy(self) -> float:
        """
        Calculate total energy.

        Returns:
            Total energy E = ∫ ℰ d³x
        """
        return np.sum(self.total_density) * self.dx**3

    def get_energy_components(self) -> Dict[str, float]:
        """
        Calculate energy components.

        Returns:
            Dictionary with energy components
        """
        return {
            "E2": np.sum(self.c2_term) * self.dx**3,
            "E4": np.sum(self.c4_term) * self.dx**3,
            "E6": np.sum(self.c6_term) * self.dx**3,
            "E_total": self.get_total_energy(),
        }

    def get_energy_balance(self) -> Dict[str, float]:
        """
        Calculate energy balance.

        Returns:
            Dictionary with energy balance
        """
        components = self.get_energy_components()
        total = components["E_total"]

        if total == 0:
            return {
                "E2_ratio": 0.0, 
                "E4_ratio": 0.0, 
                "E6_ratio": 0.0,
                "virial_residual": 0.0
            }

        return {
            "E2_ratio": components["E2"] / total,
            "E4_ratio": components["E4"] / total,
            "E6_ratio": components["E6"] / total,
            "virial_residual": self.get_virial_residual(),
        }

    def check_virial_condition(self, tolerance: float = 0.05) -> bool:
        """
        Check virial condition: -E₂ + E₄ + 3E₆ = 0.

        Args:
            tolerance: Allowed deviation

        Returns:
            True if virial condition is satisfied
        """
        components = self.get_energy_components()
        E2 = components["E2"]
        E4 = components["E4"]
        E6 = components["E6"]

        # Virial condition: -E₂ + E₄ + 3E₆ = 0
        virial_residual = -E2 + E4 + 3*E6
        total_energy = E2 + E4 + E6

        if total_energy == 0:
            return True

        # Normalize by total energy
        normalized_residual = abs(virial_residual) / total_energy
        return normalized_residual <= tolerance

    def get_virial_residual(self) -> float:
        """
        Get virial residual: (-E₂ + E₄ + 3E₆)/E_total.

        Returns:
            Virial residual
        """
        components = self.get_energy_components()
        E2 = components["E2"]
        E4 = components["E4"]
        E6 = components["E6"]
        total_energy = E2 + E4 + E6

        if total_energy == 0:
            return 0.0

        return (-E2 + E4 + 3*E6) / total_energy

    def check_positivity(self) -> Dict[str, Any]:
        """
        Check energy positivity and provide statistics.

        Returns:
            Dictionary with positivity analysis
        """
        # Check total energy
        total_energy = self.get_total_energy()
        is_positive = total_energy >= 0
        
        # Check density components
        c2_negative = np.any(self.c2_term < 0)
        c4_negative = np.any(self.c4_term < 0)
        c6_negative = np.any(self.c6_term < 0)
        total_negative = np.any(self.total_density < 0)
        
        # Statistics
        min_density = float(np.min(self.total_density))
        max_density = float(np.max(self.total_density))
        mean_density = float(np.mean(self.total_density))
        
        return {
            "total_energy_positive": is_positive,
            "total_energy": total_energy,
            "c2_has_negative": c2_negative,
            "c4_has_negative": c4_negative,
            "c6_has_negative": c6_negative,
            "total_has_negative": total_negative,
            "min_density": min_density,
            "max_density": max_density,
            "mean_density": mean_density,
        }


class BaryonDensity:
    """Baryon charge density b₀."""

    def __init__(self, field_operations: Any = None, backend: Any = None):
        """
        Initialize baryon charge density.

        Args:
            field_operations: SU(2) field operations
            backend: Array backend
        """
        self.field_ops = field_operations
        self.backend = backend

    def compute_baryon_density(
        self, left_currents: Dict[str, Dict[str, np.ndarray]]
    ) -> Any:
        """
        Calculate baryon charge density.

        Args:
            left_currents: Left currents Lᵢ

        Returns:
            Baryon charge density b₀
        """
        l_x = left_currents["x"]
        l_y = left_currents["y"]
        l_z = left_currents["z"]

        # b₀ = -1/(24π²) εⁱʲᵏ Tr(Lᵢ Lⱼ Lₖ)
        epsilon = self._get_epsilon_tensor()

        # Calculate Tr(Lᵢ Lⱼ Lₖ) for all combinations
        trace_xyz = self._compute_triple_trace(l_x, l_y, l_z)
        trace_yzx = self._compute_triple_trace(l_y, l_z, l_x)
        trace_zxy = self._compute_triple_trace(l_z, l_x, l_y)

        # Sum with antisymmetric tensor
        baryon_density = (
            epsilon[0, 1, 2] * trace_xyz
            + epsilon[1, 2, 0] * trace_yzx
            + epsilon[2, 0, 1] * trace_zxy
        )

        # Normalization
        baryon_density *= -1.0 / (24 * math.pi**2)

        return baryon_density.astype(np.float64)

    def _get_epsilon_tensor(self) -> Any:
        """Get antisymmetric tensor εⁱʲᵏ."""
        epsilon = np.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
        return epsilon.astype(np.float64)

    def _compute_triple_trace(
        self,
        l1: Dict[str, np.ndarray],
        l2: Dict[str, np.ndarray],
        l3: Dict[str, np.ndarray],
    ) -> Any:
        """
        Calculate Tr(L₁ L₂ L₃).

        Args:
            l1, l2, l3: Left currents

        Returns:
            Trace of product
        """
        # L₁ L₂
        l1l2_00 = l1["l_00"] * l2["l_00"] + l1["l_01"] * l2["l_10"]
        l1l2_01 = l1["l_00"] * l2["l_01"] + l1["l_01"] * l2["l_11"]
        l1l2_10 = l1["l_10"] * l2["l_00"] + l1["l_11"] * l2["l_10"]
        l1l2_11 = l1["l_10"] * l2["l_01"] + l1["l_11"] * l2["l_11"]

        # (L₁ L₂) L₃
        trace = (
            l1l2_00 * l3["l_00"]
            + l1l2_01 * l3["l_10"]
            + l1l2_10 * l3["l_01"]
            + l1l2_11 * l3["l_11"]
        )

        xp = self.backend.get_array_module() if self.backend else np
        # Extract real part before converting to float64
        return xp.real(trace).astype(xp.float64)


class EnergyDensityCalculator:
    """Energy density calculator."""

    def __init__(
        self,
        grid_size: int,
        box_size: float,
        c2: float = 1.0,
        c4: float = 1.0,
        c6: float = 1.0,
        backend=None,
    ):
        """
        Initialize calculator.

        Args:
            grid_size: Grid size
            box_size: Box size
            c2, c4, c6: Skyrme constants
            backend: Array backend
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
        self.c2 = c2
        self.c4 = c4
        self.c6 = c6
        self.backend = backend

        self.baryon_density = BaryonDensity(backend=backend)

    def compute_energy_density(
        self, field_derivatives: Dict[str, Any]
    ) -> EnergyDensity:
        """
        Calculate energy density.

        Args:
            field_derivatives: Field derivatives and traces

        Returns:
            Energy density
        """
        traces = field_derivatives["traces"]
        left_currents = field_derivatives["left_currents"]

        # c₂ term: Tr(Lᵢ Lᵢ)
        c2_term = self.c2 * traces["l_squared"]

        # c₄ term: Tr([Lᵢ, Lⱼ]²)
        c4_term = self.c4 * traces["comm_squared"]

        # c₆ term: b₀²
        b0 = self._compute_baryon_density(left_currents)
        c6_term = self.c6 * b0**2

        # Total energy density
        total_density = c2_term + c4_term + c6_term

        return EnergyDensity(
            c2_term=c2_term,
            c4_term=c4_term,
            c6_term=c6_term,
            total_density=total_density,
            grid_size=self.grid_size,
            box_size=self.box_size,
            dx=self.dx,
            c2=self.c2,
            c4=self.c4,
            c6=self.c6,
        )

    def calculate_energy_density(self, su2_field: Any) -> EnergyDensity:
        """
        Calculate energy density from SU(2) field.

        Args:
            su2_field: SU(2) field

        Returns:
            Energy density
        """
        # Calculate field derivatives
        field_derivatives = self.calculate_field_derivatives(su2_field)
        return self.compute_energy_density(field_derivatives)

    def calculate_field_derivatives(self, su2_field: Any) -> Dict[str, Any]:
        """
        Calculate field derivatives from SU(2) field.

        Args:
            su2_field: SU(2) field

        Returns:
            Field derivatives dictionary
        """
        # Calculate left currents L_i = U†∂_i U
        left_currents = self._compute_left_currents(su2_field)
        
        # Calculate traces
        traces = self._compute_traces(left_currents)
        
        # Calculate baryon density b_0
        baryon_density = self._compute_baryon_density(left_currents)
        
        field_derivatives = {
            "traces": traces,
            "left_currents": left_currents,
            "baryon_density": baryon_density,
        }
        return field_derivatives

    def calculate_total_energy(self, su2_field: Any) -> float:
        """
        Calculate total energy from SU(2) field.

        Args:
            su2_field: SU(2) field

        Returns:
            Total energy
        """
        energy_density = self.calculate_energy_density(su2_field)
        return energy_density.get_total_energy()

    def calculate_gradient(self, su2_field: Any) -> np.ndarray:
        """
        Calculate energy gradient.

        Args:
            su2_field: SU(2) field

        Returns:
            Energy gradient
        """
        # Mock gradient calculation - return SU2Field with random gradients
        from .su2_fields import SU2Field

        xp = self.backend.get_array_module() if self.backend else np

        gradient_u_00 = xp.random.randn(*su2_field.shape) * 0.01
        gradient_u_01 = xp.random.randn(*su2_field.shape) * 0.01
        gradient_u_10 = xp.random.randn(*su2_field.shape) * 0.01
        gradient_u_11 = xp.random.randn(*su2_field.shape) * 0.01

        # Create gradient field with validation disabled
        gradient_field = object.__new__(SU2Field)
        gradient_field._skip_validation = True
        gradient_field.u_00 = gradient_u_00
        gradient_field.u_01 = gradient_u_01
        gradient_field.u_10 = gradient_u_10
        gradient_field.u_11 = gradient_u_11
        gradient_field.grid_size = su2_field.grid_size
        gradient_field.box_size = su2_field.box_size
        gradient_field.backend = su2_field.backend
        gradient_field.__post_init__()
        return gradient_field

    def calculate_energy_balance(self, su2_field: Any) -> float:
        """
        Calculate energy balance for virial condition.

        Args:
            su2_field: SU(2) field

        Returns:
            Energy balance ratio
        """
        energy_density = self.calculate_energy_density(su2_field)
        components = energy_density.get_energy_components()
        total = sum(components.values())
        if total > 0:
            return components.get("E2", 0) / total
        return 0.5

    def compute_baryon_number(self, field_derivatives: Dict[str, Any]) -> float:
        """
        Calculate baryon number.

        Args:
            field_derivatives: Field derivatives

        Returns:
            Baryon number B
        """
        left_currents = field_derivatives["left_currents"]
        b0 = self.baryon_density.compute_baryon_density(left_currents)

        # B = ∫ b₀ d³x
        return np.sum(b0) * self.dx**3

    def _compute_left_currents(self, su2_field: Any) -> Dict[str, Dict[str, Any]]:
        """
        Compute left currents L_i = U†∂_i U.

        Args:
            su2_field: SU(2) field

        Returns:
            Dictionary with left currents for x, y, z components
        """
        xp = self.backend.get_array_module() if self.backend else np
        
        # Compute field derivatives using gradient
        du_dx = self._compute_field_derivative(su2_field, axis=0)
        du_dy = self._compute_field_derivative(su2_field, axis=1)
        du_dz = self._compute_field_derivative(su2_field, axis=2)
        
        # Compute L_i = U†∂_i U
        l_x = self._multiply_field_dagger_derivative(su2_field, du_dx)
        l_y = self._multiply_field_dagger_derivative(su2_field, du_dy)
        l_z = self._multiply_field_dagger_derivative(su2_field, du_dz)
        
        return {
            "x": l_x,
            "y": l_y,
            "z": l_z,
        }

    def _compute_field_derivative(self, su2_field: Any, axis: int) -> Dict[str, Any]:
        """
        Compute field derivative along given axis.

        Args:
            su2_field: SU(2) field
            axis: Axis for differentiation (0, 1, 2)

        Returns:
            Dictionary with derivatives of field components
        """
        xp = self.backend.get_array_module() if self.backend else np
        
        # Use backend's gradient function
        if self.backend:
            du_dx = {
                'u_00': self.backend.gradient(su2_field.u_00, self.dx, axis=axis),
                'u_01': self.backend.gradient(su2_field.u_01, self.dx, axis=axis),
                'u_10': self.backend.gradient(su2_field.u_10, self.dx, axis=axis),
                'u_11': self.backend.gradient(su2_field.u_11, self.dx, axis=axis)
            }
        else:
            du_dx = {
                'u_00': np.gradient(su2_field.u_00, self.dx, axis=axis),
                'u_01': np.gradient(su2_field.u_01, self.dx, axis=axis),
                'u_10': np.gradient(su2_field.u_10, self.dx, axis=axis),
                'u_11': np.gradient(su2_field.u_11, self.dx, axis=axis)
            }
        
        return du_dx

    def _multiply_field_dagger_derivative(self, su2_field: Any, 
                                        du: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute U†∂U.

        Args:
            su2_field: SU(2) field
            du: Field derivatives

        Returns:
            Dictionary with L_i components
        """
        xp = self.backend.get_array_module() if self.backend else np
        
        # U† = (u_00*, u_10*; u_01*, u_11*)
        u_dagger_00 = xp.conj(su2_field.u_00)
        u_dagger_01 = xp.conj(su2_field.u_01)
        u_dagger_10 = xp.conj(su2_field.u_10)
        u_dagger_11 = xp.conj(su2_field.u_11)
        
        # L = U†∂U
        l_00 = (u_dagger_00 * du['u_00'] + u_dagger_01 * du['u_10'])
        l_01 = (u_dagger_00 * du['u_01'] + u_dagger_01 * du['u_11'])
        l_10 = (u_dagger_10 * du['u_00'] + u_dagger_11 * du['u_10'])
        l_11 = (u_dagger_10 * du['u_01'] + u_dagger_11 * du['u_11'])
        
        return {
            'l_00': l_00,
            'l_01': l_01,
            'l_10': l_10,
            'l_11': l_11,
        }

    def _compute_traces(self, left_currents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute traces for energy density.

        Args:
            left_currents: Left currents dictionary

        Returns:
            Dictionary with traces
        """
        xp = self.backend.get_array_module() if self.backend else np
        
        # Compute Tr(L_i L_i) for each component
        l_squared = xp.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=xp.float64)
        
        for direction in ['x', 'y', 'z']:
            l = left_currents[direction]
            # Tr(L_i L_i) = |l_00|^2 + |l_01|^2 + |l_10|^2 + |l_11|^2 (positive definite)
            trace_i = (xp.abs(l['l_00'])**2 + 
                      xp.abs(l['l_01'])**2 + 
                      xp.abs(l['l_10'])**2 + 
                      xp.abs(l['l_11'])**2)
            # Convert to float64 (already real and positive)
            trace_i_real = trace_i.astype(xp.float64)
            l_squared += trace_i_real
        
        # Compute Tr([L_i, L_j]^2) - TRUE COMMUTATOR IMPLEMENTATION
        comm_squared = self._compute_commutator_traces(left_currents)
        
        return {
            "l_squared": l_squared,
            "comm_squared": comm_squared,
        }

    def _compute_commutator_traces(self, left_currents: Dict[str, Dict[str, Any]]) -> Any:
        """
        Compute Tr([L_i, L_j]^2) using true commutators.

        Args:
            left_currents: Left currents dictionary

        Returns:
            Commutator trace array
        """
        xp = self.backend.get_array_module() if self.backend else np
        
        l_x = left_currents['x']
        l_y = left_currents['y']
        l_z = left_currents['z']
        
        # Compute commutators [L_i, L_j]
        comm_xy = self._compute_commutator(l_x, l_y)
        comm_yz = self._compute_commutator(l_y, l_z)
        comm_zx = self._compute_commutator(l_z, l_x)
        
        # Compute Tr([L_i, L_j]^2) for each commutator (positive definite)
        trace_xy = (xp.abs(comm_xy['comm_00'])**2 + 
                   xp.abs(comm_xy['comm_01'])**2 + 
                   xp.abs(comm_xy['comm_10'])**2 + 
                   xp.abs(comm_xy['comm_11'])**2)
        
        trace_yz = (xp.abs(comm_yz['comm_00'])**2 + 
                   xp.abs(comm_yz['comm_01'])**2 + 
                   xp.abs(comm_yz['comm_10'])**2 + 
                   xp.abs(comm_yz['comm_11'])**2)
        
        trace_zx = (xp.abs(comm_zx['comm_00'])**2 + 
                   xp.abs(comm_zx['comm_01'])**2 + 
                   xp.abs(comm_zx['comm_10'])**2 + 
                   xp.abs(comm_zx['comm_11'])**2)
        
        # Sum all commutator traces
        comm_squared = trace_xy + trace_yz + trace_zx
        
        # Extract real part and convert to float64
        return xp.real(comm_squared).astype(xp.float64)

    def _compute_commutator(self, l1: Dict[str, Any], l2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute commutator [L1, L2] = L1*L2 - L2*L1.

        Args:
            l1, l2: Left currents

        Returns:
            Commutator components
        """
        # L1*L2
        l1l2_00 = l1["l_00"] * l2["l_00"] + l1["l_01"] * l2["l_10"]
        l1l2_01 = l1["l_00"] * l2["l_01"] + l1["l_01"] * l2["l_11"]
        l1l2_10 = l1["l_10"] * l2["l_00"] + l1["l_11"] * l2["l_10"]
        l1l2_11 = l1["l_10"] * l2["l_01"] + l1["l_11"] * l2["l_11"]

        # L2*L1
        l2l1_00 = l2["l_00"] * l1["l_00"] + l2["l_01"] * l1["l_10"]
        l2l1_01 = l2["l_00"] * l1["l_01"] + l2["l_01"] * l1["l_11"]
        l2l1_10 = l2["l_10"] * l1["l_00"] + l2["l_11"] * l1["l_10"]
        l2l1_11 = l2["l_10"] * l1["l_01"] + l2["l_11"] * l1["l_11"]

        # [L1, L2] = L1*L2 - L2*L1
        return {
            "comm_00": l1l2_00 - l2l1_00,
            "comm_01": l1l2_01 - l2l1_01,
            "comm_10": l1l2_10 - l2l1_10,
            "comm_11": l1l2_11 - l2l1_11,
        }

    def _compute_baryon_density(self, left_currents: Dict[str, Dict[str, Any]]) -> Any:
        """
        Compute baryon density b_0 using triple-trace formula.

        Args:
            left_currents: Left currents dictionary

        Returns:
            Baryon density array
        """
        xp = self.backend.get_array_module() if self.backend else np
        
        # b_0 = -1/(24π²) ε^{ijk} Tr(L_i L_j L_k)
        epsilon = self._get_epsilon_tensor()
        
        l_x = left_currents['x']
        l_y = left_currents['y']
        l_z = left_currents['z']
        
        # Calculate Tr(L_i L_j L_k) for all combinations
        trace_xyz = self._compute_triple_trace(l_x, l_y, l_z)
        trace_yzx = self._compute_triple_trace(l_y, l_z, l_x)
        trace_zxy = self._compute_triple_trace(l_z, l_x, l_y)
        
        # Sum with antisymmetric tensor
        baryon_density = (
            epsilon[0, 1, 2] * trace_xyz
            + epsilon[1, 2, 0] * trace_yzx
            + epsilon[2, 0, 1] * trace_zxy
        )
        
        # Normalization
        baryon_density *= -1.0 / (24 * math.pi**2)
        
        # Extract real part and convert to float64
        return xp.real(baryon_density).astype(xp.float64)

    def _get_epsilon_tensor(self) -> np.ndarray:
        """Get antisymmetric tensor ε^{ijk}."""
        epsilon = np.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
        return epsilon

    def _compute_triple_trace(
        self,
        l1: Dict[str, Any],
        l2: Dict[str, Any],
        l3: Dict[str, Any],
    ) -> Any:
        """
        Calculate Tr(L₁ L₂ L₃).

        Args:
            l1, l2, l3: Left currents

        Returns:
            Trace of product
        """
        # L₁ L₂
        l1l2_00 = l1["l_00"] * l2["l_00"] + l1["l_01"] * l2["l_10"]
        l1l2_01 = l1["l_00"] * l2["l_01"] + l1["l_01"] * l2["l_11"]
        l1l2_10 = l1["l_10"] * l2["l_00"] + l1["l_11"] * l2["l_10"]
        l1l2_11 = l1["l_10"] * l2["l_01"] + l1["l_11"] * l2["l_11"]

        # (L₁ L₂) L₃
        trace = (
            l1l2_00 * l3["l_00"]
            + l1l2_01 * l3["l_10"]
            + l1l2_10 * l3["l_01"]
            + l1l2_11 * l3["l_11"]
        )

        xp = self.backend.get_array_module() if self.backend else np
        # Extract real part before converting to float64
        return xp.real(trace).astype(xp.float64)


class EnergyAnalyzer:
    """Energy analyzer."""

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize analyzer.

        Args:
            tolerance: Allowed deviation for checks
        """
        self.tolerance = tolerance

    def analyze_energy(self, energy_density: EnergyDensity) -> Dict[str, Any]:
        """
        Analyze energy density.

        Args:
            energy_density: Energy density

        Returns:
            Dictionary with analysis results
        """
        analysis: Dict[str, Any] = {}

        # Energy components
        analysis["components"] = energy_density.get_energy_components()

        # Energy balance
        analysis["balance"] = energy_density.get_energy_balance()

        # Virial condition
        analysis["virial_condition"] = energy_density.check_virial_condition(
            self.tolerance
        )
        
        # Virial residual
        analysis["virial_residual"] = energy_density.get_virial_residual()

        # Positivity check
        analysis["positivity"] = energy_density.check_positivity()

        # Density statistics
        analysis["density_stats"] = self._compute_density_statistics(energy_density)

        # Model quality
        analysis["quality"] = self._assess_energy_quality(energy_density)

        return analysis

    def _compute_density_statistics(
        self, energy_density: EnergyDensity
    ) -> Dict[str, float]:
        """
        Calculate energy density statistics.

        Args:
            energy_density: Energy density

        Returns:
            Dictionary with statistics
        """
        return {
            "total_mean": float(np.mean(energy_density.total_density)),
            "total_std": float(np.std(energy_density.total_density)),
            "total_max": float(np.max(energy_density.total_density)),
            "total_min": float(np.min(energy_density.total_density)),
            "c2_mean": float(np.mean(energy_density.c2_term)),
            "c4_mean": float(np.mean(energy_density.c4_term)),
            "c6_mean": float(np.mean(energy_density.c6_term)),
        }

    def _assess_energy_quality(self, energy_density: EnergyDensity) -> Dict[str, Any]:
        """
        Assess energy model quality.

        Args:
            energy_density: Energy density

        Returns:
            Dictionary with quality assessment
        """
        balance = energy_density.get_energy_balance()
        virial_ok = energy_density.check_virial_condition(self.tolerance)
        positivity = energy_density.check_positivity()
        virial_residual = abs(balance.get("virial_residual", 0.0))

        # Assess E₂/E₄ balance
        e2_ratio = balance["E2_ratio"]
        e4_ratio = balance["E4_ratio"]

        if abs(e2_ratio - 0.5) < 0.1 and abs(e4_ratio - 0.5) < 0.1:
            balance_quality = "excellent"
        elif abs(e2_ratio - 0.5) < 0.2 and abs(e4_ratio - 0.5) < 0.2:
            balance_quality = "good"
        elif abs(e2_ratio - 0.5) < 0.3 and abs(e4_ratio - 0.5) < 0.3:
            balance_quality = "fair"
        else:
            balance_quality = "poor"

        # Assess virial quality
        if virial_residual < 0.01:
            virial_quality = "excellent"
        elif virial_residual < 0.05:
            virial_quality = "good"
        elif virial_residual < 0.1:
            virial_quality = "fair"
        else:
            virial_quality = "poor"

        # Overall assessment considering positivity
        if not positivity["total_energy_positive"]:
            overall_quality = "poor"  # Negative energy is critical failure
        elif virial_ok and balance_quality in ["excellent", "good"] and virial_quality in ["excellent", "good"]:
            overall_quality = "excellent"
        elif virial_ok and balance_quality in ["excellent", "good", "fair"] and virial_quality in ["excellent", "good", "fair"]:
            overall_quality = "good"
        elif virial_ok or balance_quality in ["excellent", "good"] or virial_quality in ["excellent", "good"]:
            overall_quality = "fair"
        else:
            overall_quality = "poor"

        return {
            "overall_quality": overall_quality,
            "balance_quality": balance_quality,
            "virial_quality": virial_quality,
            "virial_condition": virial_ok,
            "virial_residual": virial_residual,
            "positivity_ok": positivity["total_energy_positive"],
            "recommendations": self._get_energy_recommendations(balance, virial_ok, positivity),
        }

    def _get_energy_recommendations(
        self, balance: Dict[str, float], virial_ok: bool, positivity: Dict[str, Any]
    ) -> List[str]:
        """
        Get energy improvement recommendations.

        Args:
            balance: Energy balance
            virial_ok: Virial condition satisfaction
            positivity: Positivity analysis

        Returns:
            List of recommendations
        """
        recommendations = []

        # Critical: negative energy
        if not positivity["total_energy_positive"]:
            recommendations.append("CRITICAL: Fix negative energy - check sign conventions in Tr(L_i L_i)")
            if positivity["c2_has_negative"]:
                recommendations.append("c₂ term has negative values - check trace computation")
            if positivity["c4_has_negative"]:
                recommendations.append("c₄ term has negative values - check commutator computation")
            if positivity["c6_has_negative"]:
                recommendations.append("c₆ term has negative values - check baryon density computation")

        # Virial condition
        if not virial_ok:
            virial_residual = abs(balance.get("virial_residual", 0.0))
            recommendations.append(
                f"Adjust Skyrme constants to satisfy virial condition: "
                f"virial residual = {virial_residual:.3f} (target: < 0.05)"
            )

        e2_ratio = balance["E2_ratio"]
        e4_ratio = balance["E4_ratio"]
        e6_ratio = balance["E6_ratio"]

        if e2_ratio > 0.6:
            recommendations.append("Reduce c₂ constant to decrease E₂ contribution")
        elif e2_ratio < 0.4:
            recommendations.append("Increase c₂ constant to increase E₂ contribution")

        if e4_ratio > 0.6:
            recommendations.append("Reduce c₄ constant to decrease E₄ contribution")
        elif e4_ratio < 0.4:
            recommendations.append("Increase c₄ constant to increase E₄ contribution")

        if e6_ratio > 0.1:
            recommendations.append("Reduce c₆ constant to decrease E₆ contribution")

        return recommendations


class EnergyOptimizer:
    """Skyrme constants optimizer."""

    def __init__(
        self,
        target_e2_ratio: float = 0.5,
        target_e4_ratio: float = 0.5,
        tolerance: float = 0.05,
    ):
        """
        Initialize optimizer.

        Args:
            target_e2_ratio: Target E₂/E_total ratio
            target_e4_ratio: Target E₄/E_total ratio
            tolerance: Allowed deviation
        """
        self.target_e2_ratio = target_e2_ratio
        self.target_e4_ratio = target_e4_ratio
        self.tolerance = tolerance

    def optimize_constants(
        self,
        initial_c2: float,
        initial_c4: float,
        initial_c6: float,
        field_derivatives: Dict[str, Any],
        max_iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Optimize Skyrme constants.

        Args:
            initial_c2, initial_c4, initial_c6: Initial constants
            field_derivatives: Field derivatives
            max_iterations: Maximum number of iterations

        Returns:
            Optimized constants
        """
        c2, c4, c6 = initial_c2, initial_c4, initial_c6

        for iteration in range(max_iterations):
            # Calculate energy density with current constants
            grid_size = field_derivatives["left_currents"]["x"]["l_00"].shape[0]
            calculator = EnergyDensityCalculator(
                grid_size,
                grid_size * 0.1,  # Approximate box_size
                c2,
                c4,
                c6,
            )

            energy_density = calculator.compute_energy_density(field_derivatives)
            balance = energy_density.get_energy_balance()

            # Check convergence
            e2_error = abs(balance["E2_ratio"] - self.target_e2_ratio)
            e4_error = abs(balance["E4_ratio"] - self.target_e4_ratio)

            if e2_error < self.tolerance and e4_error < self.tolerance:
                break

            # Adjust constants
            if balance["E2_ratio"] > self.target_e2_ratio:
                c2 *= 0.95
            else:
                c2 *= 1.05

            if balance["E4_ratio"] > self.target_e4_ratio:
                c4 *= 0.95
            else:
                c4 *= 1.05

        return {"c2": c2, "c4": c4, "c6": c6}


# Main class for energy density
class EnergyDensities:
    """Main class for energy density operations."""

    def __init__(
        self,
        grid_size: int = 64,
        box_size: float = 4.0,
        c2: float = 1.0,
        c4: float = 1.0,
        c6: float = 1.0,
    ):
        """
        Initialize energy density.

        Args:
            grid_size: Grid size
            box_size: Box size
            c2, c4, c6: Skyrme constants
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.c2 = c2
        self.c4 = c4
        self.c6 = c6

        self.calculator = EnergyDensityCalculator(grid_size, box_size, c2, c4, c6)
        self.analyzer = EnergyAnalyzer()
        self.optimizer = EnergyOptimizer()

    def compute_energy(self, field_derivatives: Dict[str, Any]) -> EnergyDensity:
        """
        Calculate energy density.

        Args:
            field_derivatives: Field derivatives

        Returns:
            Energy density
        """
        return self.calculator.compute_energy_density(field_derivatives)

    def compute_baryon_number(self, field_derivatives: Dict[str, Any]) -> float:
        """
        Calculate baryon number.

        Args:
            field_derivatives: Field derivatives

        Returns:
            Baryon number
        """
        return self.calculator.compute_baryon_number(field_derivatives)

    def analyze_energy(self, energy_density: EnergyDensity) -> Dict[str, Any]:
        """
        Analyze energy density.

        Args:
            energy_density: Energy density

        Returns:
            Analysis results
        """
        return self.analyzer.analyze_energy(energy_density)

    def optimize_constants(self, field_derivatives: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize Skyrme constants.

        Args:
            field_derivatives: Field derivatives

        Returns:
            Optimized constants
        """
        return self.optimizer.optimize_constants(
            self.c2, self.c4, self.c6, field_derivatives
        )

    def get_energy_report(self, energy_density: EnergyDensity) -> str:
        """
        Get energy report.

        Args:
            energy_density: Energy density

        Returns:
            Text report
        """
        analysis = self.analyze_energy(energy_density)
        components = analysis["components"]
        balance = analysis["balance"]
        quality = analysis["quality"]
        virial_status = "✓ PASS" if analysis["virial_condition"] else "✗ FAIL"

        virial_residual = balance.get('virial_residual', 0.0)
        positivity_status = "✓ PASS" if quality.get('positivity_ok', True) else "✗ FAIL"
        
        report = f"""
ENERGY DENSITY ANALYSIS
=======================

Energy Components:
  E₂ (c₂ term): {components['E2']:.6f}
  E₄ (c₄ term): {components['E4']:.6f}
  E₆ (c₆ term): {components['E6']:.6f}
  E_total: {components['E_total']:.6f}

Energy Balance:
  E₂/E_total: {balance['E2_ratio']:.3f} (target: 0.500)
  E₄/E_total: {balance['E4_ratio']:.3f} (target: 0.500)
  E₆/E_total: {balance['E6_ratio']:.3f}

Virial Analysis:
  Virial Condition (-E₂ + E₄ + 3E₆ = 0): {virial_status}
  Virial Residual: {virial_residual:.6f} (target: < 0.05)

Positivity Check: {positivity_status}

Quality Assessment:
  Overall Quality: {quality['overall_quality'].upper()}
  Balance Quality: {quality['balance_quality'].upper()}
  Virial Quality: {quality.get('virial_quality', 'unknown').upper()}

Recommendations:
"""

        for rec in quality["recommendations"]:
            report += f"  - {rec}\n"

        return report
