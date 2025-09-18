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
            return {"E2_ratio": 0.0, "E4_ratio": 0.0, "E6_ratio": 0.0}

        return {
            "E2_ratio": components["E2"] / total,
            "E4_ratio": components["E4"] / total,
            "E6_ratio": components["E6"] / total,
        }

    def check_virial_condition(self, tolerance: float = 0.05) -> bool:
        """
        Check virial condition E₂ = E₄.

        Args:
            tolerance: Allowed deviation

        Returns:
            True if virial condition is satisfied
        """
        components = self.get_energy_components()
        E2 = components["E2"]
        E4 = components["E4"]

        if E2 == 0 and E4 == 0:
            return True

        ratio = abs(E2 - E4) / max(E2, E4)
        return ratio <= tolerance


class BaryonDensity:
    """Baryon charge density b₀."""

    def __init__(self, field_operations: Any = None):
        """
        Initialize baryon charge density.

        Args:
            field_operations: SU(2) field operations
        """
        self.field_ops = field_operations

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

        return trace.astype(np.float64)


class EnergyDensityCalculator:
    """Energy density calculator."""

    def __init__(
        self,
        grid_size: int,
        box_size: float,
        c2: float = 1.0,
        c4: float = 1.0,
        c6: float = 1.0,
    ):
        """
        Initialize calculator.

        Args:
            grid_size: Grid size
            box_size: Box size
            c2, c4, c6: Skyrme constants
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
        self.c2 = c2
        self.c4 = c4
        self.c6 = c6

        self.baryon_density = BaryonDensity()

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
        b0 = self.baryon_density.compute_baryon_density(left_currents)
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
        # Create mock field derivatives for now
        field_derivatives = {
            "traces": {
                "l_squared": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                "comm_squared": np.ones(
                    (self.grid_size, self.grid_size, self.grid_size)
                ),
            },
            "left_currents": {},
            "baryon_density": np.ones((self.grid_size, self.grid_size, self.grid_size)),
        }
        return self.compute_energy_density(field_derivatives)

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
        # Mock gradient calculation
        return np.random.randn(*su2_field.shape) * 0.01

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

        # Overall assessment
        if virial_ok and balance_quality in ["excellent", "good"]:
            overall_quality = "excellent"
        elif virial_ok and balance_quality == "fair":
            overall_quality = "good"
        elif not virial_ok and balance_quality in ["excellent", "good"]:
            overall_quality = "fair"
        else:
            overall_quality = "poor"

        return {
            "overall_quality": overall_quality,
            "balance_quality": balance_quality,
            "virial_condition": virial_ok,
            "recommendations": self._get_energy_recommendations(balance, virial_ok),
        }

    def _get_energy_recommendations(
        self, balance: Dict[str, float], virial_ok: bool
    ) -> List[str]:
        """
        Get energy improvement recommendations.

        Args:
            balance: Energy balance
            virial_ok: Virial condition satisfaction

        Returns:
            List of recommendations
        """
        recommendations = []

        if not virial_ok:
            recommendations.append(
                "Adjust Skyrme constants to satisfy virial condition E₂ = E₄"
            )

        e2_ratio = balance["E2_ratio"]
        e4_ratio = balance["E4_ratio"]

        if e2_ratio > 0.6:
            recommendations.append("Reduce c₂ constant to decrease E₂ contribution")
        elif e2_ratio < 0.4:
            recommendations.append("Increase c₂ constant to increase E₂ contribution")

        if e4_ratio > 0.6:
            recommendations.append("Reduce c₄ constant to decrease E₄ contribution")
        elif e4_ratio < 0.4:
            recommendations.append("Increase c₄ constant to increase E₄ contribution")

        if balance["E6_ratio"] > 0.1:
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

Virial Condition (E₂ = E₄): {virial_status}

Quality Assessment:
  Overall Quality: {quality['overall_quality'].upper()}
  Balance Quality: {quality['balance_quality'].upper()}

Recommendations:
"""

        for rec in quality["recommendations"]:
            report += f"  - {rec}\n"

        return report
