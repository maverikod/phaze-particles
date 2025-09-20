"""
Phase Environment Module for 7D Phase Space-Time Theory

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com

This module implements the phase environment and self-gravity of defects
according to the 7D phase space-time theory from 7d-00-15.md.

Key concepts:
- Phase field creates a "well" around defects (self-gravity)
- Balance of compression-rarefaction stabilizes phase structures
- Quantization of scales: natural scale R* and spectral scales Rn
- Robin boundary conditions with impedance operator K
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from phaze_particles.utils.mathematical_foundations import ArrayBackend


@dataclass
class PhaseWellParameters:
    """Parameters for the phase well around a defect."""
    well_depth: float = 1.0  # Depth of the phase well
    well_width: float = 1.0  # Width of the phase well
    compression_strength: float = 1.0  # Strength of phase compression
    rarefaction_strength: float = 1.0  # Strength of phase rarefaction


@dataclass
class ImpedanceParameters:
    """Parameters for the impedance operator K."""
    K_real: float = 0.0  # Real part (0 = closed, >0 = open)
    K_imag: float = 0.0  # Imaginary part
    boundary_radius: float = 1.0  # Boundary radius R
    phase_velocity: float = 1.0  # Phase velocity c_phi


@dataclass
class ScaleQuantization:
    """Results of scale quantization."""
    natural_radius_R_star: float  # Natural scale from virial conditions
    spectral_radii_Rn: List[float]  # Spectral scales from quantization
    allowed_radii: List[float]  # Physically allowed radii |Rn-R*| ≤ ΔR
    delta_R: float  # Width of allowed region


class PhaseEnvironment:
    """
    Phase environment with self-gravity of defects.
    
    Implements the phase field "well" that stabilizes defects and
    the balance of compression-rarefaction according to 7D theory.
    """
    
    def __init__(self, backend: Optional[ArrayBackend] = None):
        """Initialize phase environment."""
        self.backend = backend or ArrayBackend()
        self.xp = self.backend.get_array_module()
        
        # Phase well parameters
        self.well_params = PhaseWellParameters()
        
        # Impedance parameters
        self.impedance_params = ImpedanceParameters()
        
        # Scale quantization results
        self.scale_quantization: Optional[ScaleQuantization] = None
    
    def compute_phase_well(
        self, 
        defect_position: np.ndarray,
        field_strength: float,
        coordinates: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute the phase "well" around a defect.
        
        The phase field creates a "well" that stabilizes the defect's existence.
        This is the self-gravity mechanism from 7D theory.
        
        Args:
            defect_position: Position of the defect
            field_strength: Strength of the phase field
            coordinates: Coordinate arrays (X, Y, Z, R)
            
        Returns:
            Dictionary with phase well components
        """
        xp = self.xp
        X, Y, Z, R = coordinates['X'], coordinates['Y'], coordinates['Z'], coordinates['R']
        
        # Distance from defect
        dx = X - defect_position[0]
        dy = Y - defect_position[1] 
        dz = Z - defect_position[2]
        r_from_defect = xp.sqrt(dx**2 + dy**2 + dz**2)
        
        # Phase well profile (Yukawa-like with exponential damping)
        # This creates the "well" that stabilizes the defect
        well_profile = self._compute_well_profile(
            r_from_defect, 
            self.well_params.well_depth,
            self.well_params.well_width
        )
        
        # Phase compression in the core (positive phase)
        compression = self.well_params.compression_strength * well_profile
        
        # Phase rarefaction in the tail (negative phase)
        rarefaction = -self.well_params.rarefaction_strength * well_profile
        
        # Total phase field (compression + rarefaction)
        phase_field = compression + rarefaction
        
        # Phase gradients (for energy calculations)
        phase_grad_x = self._compute_phase_gradient(dx, r_from_defect, well_profile)
        phase_grad_y = self._compute_phase_gradient(dy, r_from_defect, well_profile)
        phase_grad_z = self._compute_phase_gradient(dz, r_from_defect, well_profile)
        
        return {
            'phase_field': phase_field,
            'compression': compression,
            'rarefaction': rarefaction,
            'phase_grad_x': phase_grad_x,
            'phase_grad_y': phase_grad_y,
            'phase_grad_z': phase_grad_z,
            'well_profile': well_profile,
            'r_from_defect': r_from_defect
        }
    
    def _compute_well_profile(
        self, 
        r: np.ndarray, 
        depth: float, 
        width: float
    ) -> np.ndarray:
        """Compute the phase well profile (Yukawa-like)."""
        xp = self.xp
        
        # Avoid division by zero
        r_safe = xp.maximum(r, 1e-10)
        
        # Yukawa-like profile: exp(-r/width) / r
        # This gives the characteristic 1/r at small distances
        # and exponential damping at large distances
        profile = depth * xp.exp(-r_safe / width) / r_safe
        
        return profile
    
    def _compute_phase_gradient(
        self, 
        coordinate: np.ndarray, 
        r: np.ndarray, 
        profile: np.ndarray
    ) -> np.ndarray:
        """Compute phase gradient in one direction."""
        xp = self.xp
        
        # Avoid division by zero
        r_safe = xp.maximum(r, 1e-10)
        
        # Gradient of Yukawa-like profile
        # d/dx [exp(-r/width) / r] = -exp(-r/width) * (1 + r/width) * x / r^3
        gradient = -profile * (1 + r / self.well_params.well_width) * coordinate / (r_safe**2)
        
        return gradient
    
    def compute_impedance_operator(
        self, 
        boundary_radius: float,
        phase_velocity: float = 1.0
    ) -> ImpedanceParameters:
        """
        Compute the impedance operator K for Robin boundary conditions.
        
        (∂n + K)ψ|r=R = 0
        
        Args:
            boundary_radius: Boundary radius R
            phase_velocity: Phase velocity c_phi
            
        Returns:
            Impedance parameters
        """
        # For closed system: K = 0 (self-adjoint, unitary)
        # For open system: K > 0 (dissipative, finite widths)
        
        # In 7D theory, K depends on the phase field strength
        # and the boundary conditions
        K_real = 0.0  # Start with closed system
        K_imag = 0.0  # No imaginary part initially
        
        # For open system, K_real > 0 would give dissipation
        # This would be determined by the specific physics
        
        self.impedance_params = ImpedanceParameters(
            K_real=K_real,
            K_imag=K_imag,
            boundary_radius=boundary_radius,
            phase_velocity=phase_velocity
        )
        
        return self.impedance_params
    
    def quantize_scales(
        self, 
        natural_radius_R_star: float,
        phase_velocity: float = 1.0,
        max_modes: int = 20
    ) -> ScaleQuantization:
        """
        Quantize scales according to the condition |Rn-R*| ≤ ΔR.
        
        This implements the scale quantization from 7D theory where
        natural scale R* and spectral scales Rn must agree.
        
        Args:
            natural_radius_R_star: Natural scale from virial conditions
            phase_velocity: Phase velocity c_phi
            max_modes: Maximum number of modes to consider
            
        Returns:
            Scale quantization results
        """
        # Spectral quantization condition: k*R + δ(k*) = π*n
        # where k* = ω/c_phi and δ is the phase shift
        
        # For simplicity, assume δ = 0 (no phase shift)
        # Then: k*R = π*n → R = π*n / k*
        
        # k* is determined by the natural frequency
        # For a stable defect: k* = 2π / λ* where λ* is the natural wavelength
        k_star = 2 * np.pi / natural_radius_R_star
        
        # Compute spectral radii Rn
        spectral_radii_Rn = []
        for n in range(1, max_modes + 1):
            Rn = np.pi * n / k_star
            spectral_radii_Rn.append(Rn)
        
        # Compute allowed radii: |Rn-R*| ≤ ΔR
        # ΔR is determined by the curvature of the energy functional
        # For now, use a simple estimate: ΔR = 0.1 * R*
        delta_R = 0.1 * natural_radius_R_star
        
        allowed_radii = []
        for Rn in spectral_radii_Rn:
            if abs(Rn - natural_radius_R_star) <= delta_R:
                allowed_radii.append(Rn)
        
        self.scale_quantization = ScaleQuantization(
            natural_radius_R_star=natural_radius_R_star,
            spectral_radii_Rn=spectral_radii_Rn,
            allowed_radii=allowed_radii,
            delta_R=delta_R
        )
        
        return self.scale_quantization
    
    def compute_compression_rarefaction_balance(
        self, 
        phase_well: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compute the balance of compression-rarefaction.
        
        This is the fundamental stability mechanism from 7D theory:
        - Phase compression in the core
        - Phase rarefaction in the tail
        - Balance ensures stability
        """
        compression = phase_well['compression']
        rarefaction = phase_well['rarefaction']
        
        # Total compression and rarefaction
        total_compression = np.sum(compression)
        total_rarefaction = np.sum(rarefaction)
        
        # Balance ratio (should be close to 1 for stability)
        balance_ratio = abs(total_compression / total_rarefaction) if total_rarefaction != 0 else 0
        
        # Stability indicator
        is_stable = 0.8 <= balance_ratio <= 1.2
        
        return {
            'total_compression': total_compression,
            'total_rarefaction': total_rarefaction,
            'balance_ratio': balance_ratio,
            'is_stable': is_stable,
            'stability_margin': min(balance_ratio, 1.0/balance_ratio) if balance_ratio > 0 else 0
        }
    
    def compute_phase_field_energy(
        self, 
        phase_well: Dict[str, np.ndarray],
        field_derivatives: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute the energy of the phase field.
        
        This includes the energy of the phase well and its interaction
        with the defect field.
        """
        phase_field = phase_well['phase_field']
        phase_grad_x = phase_well['phase_grad_x']
        phase_grad_y = phase_well['phase_grad_y']
        phase_grad_z = phase_well['phase_grad_z']
        
        # Phase field energy density
        phase_energy_density = 0.5 * (
            phase_grad_x**2 + phase_grad_y**2 + phase_grad_z**2
        )
        
        # Total phase field energy
        total_phase_energy = np.sum(phase_energy_density)
        
        # Interaction energy with defect field
        # This would depend on the specific defect field
        interaction_energy = 0.0  # Placeholder
        
        return {
            'phase_energy_density': phase_energy_density,
            'total_phase_energy': total_phase_energy,
            'interaction_energy': interaction_energy,
            'total_energy': total_phase_energy + interaction_energy
        }
    
    def get_environment_report(self) -> str:
        """Generate a report on the phase environment."""
        if not self.scale_quantization:
            return "Phase environment not initialized. Run quantize_scales() first."
        
        report = []
        report.append("PHASE ENVIRONMENT REPORT")
        report.append("=" * 50)
        
        report.append(f"\nPhase Well Parameters:")
        report.append(f"  Well depth: {self.well_params.well_depth}")
        report.append(f"  Well width: {self.well_params.well_width}")
        report.append(f"  Compression strength: {self.well_params.compression_strength}")
        report.append(f"  Rarefaction strength: {self.well_params.rarefaction_strength}")
        
        report.append(f"\nImpedance Parameters:")
        report.append(f"  K_real: {self.impedance_params.K_real}")
        report.append(f"  K_imag: {self.impedance_params.K_imag}")
        report.append(f"  Boundary radius: {self.impedance_params.boundary_radius}")
        report.append(f"  Phase velocity: {self.impedance_params.phase_velocity}")
        
        report.append(f"\nScale Quantization:")
        report.append(f"  Natural radius R*: {self.scale_quantization.natural_radius_R_star:.3f}")
        report.append(f"  Spectral radii Rn: {len(self.scale_quantization.spectral_radii_Rn)} modes")
        report.append(f"  Allowed radii: {len(self.scale_quantization.allowed_radii)} modes")
        report.append(f"  ΔR: {self.scale_quantization.delta_R:.3f}")
        
        if self.scale_quantization.allowed_radii:
            report.append(f"  First few allowed radii: {self.scale_quantization.allowed_radii[:5]}")
        
        return "\n".join(report)
