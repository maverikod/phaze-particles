"""
Topological radius analyzer for Skyrme field quantization.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from phaze_particles.utils.mathematical_foundations import ArrayBackend


@dataclass
class TopologicalRadius:
    """Effective topological radius of defect."""
    geometric_radius: float
    phase_radius: float
    effective_radius: float
    topological_charge: float
    phase_transitions: int


@dataclass
class QuantizationLaw:
    """Quantization law derived from topological radius."""
    n_bands: int
    band_spacing: float
    effective_radius: float
    topological_charge: float
    phase_factor: float


class TopologicalRadiusAnalyzer:
    """
    Analyzer for effective topological radius and quantization.
    
    Connects geometric, phase, and topological properties.
    """
    
    def __init__(self, grid_size: int, box_size: float, backend: Optional[ArrayBackend] = None):
        """
        Initialize topological analyzer.
        
        Args:
            grid_size: Grid size
            box_size: Box size
            backend: Array backend
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
        self.backend = backend or ArrayBackend()
        
        # Create coordinate grids
        x = self.backend.linspace(-box_size / 2, box_size / 2, grid_size)
        y = self.backend.linspace(-box_size / 2, box_size / 2, grid_size)
        z = self.backend.linspace(-box_size / 2, box_size / 2, grid_size)
        self.X, self.Y, self.Z = self.backend.meshgrid(x, y, z, indexing="ij")
        self.R = self.backend.sqrt(self.X**2 + self.Y**2 + self.Z**2)
    
    def analyze_topological_radius(
        self, 
        su2_field: Any, 
        energy_density: Any,
        baryon_density: Any
    ) -> TopologicalRadius:
        """
        Analyze effective topological radius.
        
        Args:
            su2_field: SU(2) field
            energy_density: Energy density
            baryon_density: Baryon density
            
        Returns:
            Topological radius analysis
        """
        # Convert to numpy
        R_np = self.R.get() if hasattr(self.R, 'get') else self.R
        
        # 1. Geometric radius (energy-weighted)
        geometric_radius = self._calculate_geometric_radius(R_np, energy_density)
        
        # 2. Phase radius (from field topology)
        phase_radius = self._calculate_phase_radius(R_np, su2_field)
        
        # 3. Effective radius (combination)
        effective_radius = self._calculate_effective_radius(geometric_radius, phase_radius)
        
        # 4. Topological charge
        topological_charge = self._calculate_topological_charge(baryon_density)
        
        # 5. Phase transitions
        phase_transitions = self._count_phase_transitions(R_np, su2_field)
        
        return TopologicalRadius(
            geometric_radius=geometric_radius,
            phase_radius=phase_radius,
            effective_radius=effective_radius,
            topological_charge=topological_charge,
            phase_transitions=phase_transitions
        )
    
    def _calculate_geometric_radius(self, R: np.ndarray, energy_density: Any) -> float:
        """Calculate energy-weighted geometric radius."""
        # Convert energy density to numpy
        e_total = energy_density.total_density.get() if hasattr(energy_density.total_density, 'get') else energy_density.total_density
        
        # Energy-weighted radius
        total_energy = np.sum(e_total)
        if total_energy == 0:
            return 0.0
        
        energy_weighted_radius = np.sum(R * e_total) / total_energy
        return float(energy_weighted_radius)
    
    def _calculate_phase_radius(self, R: np.ndarray, su2_field: Any) -> float:
        """Calculate phase radius from field topology."""
        # Extract field components
        u_00 = su2_field.u_00.get() if hasattr(su2_field.u_00, 'get') else su2_field.u_00
        u_01 = su2_field.u_01.get() if hasattr(su2_field.u_01, 'get') else su2_field.u_01
        u_10 = su2_field.u_10.get() if hasattr(su2_field.u_10, 'get') else su2_field.u_10
        u_11 = su2_field.u_11.get() if hasattr(su2_field.u_11, 'get') else su2_field.u_11
        
        # Calculate phase (argument of determinant)
        det = u_00 * u_11 - u_01 * u_10
        phase = np.angle(det)
        
        # Find radius where phase changes significantly
        phase_gradient = np.gradient(phase, axis=0)
        phase_gradient_magnitude = np.abs(phase_gradient)
        
        # Phase radius as radius of maximum phase gradient
        max_gradient_indices = np.unravel_index(np.argmax(phase_gradient_magnitude), phase_gradient_magnitude.shape)
        phase_radius = R[max_gradient_indices]
        
        return float(phase_radius)
    
    def _calculate_effective_radius(self, geometric_radius: float, phase_radius: float) -> float:
        """Calculate effective topological radius."""
        # Weighted combination of geometric and phase radii
        # Phase radius typically smaller, so weight it more
        effective_radius = 0.3 * geometric_radius + 0.7 * phase_radius
        return effective_radius
    
    def _calculate_topological_charge(self, baryon_density: Any) -> float:
        """Calculate topological charge from baryon density."""
        # Convert to numpy
        b_density = baryon_density.get() if hasattr(baryon_density, 'get') else baryon_density
        
        # Integrate baryon density
        total_charge = np.sum(b_density) * self.dx**3
        return float(total_charge)
    
    def _count_phase_transitions(self, R: np.ndarray, su2_field: Any) -> int:
        """Count phase transitions in field."""
        # Extract field components
        u_00 = su2_field.u_00.get() if hasattr(su2_field.u_00, 'get') else su2_field.u_00
        u_01 = su2_field.u_01.get() if hasattr(su2_field.u_01, 'get') else su2_field.u_01
        u_10 = su2_field.u_10.get() if hasattr(su2_field.u_10, 'get') else su2_field.u_10
        u_11 = su2_field.u_11.get() if hasattr(su2_field.u_11, 'get') else su2_field.u_11
        
        # Calculate phase
        det = u_00 * u_11 - u_01 * u_10
        phase = np.angle(det)
        
        # Count phase jumps (transitions)
        phase_diff = np.diff(phase.flatten())
        phase_jumps = np.sum(np.abs(phase_diff) > np.pi/2)  # Significant phase changes
        
        return int(phase_jumps)
    
    def derive_quantization_law(
        self, 
        topological_radius: TopologicalRadius,
        n_bands: int
    ) -> QuantizationLaw:
        """
        Derive quantization law from topological radius.
        
        Args:
            topological_radius: Topological radius analysis
            n_bands: Number of observed energy bands
            
        Returns:
            Quantization law
        """
        # Quantization law: n_bands = f(topological_charge, effective_radius)
        
        # Theoretical prediction
        predicted_bands = self._predict_band_count(
            topological_radius.topological_charge,
            topological_radius.effective_radius,
            topological_radius.phase_transitions
        )
        
        # Band spacing from effective radius
        band_spacing = self._calculate_band_spacing(
            topological_radius.effective_radius,
            topological_radius.topological_charge
        )
        
        # Phase factor
        phase_factor = self._calculate_phase_factor(
            topological_radius.phase_transitions,
            topological_radius.effective_radius
        )
        
        return QuantizationLaw(
            n_bands=n_bands,
            band_spacing=band_spacing,
            effective_radius=topological_radius.effective_radius,
            topological_charge=topological_radius.topological_charge,
            phase_factor=phase_factor
        )
    
    def _predict_band_count(self, topological_charge: float, effective_radius: float, phase_transitions: int) -> int:
        """Predict number of energy bands from topological properties."""
        # Empirical formula based on topological charge and radius
        # n_bands ∝ |B| × R_eff × phase_factor
        
        base_bands = int(abs(topological_charge) * effective_radius * 10)  # Scaling factor
        phase_contribution = phase_transitions // 2  # Every 2 phase transitions add a band
        
        predicted = base_bands + phase_contribution
        return max(1, predicted)  # At least 1 band
    
    def _calculate_band_spacing(self, effective_radius: float, topological_charge: float) -> float:
        """Calculate energy band spacing."""
        # Band spacing inversely proportional to effective radius
        # ΔE ∝ 1/R_eff × |B|
        
        spacing = abs(topological_charge) / (effective_radius + 1e-6)
        return spacing
    
    def _calculate_phase_factor(self, phase_transitions: int, effective_radius: float) -> float:
        """Calculate phase factor for quantization."""
        # Phase factor accounts for phase transitions
        phase_factor = 1.0 + 0.1 * phase_transitions / (effective_radius + 1e-6)
        return phase_factor
    
    def generate_topological_report(self, topological_radius: TopologicalRadius, quantization_law: QuantizationLaw) -> str:
        """Generate detailed topological analysis report."""
        report = []
        report.append("=" * 70)
        report.append("TOPOLOGICAL RADIUS AND QUANTIZATION ANALYSIS")
        report.append("=" * 70)
        
        report.append(f"\nTOPOLOGICAL RADIUS:")
        report.append(f"  Geometric radius: {topological_radius.geometric_radius:.4f} fm")
        report.append(f"  Phase radius: {topological_radius.phase_radius:.4f} fm")
        report.append(f"  Effective radius: {topological_radius.effective_radius:.4f} fm")
        report.append(f"  Topological charge: {topological_radius.topological_charge:.6f}")
        report.append(f"  Phase transitions: {topological_radius.phase_transitions}")
        
        report.append(f"\nQUANTIZATION LAW:")
        report.append(f"  Number of bands: {quantization_law.n_bands}")
        report.append(f"  Band spacing: {quantization_law.band_spacing:.6f}")
        report.append(f"  Phase factor: {quantization_law.phase_factor:.6f}")
        
        # Theoretical prediction
        predicted_bands = self._predict_band_count(
            topological_radius.topological_charge,
            topological_radius.effective_radius,
            topological_radius.phase_transitions
        )
        report.append(f"  Predicted bands: {predicted_bands}")
        
        # Accuracy
        accuracy = 100 * (1 - abs(quantization_law.n_bands - predicted_bands) / max(quantization_law.n_bands, predicted_bands))
        report.append(f"  Prediction accuracy: {accuracy:.1f}%")
        
        report.append(f"\nPHYSICAL INTERPRETATION:")
        report.append(f"  The effective topological radius {topological_radius.effective_radius:.4f} fm")
        report.append(f"  determines the quantization structure with {quantization_law.n_bands} bands.")
        report.append(f"  Phase transitions ({topological_radius.phase_transitions}) create")
        report.append(f"  additional quantization levels.")
        
        return "\n".join(report)


class QuantizationPredictor:
    """
    Predictor for quantization based on topological properties.
    """
    
    def __init__(self, analyzer: TopologicalRadiusAnalyzer):
        """
        Initialize quantization predictor.
        
        Args:
            analyzer: Topological radius analyzer
        """
        self.analyzer = analyzer
    
    def predict_quantization(
        self, 
        su2_field: Any, 
        energy_density: Any,
        baryon_density: Any
    ) -> Dict[str, Any]:
        """
        Predict quantization structure from topological properties.
        
        Args:
            su2_field: SU(2) field
            energy_density: Energy density
            baryon_density: Baryon density
            
        Returns:
            Quantization prediction
        """
        # Analyze topological radius
        topological_radius = self.analyzer.analyze_topological_radius(
            su2_field, energy_density, baryon_density
        )
        
        # Predict band count
        predicted_bands = self.analyzer._predict_band_count(
            topological_radius.topological_charge,
            topological_radius.effective_radius,
            topological_radius.phase_transitions
        )
        
        # Calculate band spacing
        band_spacing = self.analyzer._calculate_band_spacing(
            topological_radius.effective_radius,
            topological_radius.topological_charge
        )
        
        # Calculate phase factor
        phase_factor = self.analyzer._calculate_phase_factor(
            topological_radius.phase_transitions,
            topological_radius.effective_radius
        )
        
        return {
            'topological_radius': topological_radius,
            'predicted_bands': predicted_bands,
            'band_spacing': band_spacing,
            'phase_factor': phase_factor,
            'effective_radius': topological_radius.effective_radius,
            'topological_charge': topological_radius.topological_charge
        }
