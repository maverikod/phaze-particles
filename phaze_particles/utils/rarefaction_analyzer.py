"""
Rarefaction zone analyzer for Skyrme field.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from phaze_particles.utils.mathematical_foundations import ArrayBackend


@dataclass
class RarefactionZone:
    """Rarefaction zone analysis."""
    core_radius: float
    rarefaction_radius: float
    vacuum_radius: float
    rarefaction_energy: float
    background_field_strength: float
    tail_overlap_factor: float


@dataclass
class DefectTail:
    """Individual defect tail structure."""
    tail_radius: float
    tail_energy: float
    decay_length: float
    mode_count: int
    overlap_region: float


class RarefactionZoneAnalyzer:
    """
    Analyzer for rarefaction zone and tail overlap.
    
    Studies the intersection of defect tails and background field.
    """
    
    def __init__(self, grid_size: int, box_size: float, backend: Optional[ArrayBackend] = None):
        """
        Initialize rarefaction analyzer.
        
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
    
    def analyze_rarefaction_zone(
        self, 
        energy_density: Any, 
        field_profile: Any,
        core_threshold: float = 0.8,
        rarefaction_threshold: float = 0.1
    ) -> RarefactionZone:
        """
        Analyze rarefaction zone structure.
        
        Args:
            energy_density: Energy density
            field_profile: Radial profile
            core_threshold: Threshold for core region
            rarefaction_threshold: Threshold for rarefaction zone
            
        Returns:
            Rarefaction zone analysis
        """
        # Convert to numpy
        R_np = self.R.get() if hasattr(self.R, 'get') else self.R
        e_total = energy_density.total_density.get() if hasattr(energy_density.total_density, 'get') else energy_density.total_density
        
        # Find core radius
        core_radius = self._find_core_radius(R_np, e_total, core_threshold)
        
        # Find rarefaction zone
        rarefaction_radius = self._find_rarefaction_radius(R_np, e_total, rarefaction_threshold)
        
        # Find vacuum boundary
        vacuum_radius = self._find_vacuum_radius(R_np, e_total)
        
        # Calculate rarefaction energy
        rarefaction_energy = self._calculate_rarefaction_energy(
            R_np, e_total, core_radius, rarefaction_radius
        )
        
        # Calculate background field strength
        background_strength = self._calculate_background_strength(
            R_np, e_total, rarefaction_radius, vacuum_radius
        )
        
        # Calculate tail overlap factor
        overlap_factor = self._calculate_tail_overlap_factor(
            R_np, e_total, core_radius, rarefaction_radius
        )
        
        return RarefactionZone(
            core_radius=core_radius,
            rarefaction_radius=rarefaction_radius,
            vacuum_radius=vacuum_radius,
            rarefaction_energy=rarefaction_energy,
            background_field_strength=background_strength,
            tail_overlap_factor=overlap_factor
        )
    
    def _find_core_radius(self, R: np.ndarray, energy_density: np.ndarray, threshold: float) -> float:
        """Find core radius containing threshold fraction of energy."""
        total_energy = np.sum(energy_density)
        cumulative_energy = 0
        
        r_bins = np.linspace(0, np.max(R), 50)
        for i in range(len(r_bins) - 1):
            mask = (R >= r_bins[i]) & (R < r_bins[i+1])
            shell_energy = np.sum(energy_density[mask])
            cumulative_energy += shell_energy
            
            if cumulative_energy >= threshold * total_energy:
                return r_bins[i+1]
        
        return np.max(R)
    
    def _find_rarefaction_radius(self, R: np.ndarray, energy_density: np.ndarray, threshold: float) -> float:
        """Find rarefaction zone radius."""
        # Find radius where energy density drops to threshold of maximum
        max_energy = np.max(energy_density)
        threshold_energy = threshold * max_energy
        
        # Find outermost radius with significant energy
        r_bins = np.linspace(0, np.max(R), 50)
        for i in range(len(r_bins) - 1, 0, -1):  # Search from outside in
            mask = (R >= r_bins[i-1]) & (R < r_bins[i])
            if np.any(mask):
                shell_max_energy = np.max(energy_density[mask])
                if shell_max_energy >= threshold_energy:
                    return r_bins[i]
        
        return 0.0
    
    def _find_vacuum_radius(self, R: np.ndarray, energy_density: np.ndarray) -> float:
        """Find vacuum boundary radius."""
        # Find radius where energy density becomes negligible
        max_energy = np.max(energy_density)
        vacuum_threshold = 1e-6 * max_energy
        
        r_bins = np.linspace(0, np.max(R), 50)
        for i in range(len(r_bins) - 1, 0, -1):
            mask = (R >= r_bins[i-1]) & (R < r_bins[i])
            if np.any(mask):
                shell_max_energy = np.max(energy_density[mask])
                if shell_max_energy <= vacuum_threshold:
                    return r_bins[i-1]
        
        return np.max(R)
    
    def _calculate_rarefaction_energy(
        self, 
        R: np.ndarray, 
        energy_density: np.ndarray, 
        core_radius: float, 
        rarefaction_radius: float
    ) -> float:
        """Calculate energy in rarefaction zone."""
        mask = (R > core_radius) & (R <= rarefaction_radius)
        rarefaction_energy = np.sum(energy_density[mask]) * self.dx**3
        return float(rarefaction_energy)
    
    def _calculate_background_strength(
        self, 
        R: np.ndarray, 
        energy_density: np.ndarray, 
        rarefaction_radius: float, 
        vacuum_radius: float
    ) -> float:
        """Calculate background field strength in rarefaction zone."""
        mask = (R > rarefaction_radius) & (R < vacuum_radius)
        if not np.any(mask):
            return 0.0
        
        background_energy = np.sum(energy_density[mask]) * self.dx**3
        background_volume = np.sum(mask) * self.dx**3
        background_strength = background_energy / background_volume if background_volume > 0 else 0.0
        
        return float(background_strength)
    
    def _calculate_tail_overlap_factor(
        self, 
        R: np.ndarray, 
        energy_density: np.ndarray, 
        core_radius: float, 
        rarefaction_radius: float
    ) -> float:
        """Calculate tail overlap factor."""
        # Measure how much energy is in overlap region vs individual tails
        core_energy = np.sum(energy_density[R <= core_radius]) * self.dx**3
        tail_energy = np.sum(energy_density[(R > core_radius) & (R <= rarefaction_radius)]) * self.dx**3
        
        if core_energy == 0:
            return 0.0
        
        # Overlap factor: ratio of tail energy to core energy
        overlap_factor = tail_energy / core_energy
        return float(overlap_factor)
    
    def analyze_defect_tails(
        self, 
        energy_density: Any,
        n_defects: int = 1
    ) -> List[DefectTail]:
        """
        Analyze individual defect tails.
        
        Args:
            energy_density: Energy density
            n_defects: Number of defects (for overlap calculation)
            
        Returns:
            List of defect tail structures
        """
        # Convert to numpy
        R_np = self.R.get() if hasattr(self.R, 'get') else self.R
        e_total = energy_density.total_density.get() if hasattr(energy_density.total_density, 'get') else energy_density.total_density
        
        tails = []
        
        # For single defect, analyze radial structure
        if n_defects == 1:
            tail = self._analyze_single_defect_tail(R_np, e_total)
            tails.append(tail)
        else:
            # For multiple defects, analyze overlap regions
            for i in range(n_defects):
                tail = self._analyze_defect_tail_with_overlap(R_np, e_total, i, n_defects)
                tails.append(tail)
        
        return tails
    
    def _analyze_single_defect_tail(
        self, 
        R: np.ndarray, 
        energy_density: np.ndarray
    ) -> DefectTail:
        """Analyze single defect tail structure."""
        # Find tail region (outside core)
        core_radius = self._find_core_radius(R, energy_density, 0.8)
        tail_mask = R > core_radius
        
        if not np.any(tail_mask):
            return DefectTail(0.0, 0.0, 0.0, 0, 0.0)
        
        # Calculate tail properties
        tail_radius = np.max(R[tail_mask])
        tail_energy = np.sum(energy_density[tail_mask]) * self.dx**3
        
        # Calculate decay length (exponential fit)
        decay_length = self._fit_exponential_decay(R[tail_mask], energy_density[tail_mask])
        
        # Count modes in tail
        mode_count = self._count_tail_modes(R[tail_mask], energy_density[tail_mask])
        
        # Overlap region (for single defect, this is the tail itself)
        overlap_region = tail_radius - core_radius
        
        return DefectTail(
            tail_radius=tail_radius,
            tail_energy=tail_energy,
            decay_length=decay_length,
            mode_count=mode_count,
            overlap_region=overlap_region
        )
    
    def _analyze_defect_tail_with_overlap(
        self, 
        R: np.ndarray, 
        energy_density: np.ndarray, 
        defect_index: int, 
        total_defects: int
    ) -> DefectTail:
        """Analyze defect tail with overlap from other defects."""
        # Simplified analysis for multiple defects
        # In real implementation, would need to separate individual defect contributions
        
        core_radius = self._find_core_radius(R, energy_density, 0.8)
        tail_mask = R > core_radius
        
        if not np.any(tail_mask):
            return DefectTail(0.0, 0.0, 0.0, 0, 0.0)
        
        # Calculate properties (scaled for multiple defects)
        tail_radius = np.max(R[tail_mask])
        tail_energy = np.sum(energy_density[tail_mask]) * self.dx**3 / total_defects
        decay_length = self._fit_exponential_decay(R[tail_mask], energy_density[tail_mask])
        mode_count = self._count_tail_modes(R[tail_mask], energy_density[tail_mask])
        
        # Overlap region increases with number of defects
        overlap_region = (tail_radius - core_radius) * np.sqrt(total_defects)
        
        return DefectTail(
            tail_radius=tail_radius,
            tail_energy=tail_energy,
            decay_length=decay_length,
            mode_count=mode_count,
            overlap_region=overlap_region
        )
    
    def _fit_exponential_decay(self, R: np.ndarray, energy_density: np.ndarray) -> float:
        """Fit exponential decay to tail energy."""
        if len(R) < 2:
            return 0.0
        
        # Simple exponential fit: E(r) = E0 * exp(-r/λ)
        # Use log-linear regression
        valid_mask = energy_density > 1e-10
        if not np.any(valid_mask):
            return 0.0
        
        R_valid = R[valid_mask]
        E_valid = energy_density[valid_mask]
        
        # Log-linear fit
        log_E = np.log(E_valid)
        coeffs = np.polyfit(R_valid, log_E, 1)
        decay_length = -1.0 / coeffs[0] if coeffs[0] != 0 else 0.0
        
        return float(decay_length)
    
    def _count_tail_modes(self, R: np.ndarray, energy_density: np.ndarray) -> int:
        """Count quantized modes in tail region."""
        if len(R) < 3:
            return 0
        
        # Create radial bins
        r_bins = np.linspace(np.min(R), np.max(R), 20)
        mode_count = 0
        
        for i in range(len(r_bins) - 1):
            mask = (R >= r_bins[i]) & (R < r_bins[i+1])
            if np.any(mask):
                shell_energy = np.sum(energy_density[mask])
                if shell_energy > 0.01 * np.sum(energy_density):
                    mode_count += 1
        
        return mode_count
    
    def generate_rarefaction_report(self, rarefaction_zone: RarefactionZone, defect_tails: List[DefectTail]) -> str:
        """Generate detailed rarefaction zone report."""
        report = []
        report.append("=" * 70)
        report.append("RAREFACTION ZONE AND TAIL OVERLAP ANALYSIS")
        report.append("=" * 70)
        
        report.append(f"\nZONE STRUCTURE:")
        report.append(f"  Core radius: {rarefaction_zone.core_radius:.4f} fm")
        report.append(f"  Rarefaction radius: {rarefaction_zone.rarefaction_radius:.4f} fm")
        report.append(f"  Vacuum radius: {rarefaction_zone.vacuum_radius:.4f} fm")
        
        report.append(f"\nENERGY DISTRIBUTION:")
        report.append(f"  Rarefaction energy: {rarefaction_zone.rarefaction_energy:.3f} MeV")
        report.append(f"  Background field strength: {rarefaction_zone.background_field_strength:.6f} MeV/fm³")
        report.append(f"  Tail overlap factor: {rarefaction_zone.tail_overlap_factor:.4f}")
        
        report.append(f"\nDEFECT TAILS ({len(defect_tails)} defects):")
        report.append("Defect | Tail R (fm) | Energy (MeV) | Decay λ (fm) | Modes | Overlap (fm)")
        report.append("-" * 75)
        
        for i, tail in enumerate(defect_tails):
            report.append(
                f"{i+1:6d} | {tail.tail_radius:10.4f} | {tail.tail_energy:11.3f} | "
                f"{tail.decay_length:11.4f} | {tail.mode_count:5d} | {tail.overlap_region:11.4f}"
            )
        
        report.append(f"\nPHYSICAL INTERPRETATION:")
        report.append(f"  The rarefaction zone extends from {rarefaction_zone.core_radius:.4f} to {rarefaction_zone.rarefaction_radius:.4f} fm.")
        report.append(f"  This region contains {rarefaction_zone.rarefaction_energy:.3f} MeV of energy")
        report.append(f"  from overlapping defect tails, creating quantized background field.")
        report.append(f"  The overlap factor {rarefaction_zone.tail_overlap_factor:.4f} indicates")
        report.append(f"  the relative strength of tail interactions.")
        
        return "\n".join(report)
