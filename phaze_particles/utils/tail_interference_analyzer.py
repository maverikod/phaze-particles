"""
Tail interference analyzer for Skyrme field fluctuations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from phaze_particles.utils.mathematical_foundations import ArrayBackend


@dataclass
class InterferencePattern:
    """Interference pattern from defect tails."""
    constructive_regions: int
    destructive_regions: int
    interference_strength: float
    phase_correlation: float
    fluctuation_amplitude: float


@dataclass
class TailInterference:
    """Tail interference analysis."""
    n_tails: int
    tail_separation: float
    interference_pattern: InterferencePattern
    fluctuation_energy: float
    background_field_strength: float


class TailInterferenceAnalyzer:
    """
    Analyzer for tail interference and fluctuations.
    
    Studies how defect tails interfere to create quantized modes.
    """
    
    def __init__(self, grid_size: int, box_size: float, backend: Optional[ArrayBackend] = None):
        """
        Initialize tail interference analyzer.
        
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
    
    def analyze_tail_interference(
        self, 
        energy_density: Any, 
        su2_field: Any,
        n_tails: int = 1
    ) -> TailInterference:
        """
        Analyze tail interference patterns.
        
        Args:
            energy_density: Energy density
            su2_field: SU(2) field
            n_tails: Number of interfering tails
            
        Returns:
            Tail interference analysis
        """
        # Convert to numpy
        R_np = self.R.get() if hasattr(self.R, 'get') else self.R
        e_total = energy_density.total_density.get() if hasattr(energy_density.total_density, 'get') else energy_density.total_density
        
        # Analyze interference pattern
        interference_pattern = self._analyze_interference_pattern(R_np, e_total, su2_field)
        
        # Calculate tail separation
        tail_separation = self._calculate_tail_separation(R_np, e_total, n_tails)
        
        # Calculate fluctuation energy
        fluctuation_energy = self._calculate_fluctuation_energy(R_np, e_total)
        
        # Calculate background field strength
        background_strength = self._calculate_background_strength(R_np, e_total)
        
        return TailInterference(
            n_tails=n_tails,
            tail_separation=tail_separation,
            interference_pattern=interference_pattern,
            fluctuation_energy=fluctuation_energy,
            background_field_strength=background_strength
        )
    
    def _analyze_interference_pattern(
        self, 
        R: np.ndarray, 
        energy_density: np.ndarray, 
        su2_field: Any
    ) -> InterferencePattern:
        """Analyze interference pattern from tail overlap."""
        # Find tail region (outside core)
        core_radius = self._find_core_radius(R, energy_density, 0.8)
        tail_mask = R > core_radius
        
        if not np.any(tail_mask):
            return InterferencePattern(0, 0, 0.0, 0.0, 0.0)
        
        # Analyze energy fluctuations in tail
        tail_energy = energy_density[tail_mask]
        tail_radius = R[tail_mask]
        
        # Find constructive and destructive regions
        constructive_regions, destructive_regions = self._find_interference_regions(
            tail_radius, tail_energy
        )
        
        # Calculate interference strength
        interference_strength = self._calculate_interference_strength(tail_energy)
        
        # Calculate phase correlation
        phase_correlation = self._calculate_phase_correlation(su2_field, tail_mask)
        
        # Calculate fluctuation amplitude
        fluctuation_amplitude = self._calculate_fluctuation_amplitude(tail_energy)
        
        return InterferencePattern(
            constructive_regions=constructive_regions,
            destructive_regions=destructive_regions,
            interference_strength=interference_strength,
            phase_correlation=phase_correlation,
            fluctuation_amplitude=fluctuation_amplitude
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
    
    def _find_interference_regions(
        self, 
        R: np.ndarray, 
        energy_density: np.ndarray
    ) -> Tuple[int, int]:
        """Find constructive and destructive interference regions."""
        # Create radial bins
        r_bins = np.linspace(np.min(R), np.max(R), 30)
        r_centers = (r_bins[1:] + r_bins[:-1]) / 2
        
        # Calculate average energy in each bin
        bin_energies = []
        for i in range(len(r_bins) - 1):
            mask = (R >= r_bins[i]) & (R < r_bins[i+1])
            if np.any(mask):
                bin_energy = np.mean(energy_density[mask])
                bin_energies.append(bin_energy)
            else:
                bin_energies.append(0.0)
        
        bin_energies = np.array(bin_energies)
        
        # Find regions above and below average
        mean_energy = np.mean(bin_energies[bin_energies > 0])
        if mean_energy == 0:
            return 0, 0
        
        # Constructive regions (above average)
        constructive_regions = np.sum(bin_energies > 1.2 * mean_energy)
        
        # Destructive regions (below average)
        destructive_regions = np.sum(bin_energies < 0.8 * mean_energy)
        
        return int(constructive_regions), int(destructive_regions)
    
    def _calculate_interference_strength(self, energy_density: np.ndarray) -> float:
        """Calculate interference strength from energy fluctuations."""
        if len(energy_density) < 2:
            return 0.0
        
        # Interference strength = relative standard deviation
        mean_energy = np.mean(energy_density)
        if mean_energy == 0:
            return 0.0
        
        std_energy = np.std(energy_density)
        interference_strength = std_energy / mean_energy
        
        return float(interference_strength)
    
    def _calculate_phase_correlation(self, su2_field: Any, tail_mask: np.ndarray) -> float:
        """Calculate phase correlation in tail region."""
        # Extract field components
        u_00 = su2_field.u_00.get() if hasattr(su2_field.u_00, 'get') else su2_field.u_00
        u_01 = su2_field.u_01.get() if hasattr(su2_field.u_01, 'get') else su2_field.u_01
        u_10 = su2_field.u_10.get() if hasattr(su2_field.u_10, 'get') else su2_field.u_10
        u_11 = su2_field.u_11.get() if hasattr(su2_field.u_11, 'get') else su2_field.u_11
        
        # Calculate phase
        det = u_00 * u_11 - u_01 * u_10
        phase = np.angle(det)
        
        # Calculate phase correlation in tail
        tail_phase = phase[tail_mask]
        if len(tail_phase) < 2:
            return 0.0
        
        # Phase correlation = how much phase varies
        phase_correlation = np.std(tail_phase) / (2 * np.pi)
        
        return float(phase_correlation)
    
    def _calculate_fluctuation_amplitude(self, energy_density: np.ndarray) -> float:
        """Calculate amplitude of energy fluctuations."""
        if len(energy_density) < 2:
            return 0.0
        
        # Fluctuation amplitude = range of energy variations
        max_energy = np.max(energy_density)
        min_energy = np.min(energy_density)
        mean_energy = np.mean(energy_density)
        
        if mean_energy == 0:
            return 0.0
        
        fluctuation_amplitude = (max_energy - min_energy) / mean_energy
        
        return float(fluctuation_amplitude)
    
    def _calculate_tail_separation(self, R: np.ndarray, energy_density: np.ndarray, n_tails: int) -> float:
        """Calculate average separation between tails."""
        if n_tails <= 1:
            return 0.0
        
        # For single defect, estimate tail separation from energy distribution
        # This is a simplified model - in reality would need multiple defects
        
        # Find characteristic tail length
        core_radius = self._find_core_radius(R, energy_density, 0.8)
        tail_mask = R > core_radius
        
        if not np.any(tail_mask):
            return 0.0
        
        tail_energy = energy_density[tail_mask]
        tail_radius = R[tail_mask]
        
        # Find radius where energy drops to 1/e of maximum
        max_tail_energy = np.max(tail_energy)
        if max_tail_energy == 0:
            return 0.0
        
        threshold_energy = max_tail_energy / np.e
        tail_length = 0.0
        
        for r in np.sort(tail_radius):
            mask = tail_radius <= r
            if np.any(mask):
                local_max = np.max(tail_energy[mask])
                if local_max <= threshold_energy:
                    tail_length = r - core_radius
                    break
        
        # Estimate separation for multiple tails
        # Assuming tails are arranged in a regular pattern
        if n_tails > 1:
            # Rough estimate: separation ≈ tail_length / sqrt(n_tails)
            separation = tail_length / np.sqrt(n_tails)
        else:
            separation = tail_length
        
        return float(separation)
    
    def _calculate_fluctuation_energy(self, R: np.ndarray, energy_density: np.ndarray) -> float:
        """Calculate energy in fluctuations."""
        # Find tail region
        core_radius = self._find_core_radius(R, energy_density, 0.8)
        tail_mask = R > core_radius
        
        if not np.any(tail_mask):
            return 0.0
        
        # Calculate fluctuation energy (energy above smooth background)
        tail_energy = energy_density[tail_mask]
        
        # Estimate smooth background (exponential decay)
        tail_radius = R[tail_mask]
        if len(tail_radius) < 2:
            return 0.0
        
        # Fit exponential background
        log_energy = np.log(tail_energy + 1e-10)
        coeffs = np.polyfit(tail_radius, log_energy, 1)
        background = np.exp(coeffs[0] * tail_radius + coeffs[1])
        
        # Fluctuation energy = total - background
        fluctuation_energy = np.sum(np.maximum(0, tail_energy - background)) * self.dx**3
        
        return float(fluctuation_energy)
    
    def _calculate_background_strength(self, R: np.ndarray, energy_density: np.ndarray) -> float:
        """Calculate background field strength from tail interference."""
        # Find tail region
        core_radius = self._find_core_radius(R, energy_density, 0.8)
        tail_mask = R > core_radius
        
        if not np.any(tail_mask):
            return 0.0
        
        # Calculate average energy density in tail
        tail_energy = energy_density[tail_mask]
        tail_volume = np.sum(tail_mask) * self.dx**3
        
        if tail_volume == 0:
            return 0.0
        
        background_strength = np.sum(tail_energy) * self.dx**3 / tail_volume
        
        return float(background_strength)
    
    def generate_interference_report(self, interference: TailInterference) -> str:
        """Generate detailed interference analysis report."""
        report = []
        report.append("=" * 70)
        report.append("TAIL INTERFERENCE AND FLUCTUATION ANALYSIS")
        report.append("=" * 70)
        
        report.append(f"\nINTERFERENCE STRUCTURE:")
        report.append(f"  Number of tails: {interference.n_tails}")
        report.append(f"  Tail separation: {interference.tail_separation:.4f} fm")
        report.append(f"  Fluctuation energy: {interference.fluctuation_energy:.3f} MeV")
        report.append(f"  Background field strength: {interference.background_field_strength:.6f} MeV/fm³")
        
        pattern = interference.interference_pattern
        report.append(f"\nINTERFERENCE PATTERN:")
        report.append(f"  Constructive regions: {pattern.constructive_regions}")
        report.append(f"  Destructive regions: {pattern.destructive_regions}")
        report.append(f"  Interference strength: {pattern.interference_strength:.4f}")
        report.append(f"  Phase correlation: {pattern.phase_correlation:.4f}")
        report.append(f"  Fluctuation amplitude: {pattern.fluctuation_amplitude:.4f}")
        
        report.append(f"\nPHYSICAL INTERPRETATION:")
        report.append(f"  The {interference.n_tails} defect tails interfere to create")
        report.append(f"  {pattern.constructive_regions} constructive and {pattern.destructive_regions} destructive regions.")
        report.append(f"  This interference produces {interference.fluctuation_energy:.3f} MeV of")
        report.append(f"  fluctuation energy, creating quantized background field.")
        report.append(f"  The interference strength {pattern.interference_strength:.4f} indicates")
        report.append(f"  the relative magnitude of tail interactions.")
        
        return "\n".join(report)
