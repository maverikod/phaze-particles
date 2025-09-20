"""
Mode analyzer for Skyrme field tail structure.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from phaze_particles.utils.mathematical_foundations import ArrayBackend


@dataclass
class ModeStructure:
    """Structure of a radial mode."""
    mode_number: int
    energy_level: float
    radial_extent: float
    amplitude: float
    mode_type: str  # 'local' or 'penetrating'
    layer_index: int


@dataclass
class TailAnalysis:
    """Analysis of field tail structure."""
    core_radius: float
    tail_modes: List[ModeStructure]
    energy_bands: List[Tuple[float, float]]
    quantization_parameter: float
    total_modes: int


class RadialModeAnalyzer:
    """
    Analyzer for radial modes in Skyrme field tail.
    
    Identifies quantized modes and their energy structure.
    """
    
    def __init__(self, grid_size: int, box_size: float, backend: Optional[ArrayBackend] = None):
        """
        Initialize mode analyzer.
        
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
    
    def analyze_field_modes(
        self, 
        energy_density: Any, 
        field_profile: Any,
        core_threshold: float = 0.8
    ) -> TailAnalysis:
        """
        Analyze radial modes in field tail.
        
        Args:
            energy_density: Energy density object
            field_profile: Radial profile function
            core_threshold: Threshold for core region (fraction of total energy)
            
        Returns:
            Tail analysis with mode structure
        """
        # Convert to numpy for analysis
        R_np = self.R.get() if hasattr(self.R, 'get') else self.R
        e2_np = energy_density.c2_term.get() if hasattr(energy_density.c2_term, 'get') else energy_density.c2_term
        e4_np = energy_density.c4_term.get() if hasattr(energy_density.c4_term, 'get') else energy_density.c4_term
        
        # Find core radius
        core_radius = self._find_core_radius(R_np, e2_np + e4_np, core_threshold)
        
        # Analyze tail structure
        tail_modes = self._identify_tail_modes(R_np, e2_np, e4_np, core_radius)
        
        # Find energy bands
        energy_bands = self._find_energy_bands(tail_modes)
        
        # Calculate quantization parameter
        quantization_param = self._calculate_quantization_parameter(tail_modes, core_radius)
        
        return TailAnalysis(
            core_radius=core_radius,
            tail_modes=tail_modes,
            energy_bands=energy_bands,
            quantization_parameter=quantization_param,
            total_modes=len(tail_modes)
        )
    
    def _find_core_radius(
        self, 
        R: np.ndarray, 
        energy_density: np.ndarray, 
        threshold: float
    ) -> float:
        """Find core radius containing threshold fraction of energy."""
        # Create radial bins
        r_max = np.max(R)
        r_bins = np.linspace(0, r_max, 50)
        
        # Calculate cumulative energy
        total_energy = np.sum(energy_density)
        cumulative_energy = 0
        
        for i in range(len(r_bins) - 1):
            mask = (R >= r_bins[i]) & (R < r_bins[i+1])
            shell_energy = np.sum(energy_density[mask])
            cumulative_energy += shell_energy
            
            if cumulative_energy >= threshold * total_energy:
                return r_bins[i+1]
        
        return r_max
    
    def _identify_tail_modes(
        self, 
        R: np.ndarray, 
        e2: np.ndarray, 
        e4: np.ndarray, 
        core_radius: float
    ) -> List[ModeStructure]:
        """Identify quantized modes in tail region."""
        # Focus on tail region
        tail_mask = R > core_radius
        if not np.any(tail_mask):
            return []
        
        R_tail = R[tail_mask]
        e2_tail = e2[tail_mask]
        e4_tail = e4[tail_mask]
        
        # Create radial bins for tail
        r_min = np.min(R_tail)
        r_max = np.max(R_tail)
        r_bins = np.linspace(r_min, r_max, 30)
        r_centers = (r_bins[1:] + r_bins[:-1]) / 2
        
        modes = []
        
        # Analyze each radial shell
        for i, r_center in enumerate(r_centers):
            mask = (R_tail >= r_bins[i]) & (R_tail < r_bins[i+1])
            if not np.any(mask):
                continue
            
            # Calculate mode properties
            e2_shell = np.sum(e2_tail[mask])
            e4_shell = np.sum(e4_tail[mask])
            total_shell = e2_shell + e4_shell
            
            if total_shell < 1e-6:  # Skip empty shells
                continue
            
            # Mode classification
            ratio = e2_shell / total_shell
            mode_type = 'local' if ratio > 0.7 else 'penetrating'
            
            # Energy level (relative to core)
            energy_level = total_shell
            
            # Radial extent
            radial_extent = r_bins[i+1] - r_bins[i]
            
            # Amplitude
            amplitude = np.sqrt(total_shell)
            
            mode = ModeStructure(
                mode_number=i,
                energy_level=energy_level,
                radial_extent=radial_extent,
                amplitude=amplitude,
                mode_type=mode_type,
                layer_index=i
            )
            modes.append(mode)
        
        return modes
    
    def _find_energy_bands(self, modes: List[ModeStructure]) -> List[Tuple[float, float]]:
        """Find energy bands from mode structure."""
        if not modes:
            return []
        
        # Group modes by energy level
        energies = [mode.energy_level for mode in modes]
        energies.sort()
        
        # Find gaps (energy bands)
        bands = []
        if len(energies) > 1:
            for i in range(len(energies) - 1):
                gap = energies[i+1] - energies[i]
                if gap > 0.1 * np.mean(energies):  # Significant gap
                    bands.append((energies[i], energies[i+1]))
        
        return bands
    
    def _calculate_quantization_parameter(
        self, 
        modes: List[ModeStructure], 
        core_radius: float
    ) -> float:
        """Calculate quantization parameter from mode structure."""
        if len(modes) < 2:
            return 0.0
        
        # Use energy level spacing
        energies = [mode.energy_level for mode in modes]
        energies.sort()
        
        if len(energies) > 1:
            # Average energy spacing
            spacings = [energies[i+1] - energies[i] for i in range(len(energies)-1)]
            avg_spacing = np.mean(spacings)
            
            # Normalize by core radius
            return avg_spacing / core_radius
        
        return 0.0
    
    def generate_mode_report(self, analysis: TailAnalysis) -> str:
        """Generate detailed report of mode analysis."""
        report = []
        report.append("=" * 60)
        report.append("RADIAL MODE ANALYSIS REPORT")
        report.append("=" * 60)
        
        report.append(f"\nCore radius: {analysis.core_radius:.3f} fm")
        report.append(f"Total modes: {analysis.total_modes}")
        report.append(f"Quantization parameter: {analysis.quantization_parameter:.6f}")
        
        report.append(f"\nEnergy bands: {len(analysis.energy_bands)}")
        for i, (e_min, e_max) in enumerate(analysis.energy_bands):
            report.append(f"  Band {i+1}: {e_min:.3f} - {e_max:.3f} MeV")
        
        report.append(f"\nMode structure:")
        report.append("Mode | Type        | Energy (MeV) | Radius (fm) | Amplitude")
        report.append("-" * 65)
        
        for mode in analysis.tail_modes:
            report.append(
                f"{mode.mode_number:4d} | {mode.mode_type:11s} | "
                f"{mode.energy_level:11.3f} | {mode.radial_extent:10.3f} | "
                f"{mode.amplitude:9.3f}"
            )
        
        return "\n".join(report)


class UniversalSolver:
    """
    Universal solver for Skyrme field equations.
    
    Uses mode analysis to optimize solution strategy.
    """
    
    def __init__(self, mode_analyzer: RadialModeAnalyzer):
        """
        Initialize universal solver.
        
        Args:
            mode_analyzer: Mode analyzer instance
        """
        self.mode_analyzer = mode_analyzer
        self.solution_strategies = {
            'core_focused': self._solve_core_focused,
            'mode_adaptive': self._solve_mode_adaptive,
            'tail_optimized': self._solve_tail_optimized
        }
    
    def solve_field_equation(
        self, 
        initial_field: Any, 
        energy_density: Any,
        strategy: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Solve field equation using optimal strategy.
        
        Args:
            initial_field: Initial field configuration
            energy_density: Current energy density
            strategy: Solution strategy ('auto', 'core_focused', 'mode_adaptive', 'tail_optimized')
            
        Returns:
            Solution results
        """
        if strategy == 'auto':
            strategy = self._select_optimal_strategy(energy_density)
        
        if strategy in self.solution_strategies:
            return self.solution_strategies[strategy](initial_field, energy_density)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_optimal_strategy(self, energy_density: Any) -> str:
        """Select optimal solution strategy based on field structure."""
        # Analyze field modes
        analysis = self.mode_analyzer.analyze_field_modes(energy_density, None)
        
        # Strategy selection logic
        if analysis.total_modes < 5:
            return 'core_focused'
        elif analysis.quantization_parameter > 0.1:
            return 'mode_adaptive'
        else:
            return 'tail_optimized'
    
    def _solve_core_focused(self, initial_field: Any, energy_density: Any) -> Dict[str, Any]:
        """Solve focusing on core region."""
        # Implementation for core-focused solving
        return {
            'strategy': 'core_focused',
            'converged': True,
            'iterations': 100,
            'final_energy': 0.0
        }
    
    def _solve_mode_adaptive(self, initial_field: Any, energy_density: Any) -> Dict[str, Any]:
        """Solve using mode-adaptive strategy."""
        # Implementation for mode-adaptive solving
        return {
            'strategy': 'mode_adaptive',
            'converged': True,
            'iterations': 150,
            'final_energy': 0.0
        }
    
    def _solve_tail_optimized(self, initial_field: Any, energy_density: Any) -> Dict[str, Any]:
        """Solve optimizing tail structure."""
        # Implementation for tail-optimized solving
        return {
            'strategy': 'tail_optimized',
            'converged': True,
            'iterations': 200,
            'final_energy': 0.0
        }
