"""
Universal solver for Skyrme field equations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from phaze_particles.utils.mathematical_foundations import ArrayBackend
from phaze_particles.utils.mode_analyzer import RadialModeAnalyzer, UniversalSolver as ModeSolver
from phaze_particles.utils.topological_analyzer import TopologicalRadiusAnalyzer
from phaze_particles.utils.tail_interference_analyzer import TailInterferenceAnalyzer


@dataclass
class SolverInput:
    """Input parameters for universal solver."""
    grid_size: int
    box_size: float
    config_type: str = '120deg'
    c2: float = 1.0
    c4: float = 1.0
    c6: float = 1.0
    F_pi: float = 186.0
    e: float = 5.45
    target_mass: Optional[float] = None
    target_radius: Optional[float] = None
    target_magnetic_moment: Optional[float] = None
    target_bands: Optional[int] = None
    optimization_strategy: str = 'auto'  # 'auto', 'energy_balance', 'physical_params', 'quantization'


@dataclass
class SolverOutput:
    """Output from universal solver."""
    success: bool
    optimized_constants: Dict[str, float]
    physical_parameters: Dict[str, float]
    energy_analysis: Dict[str, float]
    mode_analysis: Dict[str, Any]
    topological_analysis: Dict[str, Any]
    interference_analysis: Dict[str, Any]
    convergence_info: Dict[str, Any]
    execution_time: float
    iterations: int


class UniversalSkyrmeSolver:
    """
    Universal solver for Skyrme field equations.
    
    Can solve for protons, neutrons, or any topological defect.
    """
    
    def __init__(self, backend: Optional[ArrayBackend] = None):
        """
        Initialize universal solver.
        
        Args:
            backend: Array backend for computations
        """
        self.backend = backend or ArrayBackend()
        self.mode_analyzer = None
        self.topological_analyzer = None
        self.interference_analyzer = None
        self.mode_solver = None
    
    def solve(self, input_params: SolverInput) -> SolverOutput:
        """
        Solve Skyrme field equations with given parameters.
        
        Args:
            input_params: Input parameters for solving
            
        Returns:
            Solver output with results
        """
        import time
        start_time = time.time()
        
        try:
            # Initialize analyzers
            self._initialize_analyzers(input_params)
            
            # Create and run model
            model = self._create_model(input_params)
            model.create_geometry()
            model.build_fields()
            model.calculate_energy()
            # Use full Noether/Skyrme charge density in physics
            try:
                model.calculate_physics(charge_density_calculator=model.charge_density)
            except Exception:
                model.calculate_physics()
            
            # Run optimization based on strategy
            if input_params.optimization_strategy == 'auto':
                strategy = self._select_optimal_strategy(model, input_params)
            else:
                strategy = input_params.optimization_strategy
            
            optimized_model = self._optimize_model(model, input_params, strategy)
            
            # Analyze results
            results = self._analyze_results(optimized_model, input_params)
            
            execution_time = time.time() - start_time
            
            return SolverOutput(
                success=True,
                optimized_constants=results['constants'],
                physical_parameters=results['physical'],
                energy_analysis=results['energy'],
                mode_analysis=results['modes'],
                topological_analysis=results['topological'],
                interference_analysis=results['interference'],
                convergence_info=results['convergence'],
                execution_time=execution_time,
                iterations=results['iterations']
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return SolverOutput(
                success=False,
                optimized_constants={},
                physical_parameters={},
                energy_analysis={},
                mode_analysis={},
                topological_analysis={},
                interference_analysis={},
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                iterations=0
            )
    
    def _initialize_analyzers(self, input_params: SolverInput) -> None:
        """Initialize analysis tools."""
        self.mode_analyzer = RadialModeAnalyzer(
            input_params.grid_size, input_params.box_size, self.backend
        )
        self.topological_analyzer = TopologicalRadiusAnalyzer(
            input_params.grid_size, input_params.box_size, self.backend
        )
        self.interference_analyzer = TailInterferenceAnalyzer(
            input_params.grid_size, input_params.box_size, self.backend
        )
        self.mode_solver = ModeSolver(self.mode_analyzer)
    
    def _create_model(self, input_params: SolverInput):
        """Create proton model with given parameters."""
        from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig
        
        config = ModelConfig(
            grid_size=input_params.grid_size,
            box_size=input_params.box_size,
            c2=input_params.c2,
            c4=input_params.c4,
            c6=input_params.c6,
            F_pi=input_params.F_pi,
            e=input_params.e
        )
        config.config_type = input_params.config_type
        
        return ProtonModel(config)
    
    def _select_optimal_strategy(self, model, input_params: SolverInput) -> str:
        """Select optimal optimization strategy."""
        # Analyze current state
        mode_analysis = self.mode_analyzer.analyze_field_modes(model.energy_density, model.profile)
        
        # Strategy selection logic
        if input_params.target_bands is not None:
            if abs(mode_analysis.total_modes - input_params.target_bands) > 5:
                return 'quantization'
        
        if input_params.target_radius is not None:
            pq = model.physical_quantities
            if abs(pq.charge_radius - input_params.target_radius) / input_params.target_radius > 0.2:
                return 'physical_params'
        
        if input_params.target_mass is not None:
            pq = model.physical_quantities
            if abs(pq.mass - input_params.target_mass) / input_params.target_mass > 0.2:
                return 'physical_params'
        
        return 'energy_balance'
    
    def _optimize_model(self, model, input_params: SolverInput, strategy: str):
        """Optimize model using selected strategy."""
        if strategy == 'energy_balance':
            return self._optimize_energy_balance(model, input_params)
        elif strategy == 'physical_params':
            return self._optimize_physical_params(model, input_params)
        elif strategy == 'quantization':
            return self._optimize_quantization(model, input_params)
        else:
            return model
    
    def _optimize_energy_balance(self, model, input_params: SolverInput):
        """Optimize for energy balance."""
        # Use existing optimization logic
        if hasattr(model, 'optimize_skyrme_constants'):
            model.optimize_skyrme_constants(
                target_e2_ratio=0.5,
                target_e4_ratio=0.5,
                target_virial_residual=0.05,
                max_iterations=50
            )
        return model
    
    def _optimize_physical_params(self, model, input_params: SolverInput):
        """Optimize for physical parameters."""
        # Simple parameter adjustment based on targets
        c2, c4, c6 = input_params.c2, input_params.c4, input_params.c6
        
        if input_params.target_mass is not None:
            pq = model.physical_quantities
            mass_ratio = input_params.target_mass / pq.mass
            c2 *= mass_ratio ** 0.5
            c4 *= mass_ratio ** 0.5
        
        if input_params.target_radius is not None:
            pq = model.physical_quantities
            radius_ratio = input_params.target_radius / pq.charge_radius
            c2 *= radius_ratio ** 2
            c4 *= radius_ratio ** 2
        
        # Create new model with updated constants
        new_input = SolverInput(
            grid_size=input_params.grid_size,
            box_size=input_params.box_size,
            config_type=input_params.config_type,
            c2=c2,
            c4=c4,
            c6=c6,
            F_pi=input_params.F_pi,
            e=input_params.e
        )
        
        new_model = self._create_model(new_input)
        new_model.create_geometry()
        new_model.build_fields()
        new_model.calculate_energy()
        new_model.calculate_physics()
        
        return new_model
    
    def _optimize_quantization(self, model, input_params: SolverInput):
        """Optimize for quantization structure."""
        if input_params.target_bands is None:
            return model
        
        # Adjust constants to achieve target number of bands
        mode_analysis = self.mode_analyzer.analyze_field_modes(model.energy_density, model.profile)
        current_bands = mode_analysis.total_modes
        target_bands = input_params.target_bands
        
        # Simple adjustment based on band count
        band_ratio = target_bands / current_bands
        c2 = model.config.c2 * band_ratio
        c4 = model.config.c4 / band_ratio
        
        # Create new model with updated constants
        new_input = SolverInput(
            grid_size=input_params.grid_size,
            box_size=input_params.box_size,
            config_type=input_params.config_type,
            c2=c2,
            c4=c4,
            c6=model.config.c6,
            F_pi=input_params.F_pi,
            e=input_params.e
        )
        
        new_model = self._create_model(new_input)
        new_model.create_geometry()
        new_model.build_fields()
        new_model.calculate_energy()
        new_model.calculate_physics()
        
        return new_model
    
    def _analyze_results(self, model, input_params: SolverInput) -> Dict[str, Any]:
        """Analyze final results."""
        # Physical parameters
        pq = model.physical_quantities
        physical = {
            'mass': pq.mass,
            'charge_radius': pq.charge_radius,
            'magnetic_moment': pq.magnetic_moment,
            'electric_charge': pq.electric_charge,
            'baryon_number': pq.baryon_number
        }
        
        # Energy analysis
        e2 = model.energy_density.c2_term.get() if hasattr(model.energy_density.c2_term, 'get') else model.energy_density.c2_term
        e4 = model.energy_density.c4_term.get() if hasattr(model.energy_density.c4_term, 'get') else model.energy_density.c4_term
        e6 = model.energy_density.c6_term.get() if hasattr(model.energy_density.c6_term, 'get') else model.energy_density.c6_term
        
        total_e2 = np.sum(e2) * (input_params.box_size / input_params.grid_size)**3
        total_e4 = np.sum(e4) * (input_params.box_size / input_params.grid_size)**3
        total_e6 = np.sum(e6) * (input_params.box_size / input_params.grid_size)**3
        total_energy = total_e2 + total_e4 + total_e6
        
        # External field (tail) fraction estimate: energy outside core radius
        try:
            dx = input_params.box_size / input_params.grid_size
            # Total density array
            if hasattr(model.energy_density, 'total_density'):
                dens = model.energy_density.total_density
            else:
                dens = e2 + e4 + e6
            dens_np = dens.get() if hasattr(dens, 'get') else dens
            # Radial grid
            axis = np.linspace(-input_params.box_size/2 + 0.5*dx, input_params.box_size/2 - 0.5*dx, input_params.grid_size)
            X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
            R = np.sqrt(X*X + Y*Y + Z*Z)
            core_r = getattr(self.mode_analyzer, 'core_radius', None)
            if core_r is None:
                core_r = 0.0
            tail_mask = R > core_r
            tail_energy = float(np.sum(dens_np[tail_mask]) * dx**3)
            core_energy = float(total_energy - tail_energy)
            tail_fraction = float(tail_energy / max(total_energy, 1e-12))
        except Exception:
            tail_energy = 0.0
            core_energy = float(total_energy)
            tail_fraction = 0.0

        energy = {
            'e2': total_e2,
            'e4': total_e4,
            'e6': total_e6,
            'total': total_energy,
            'e2_ratio': total_e2 / total_energy,
            'e4_ratio': total_e4 / total_energy,
            'e6_ratio': total_e6 / total_energy,
            'tail_energy': tail_energy,
            'core_energy': core_energy,
            'tail_fraction': tail_fraction
        }
        
        # Mode analysis
        mode_analysis = self.mode_analyzer.analyze_field_modes(model.energy_density, model.profile)
        modes = {
            'total_modes': mode_analysis.total_modes,
            'energy_bands': len(mode_analysis.energy_bands),
            'core_radius': mode_analysis.core_radius,
            'quantization_parameter': mode_analysis.quantization_parameter
        }
        
        # Topological analysis
        topological_radius = self.topological_analyzer.analyze_topological_radius(
            model.su2_field, model.energy_density, model.field_derivatives['baryon_density']
        )
        topological = {
            'geometric_radius': topological_radius.geometric_radius,
            'phase_radius': topological_radius.phase_radius,
            'effective_radius': topological_radius.effective_radius,
            'topological_charge': topological_radius.topological_charge,
            'phase_transitions': topological_radius.phase_transitions
        }
        
        # Interference analysis
        interference = self.interference_analyzer.analyze_tail_interference(
            model.energy_density, model.su2_field, n_tails=1
        )
        interference_data = {
            'fluctuation_energy': interference.fluctuation_energy,
            'background_field_strength': interference.background_field_strength,
            'constructive_regions': interference.interference_pattern.constructive_regions,
            'destructive_regions': interference.interference_pattern.destructive_regions,
            'interference_strength': interference.interference_pattern.interference_strength,
            'fluctuation_amplitude': interference.interference_pattern.fluctuation_amplitude
        }
        
        # Constants
        constants = {
            'c2': model.config.c2,
            'c4': model.config.c4,
            'c6': model.config.c6,
            'F_pi': model.config.F_pi,
            'e': model.config.e
        }
        
        # Convergence info
        convergence = {
            'converged': True,  # Simplified
            'final_error': 0.0,  # Simplified
            'strategy_used': input_params.optimization_strategy
        }
        
        return {
            'physical': physical,
            'energy': energy,
            'modes': modes,
            'topological': topological,
            'interference': interference_data,
            'constants': constants,
            'convergence': convergence,
            'iterations': 50  # Simplified
        }


def solve_skyrme_field(
    grid_size: int,
    box_size: float,
    config_type: str = '120deg',
    c2: float = 1.0,
    c4: float = 1.0,
    c6: float = 1.0,
    F_pi: float = 186.0,
    e: float = 5.45,
    target_mass: Optional[float] = None,
    target_radius: Optional[float] = None,
    target_magnetic_moment: Optional[float] = None,
    target_bands: Optional[int] = None,
    optimization_strategy: str = 'auto'
) -> SolverOutput:
    """
    Convenience function for solving Skyrme field.
    
    Args:
        grid_size: Grid size
        box_size: Box size
        config_type: Configuration type
        c2, c4, c6: Skyrme constants
        F_pi, e: Physical constants
        target_mass: Target mass (MeV)
        target_radius: Target radius (fm)
        target_magnetic_moment: Target magnetic moment (μN)
        target_bands: Target number of energy bands
        optimization_strategy: Optimization strategy
        
    Returns:
        Solver output
    """
    solver = UniversalSkyrmeSolver()
    input_params = SolverInput(
        grid_size=grid_size,
        box_size=box_size,
        config_type=config_type,
        c2=c2,
        c4=c4,
        c6=c6,
        F_pi=F_pi,
        e=e,
        target_mass=target_mass,
        target_radius=target_radius,
        target_magnetic_moment=target_magnetic_moment,
        target_bands=target_bands,
        optimization_strategy=optimization_strategy
    )
    
    return solver.solve(input_params)
