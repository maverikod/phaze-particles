#!/usr/bin/env python3
"""
Integrated proton model implementation.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, fields
from enum import Enum
import json
import time

# Import all modules
from phaze_particles.utils.mathematical_foundations import MathematicalFoundations
from phaze_particles.utils.torus_geometries import (
    TorusGeometries,
    TorusConfiguration,
)
from phaze_particles.utils.su2_fields import SU2FieldBuilder
from phaze_particles.utils.energy_densities import EnergyDensityCalculator
from phaze_particles.utils.physics import (
    PhysicalQuantitiesCalculator,
    SkyrmeLagrangian,
    NoetherCurrent,
    ChargeDensity
)
from phaze_particles.utils.numerical_methods import (
    RelaxationSolver,
    RelaxationConfig,
    ConstraintConfig,
)
from phaze_particles.utils.validation import (
    ValidationSystem,
    ExperimentalData,
    CalculatedData,
)
from phaze_particles.utils.cuda import get_cuda_manager
from phaze_particles.utils.progress import create_progress_bar
from phaze_particles.utils.skyrme_optimizer import (
    SkyrmeConstantsOptimizer,
    OptimizationTargets,
    AdaptiveOptimizer
)
from phaze_particles.utils.universal_solver import (
    UniversalSkyrmeSolver,
    SolverInput,
    SolverOutput
)
from phaze_particles.utils.phase_environment import PhaseEnvironment
from phaze_particles.utils.phase_tail_analyzer import PhaseTailAnalyzer, PhaseTailResult


class ModelStatus(Enum):
    """Model status enumeration."""

    INITIALIZED = "initialized"
    GEOMETRY_CREATED = "geometry_created"
    FIELDS_BUILT = "fields_built"
    ENERGY_CALCULATED = "energy_calculated"
    PHYSICS_CALCULATED = "physics_calculated"
    OPTIMIZED = "optimized"
    VALIDATED = "validated"
    FAILED = "failed"


@dataclass
class ModelConfig:
    """Proton model configuration."""

    # Geometric parameters
    grid_size: int = 64
    box_size: float = 4.0
    torus_config: str = "120deg"
    R_torus: float = 1.0
    r_torus: float = 0.2

    # Profile parameters
    profile_type: str = "tanh"
    f_0: float = np.pi
    f_inf: float = 0.0
    r_scale: float = 0.6
    auto_tune_baryon: bool = True

    # Skyrme constants
    c2: float = 1.0
    c4: float = 1.0
    c6: float = 1.0
    
    # Physical constants for full Skyrme Lagrangian
    F_pi: float = 186.0  # MeV (pion decay constant)
    e: float = 5.45      # dimensionless Skyrme constant

    # Circumferential mode calibration factors (phase tail analysis)
    phase_density_factor: float = 1.0
    phase_velocity_factor: float = 1.0

    # Relaxation parameters
    max_iterations: int = 1000
    convergence_tol: float = 1e-6
    step_size: float = 0.01
    relaxation_method: str = "gradient_descent"

    # Constraint parameters
    lambda_B: float = 1000.0
    lambda_Q: float = 1000.0
    lambda_virial: float = 1000.0

    # Validation parameters
    validation_enabled: bool = True
    save_reports: bool = True
    output_dir: str = "results"

    # CUDA configuration (optional)
    cuda_device_id: Optional[int] = None
    cuda: Optional[Dict[str, Any]] = None

    @classmethod
    def from_file(cls, config_path: str) -> "ModelConfig":
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Model configuration
        """
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Filter only known fields to avoid errors on extra keys
        known_field_names = {f.name for f in fields(cls)}
        filtered: Dict[str, Any] = {}
        for key, value in config_data.items():
            if key in known_field_names:
                filtered[key] = value
            # Map common aliases
            elif key == "config_type":
                filtered["torus_config"] = value
            elif key == "convergence_tolerance":
                filtered["convergence_tol"] = value
            elif key == "output":
                filtered["output_dir"] = value
        return cls(**filtered)

    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to file.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors
        """
        errors = []

        if self.grid_size <= 0:
            errors.append("grid_size must be positive")

        if self.box_size <= 0:
            errors.append("box_size must be positive")

        if self.torus_config not in ["120deg", "clover", "cartesian"]:
            errors.append("torus_config must be one of: 120deg, clover, cartesian")

        if self.R_torus <= 0:
            errors.append("R_torus must be positive")

        if self.r_torus <= 0:
            errors.append("r_torus must be positive")

        if self.c2 <= 0 or self.c4 <= 0 or self.c6 < 0:
            errors.append("Skyrme constants c2, c4 must be positive, c6 must be non-negative")

        if self.max_iterations <= 0:
            errors.append("max_iterations must be positive")

        if self.convergence_tol <= 0:
            errors.append("convergence_tol must be positive")

        return errors


@dataclass
class ModelResults:
    """Proton model results."""

    # Main results
    status: ModelStatus
    execution_time: float
    iterations: int
    converged: bool

    # Physical parameters
    proton_mass: float
    charge_radius: float
    magnetic_moment: float
    electric_charge: float
    baryon_number: float
    energy_balance: float
    total_energy: float

    # Validation results
    validation_status: Optional[str] = None
    validation_score: Optional[float] = None

    # Additional information
    config: Optional[ModelConfig] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary with results
        """
        return asdict(self)

    def save_to_file(self, output_path: str) -> None:
        """
        Save results to file.

        Args:
            output_path: Path to results file
        """
        result_dict = self.to_dict()
        # Convert ModelStatus enum to string value
        if "status" in result_dict and hasattr(result_dict["status"], "value"):
            result_dict["status"] = result_dict["status"].value

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)


class ProtonModel:
    """Main proton model class."""

    def __init__(self, config: ModelConfig):
        """
        Initialize proton model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.status = ModelStatus.INITIALIZED

        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")

        # Initialize CUDA manager
        self.cuda_manager = get_cuda_manager()

        # Initialize backend
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        self.backend = ArrayBackend()

        # Initialize components
        self._initialize_components()

        # Results
        self.results: Optional[ModelResults] = None
        self.error_message: Optional[str] = None

    def _initialize_components(self) -> None:
        """Initialize all model components."""
        try:
            # Mathematical foundations
            self.math_foundations = MathematicalFoundations(
                grid_size=self.config.grid_size, box_size=self.config.box_size
            )

            # Torus geometries
            from phaze_particles.utils.torus_geometries import TorusGeometryManager
            self.torus_geometries = TorusGeometryManager(
                grid_size=self.config.grid_size, box_size=self.config.box_size
            )

            # SU(2) fields
            self.su2_field_builder = SU2FieldBuilder(
                grid_size=self.config.grid_size,
                box_size=self.config.box_size,
                backend=self.backend,
            )

            # Energy densities
            self.energy_calculator = EnergyDensityCalculator(
                grid_size=self.config.grid_size,
                box_size=self.config.box_size,
                c2=self.config.c2,
                c4=self.config.c4,
                c6=self.config.c6,
                backend=self.backend,
            )

            # Physical quantities
            self.physics_calculator = PhysicalQuantitiesCalculator(
                grid_size=self.config.grid_size,
                box_size=self.config.box_size,
                backend=self.backend,
            )
            
            # Full Skyrme physics components
            self.skyrme_lagrangian = SkyrmeLagrangian(
                F_pi=self.config.F_pi,
                e=self.config.e,
                c6=self.config.c6,
                backend=self.backend
            )
            
            self.noether_current = NoetherCurrent(
                F_pi=self.config.F_pi,
                e=self.config.e,
                c6=self.config.c6,
                backend=self.backend
            )
            
            self.charge_density = ChargeDensity(
                F_pi=self.config.F_pi,
                e=self.config.e,
                c6=self.config.c6,
                grid_size=self.config.grid_size,
                box_size=self.config.box_size,
                backend=self.backend
            )

            # Numerical methods
            relaxation_config = RelaxationConfig(
                method=self.config.relaxation_method,
                max_iterations=self.config.max_iterations,
                convergence_tol=self.config.convergence_tol,
                step_size=self.config.step_size,
            )

            constraint_config = ConstraintConfig(
                lambda_B=self.config.lambda_B,
                lambda_Q=self.config.lambda_Q,
                lambda_virial=self.config.lambda_virial,
            )

            self.relaxation_solver = RelaxationSolver(
                relaxation_config, constraint_config
            )

            # Validation system
            if self.config.validation_enabled:
                experimental_data = ExperimentalData()
                self.validation_system = ValidationSystem(experimental_data)
            else:
                self.validation_system = None
            
            # Phase environment (7D theory)
            self.phase_environment = PhaseEnvironment(backend=self.backend)
            
            # Инициализация анализатора фазовых хвостов
            self.phase_tail_analyzer = PhaseTailAnalyzer(self.backend)

            print("All model components successfully initialized")
            print(f"CUDA Status: {self.cuda_manager.get_status_string()}")

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Initialization error: {str(e)}"
            raise

    def create_geometry(self) -> bool:
        """
        Create torus geometry.

        Returns:
            True if successful
        """
        try:
            # Determine torus configuration
            if self.config.torus_config == "120deg":
                config_type = TorusConfiguration.CONFIG_120_DEG
            elif self.config.torus_config == "clover":
                config_type = TorusConfiguration.CONFIG_CLOVER
            elif self.config.torus_config == "cartesian":
                config_type = TorusConfiguration.CONFIG_CARTESIAN
            else:
                raise ValueError(
                    f"Unknown torus configuration: " f"{self.config.torus_config}"
                )

            # Create geometry
            self.field_direction = self.torus_geometries.create_field_direction(
                config_type=config_type,
                radius=self.config.R_torus,
                thickness=self.config.r_torus,
            )
            
            # Create torus geometry object for proper topology
            self.torus_geometry = self.torus_geometries.create_configuration(
                config_type=config_type,
                radius=self.config.R_torus,
                thickness=self.config.r_torus,
                strength=1.0,
            )

            self.status = ModelStatus.GEOMETRY_CREATED
            print(f"Torus geometry created: {self.config.torus_config}")
            return True

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Geometry creation error: {str(e)}"
            return False

    def build_fields(self) -> bool:
        """
        Build SU(2) fields.

        Returns:
            True if successful
        """
        try:
            if self.status != ModelStatus.GEOMETRY_CREATED:
                raise ValueError("Geometry must be created first")

            # Create profile for field building
            from phaze_particles.utils.su2_fields import RadialProfile

            self.profile = RadialProfile(
                self.config.profile_type,
                self.config.r_scale,
                self.config.f_0,
                self.backend,
            )

            # Build SU(2) field using torus configuration
            if hasattr(self, 'torus_geometry') and self.torus_geometry:
                # Use torus configuration for proper topology
                self.su2_field = self.su2_field_builder.build_from_torus_config(
                    torus_config=self.torus_geometry,
                    profile=self.profile,
                )
            else:
                # Fallback to simple field direction (should not happen)
                self.su2_field = self.su2_field_builder.build_field(
                    field_direction=self.field_direction,
                    profile=self.profile,
                )

            # Optional: auto-tune r_scale to reach B≈1 using phase-based baryon density
            if getattr(self.config, "auto_tune_baryon", True):
                try:
                    from phaze_particles.utils.physics import BaryonNumberCalculator

                    tuner = BaryonNumberCalculator(
                        grid_size=self.config.grid_size,
                        box_size=self.config.box_size,
                        backend=self.backend,
                    )

                    def eval_B(r_scale_val: float) -> float:
                        # Rebuild profile and field using torus config to preserve topology
                        cand_profile = type(self.profile)(
                            self.config.profile_type,
                            r_scale_val,
                            self.config.f_0,
                            self.backend,
                        )
                        if hasattr(self, 'torus_geometry') and self.torus_geometry:
                            cand_field = self.su2_field_builder.build_from_torus_config(
                                torus_config=self.torus_geometry,
                                profile=cand_profile,
                            )
                        else:
                            cand_field = self.su2_field_builder.build_field(
                                field_direction=self.field_direction,
                                profile=cand_profile,
                            )
                        return float(tuner.compute_baryon_number_phase(cand_field))

                    # Iterative refinement around current r_scale
                    base = float(self.config.r_scale)
                    best_r, best_err = base, abs(eval_B(base) - 1.0)
                    for _ in range(3):
                        span = 0.3 * best_r
                        grid = 9
                        for k in range(grid):
                            r = max(1e-6, best_r - span/2 + span * k/(grid-1))
                            B = eval_B(r)
                            err = abs(B - 1.0)
                            if err < best_err:
                                best_r, best_err = r, err
                        # shrink span for next iteration
                        # implicit by recomputing span from updated best_r

                    # Apply best r_scale and rebuild final field
                    self.config.r_scale = float(best_r)
                    self.profile = type(self.profile)(
                        self.config.profile_type,
                        self.config.r_scale,
                        self.config.f_0,
                        self.backend,
                    )
                    if hasattr(self, 'torus_geometry') and self.torus_geometry:
                        self.su2_field = self.su2_field_builder.build_from_torus_config(
                            torus_config=self.torus_geometry,
                            profile=self.profile,
                        )
                    else:
                        self.su2_field = self.su2_field_builder.build_field(
                            field_direction=self.field_direction,
                            profile=self.profile,
                        )
                except Exception:
                    pass

            self.status = ModelStatus.FIELDS_BUILT
            print("SU(2) fields built")
            return True

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Field building error: {str(e)}"
            print(f"Field building error details: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def calculate_energy(self) -> bool:
        """
        Calculate energy density with phase environment.

        Returns:
            True if successful
        """
        try:
            if self.status != ModelStatus.FIELDS_BUILT:
                raise ValueError("Fields must be built first")

            # Calculate energy density and get field derivatives
            self.energy_density = self.energy_calculator.calculate_energy_density(
                su2_field=self.su2_field
            )
            self.field_derivatives = self.energy_calculator.calculate_field_derivatives(
                su2_field=self.su2_field
            )
            
            # Calculate phase environment energy contribution
            if self.phase_environment:
                self._calculate_phase_environment_energy()

            # Analyze phase tails and interference
            self._analyze_phase_tails_integration()

            self.status = ModelStatus.ENERGY_CALCULATED
            print("Energy density calculated with phase environment and phase tails")
            return True

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Energy calculation error: {str(e)}"
            print(f"Energy calculation error details: {str(e)}")
            import traceback

            traceback.print_exc()
            return False
    
    def _calculate_phase_environment_energy(self) -> None:
        """
        Calculate phase environment energy contribution.
        
        This integrates the phase field energy into the total energy calculation
        according to 7D theory.
        """
        try:
            # Get coordinates from SU2FieldBuilder
            coordinates = {
                'X': self.su2_field_builder.X,
                'Y': self.su2_field_builder.Y,
                'Z': self.su2_field_builder.Z,
                'R': self.su2_field_builder.R
            }
            
            # Defect position (center of the field)
            defect_position = np.array([0.0, 0.0, 0.0])
            
            # Compute phase well
            phase_well = self.phase_environment.compute_phase_well(
                defect_position=defect_position,
                field_strength=1.0,  # Normalized
                coordinates=coordinates
            )
            
            # Compute phase field energy
            phase_energy = self.phase_environment.compute_phase_field_energy(
                phase_well, 
                self.field_derivatives
            )
            
            # Store phase environment results
            self.phase_environment_results = {
                'phase_well': phase_well,
                'phase_energy': phase_energy,
                'compression_rarefaction_balance': self.phase_environment.compute_compression_rarefaction_balance(phase_well)
            }
            
            # Add phase energy to total energy (if needed)
            # For now, we store it separately to analyze its contribution
            if hasattr(self.energy_density, 'total_energy'):
                # If energy_density has a total_energy attribute, we could add phase energy
                # self.energy_density.total_energy += phase_energy['total_energy']
                pass
                
        except Exception as e:
            print(f"Warning: Phase environment energy calculation failed: {str(e)}")
            # Don't fail the entire calculation if phase environment fails
            self.phase_environment_results = None
    
    def _analyze_phase_tails_integration(self):
        """Analyze phase tails and integrate results into model"""
        try:
            # Run phase tail analysis
            self.phase_tail_results = self.analyze_phase_tails()
            
            # Apply phase tail corrections to energy density
            self._apply_phase_tail_corrections()
            
            # Update physical quantities with phase tail effects
            self._update_physics_with_phase_tails()
            
        except Exception as e:
            print(f"Warning: Phase tail analysis error: {e}")
            self.phase_tail_results = None
    
    def _apply_phase_tail_corrections(self):
        """Apply phase tail corrections to energy density"""
        if not self.phase_tail_results:
            return
        
        # Get phase tail energy contribution
        tail_energy = self.phase_tail_results.tail_energy
        tail_contribution = self.phase_tail_results.tail_contribution
        
        # Apply corrections to energy density components
        if hasattr(self.energy_density, 'c2_term'):
            # Enhance c2 term with phase tail contribution
            phase_correction = tail_energy * 0.1  # 10% of tail energy
            self.energy_density.c2_term += phase_correction
        
        if hasattr(self.energy_density, 'c4_term'):
            # Enhance c4 term with interference effects
            interference_correction = self.phase_tail_results.interference_strength * 0.05
            self.energy_density.c4_term += interference_correction
        
        # Enforce compression-rarefaction balance and continuity by radial re-weighting
        try:
            xp = self.backend.get_array_module()
            R = self.su2_field_builder.R
            # Determine core radius and shell thickness
            R_core = float(getattr(self.phase_tail_results, 'effective_radius', 0.6 * float(self.config.box_size) / 2.0))
            Delta = float(getattr(self.phase_tail_results, 'coherence_length', 0.25 * R_core))
            if Delta <= 0:
                Delta = 0.25 * max(R_core, 1e-6)

            # Smooth radial windows
            # w_core ~1 inside core, decays across shell; w_tail complements
            w_core = 0.5 * (1.0 - xp.tanh((R - R_core) / (Delta + 1e-12)))
            w_tail = 1.0 - w_core

            # Compute imbalance proxy using tail energy fraction
            # Positive if tail overweighted → shift weight from tail to core
            imbalance = float(tail_contribution) - 0.5
            imbalance = max(-0.5, min(0.5, imbalance))

            # Scaling factors (small, conservative)
            core_scale = 1.0 + 0.15 * imbalance
            tail_scale = 1.0 - 0.15 * imbalance

            # Apply re-weighting: push low-order (c2) towards core, reduce high-order (c4) in tail
            if hasattr(self.energy_density, 'c2_term'):
                self.energy_density.c2_term = (
                    self.energy_density.c2_term * (1.0 + (core_scale - 1.0) * w_core)
                )
            if hasattr(self.energy_density, 'c4_term'):
                self.energy_density.c4_term = (
                    self.energy_density.c4_term * (1.0 + (tail_scale - 1.0) * w_tail)
                )
        except Exception as _e:
            # Non-fatal; keep base corrections
            pass

        # Update total density
        if hasattr(self.energy_density, 'total_density'):
            self.energy_density.total_density = (
                self.energy_density.c2_term + 
                self.energy_density.c4_term + 
                self.energy_density.c6_term
            )
    
    def _update_physics_with_phase_tails(self):
        """Update physical quantities with phase tail effects"""
        if not self.phase_tail_results:
            return
        
        # Apply geometric corrections from phase tails
        metric_correction = self.phase_tail_results.effective_metric_correction
        source_enhancement = self.phase_tail_results.newtonian_source_enhancement
        
        # Store phase tail corrections for use in physics calculations
        self.phase_tail_corrections = {
            'metric_correction': metric_correction,
            'source_enhancement': source_enhancement,
            'coherence_length': self.phase_tail_results.coherence_length,
            'interference_strength': self.phase_tail_results.interference_strength
        }

        # Propagate tail metrics to magnetic calculator for physically-based weighting
        try:
            mm = self.physics_calculator.magnetic_calculator
            setattr(mm, 'phase_coherence_length', float(self.phase_tail_results.coherence_length))
            setattr(mm, 'phase_interference_strength', float(self.phase_tail_results.interference_strength))
            # Circumferential modes radial profiles (if available)
            circ = getattr(self.phase_tail_results, 'circumferential_modes', None)
            if circ:
                try:
                    r_centers = [float(c.get('r_center', 0.0)) for c in circ]
                    m_fft = [int(c.get('m_fft', c.get('m', 0))) for c in circ]
                    S_vals = [float(c.get('S', 0.0)) for c in circ]
                    setattr(mm, 'circ_r_centers', r_centers)
                    setattr(mm, 'circ_m_fft', m_fft)
                    setattr(mm, 'circ_S_vals', S_vals)
                except Exception:
                    pass
        except Exception:
            pass

    def calculate_physics(self) -> bool:
        """
        Calculate physical quantities with phase environment.

        Returns:
            True if successful
        """
        try:
            if self.status != ModelStatus.ENERGY_CALCULATED:
                raise ValueError("Energy must be calculated first")

            # Calculate physical quantities using full Skyrme physics
            self.physical_quantities = self.physics_calculator.calculate_quantities(
                su2_field=self.su2_field,
                energy_density=self.energy_density,
                profile=self.profile,
                field_derivatives=self.field_derivatives,
                charge_density_calculator=self.charge_density,  # Pass new charge density calculator
            )
            
            # Update physics with phase environment corrections
            if self.phase_environment and hasattr(self, 'phase_environment_results') and self.phase_environment_results:
                self._apply_phase_environment_corrections()
            
            # Apply phase tail corrections to physical quantities
            if hasattr(self, 'phase_tail_corrections') and self.phase_tail_corrections:
                self._apply_phase_tail_physics_corrections()

            self.status = ModelStatus.PHYSICS_CALCULATED
            print("Physical quantities calculated with phase environment and phase tails")
            return True

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Physics calculation error: {str(e)}"
            print(f"Physics calculation error details: {str(e)}")
            import traceback

            traceback.print_exc()
            return False
    
    def _apply_phase_environment_corrections(self) -> None:
        """
        Apply phase environment corrections to physical quantities.
        
        This method applies corrections based on the phase environment analysis
        according to 7D theory principles.
        """
        try:
            if not self.phase_environment_results:
                return
                
            phase_energy = self.phase_environment_results['phase_energy']
            balance = self.phase_environment_results['compression_rarefaction_balance']
            
            # Apply corrections based on phase environment
            # 1. Mass correction from phase field energy
            if 'total_phase_energy' in phase_energy:
                phase_energy_contribution = phase_energy['total_phase_energy']
                # Convert to MeV (assuming natural units)
                phase_energy_mev = phase_energy_contribution * 197.3  # Conversion factor
                
                # Add phase energy to mass (small correction)
                if hasattr(self.physical_quantities, 'mass'):
                    self.physical_quantities.mass += phase_energy_mev * 0.01  # 1% contribution
            
            # 2. Radius correction from scale quantization
            if self.phase_environment.scale_quantization:
                allowed_radii = self.phase_environment.scale_quantization.allowed_radii
                if allowed_radii:
                    # Use the first allowed radius as a correction
                    corrected_radius = allowed_radii[0]
                    if hasattr(self.physical_quantities, 'charge_radius'):
                        # Blend with original radius (weighted average)
                        original_radius = self.physical_quantities.charge_radius
                        self.physical_quantities.charge_radius = 0.8 * original_radius + 0.2 * corrected_radius
            
            # 3. Stability check from compression-rarefaction balance
            if balance['is_stable']:
                # Stable system - no additional corrections needed
                pass
            else:
                # Unstable system - apply stability corrections
                stability_factor = balance['stability_margin']
                if hasattr(self.physical_quantities, 'mass'):
                    # Reduce mass slightly for unstable systems
                    self.physical_quantities.mass *= stability_factor
                    
        except Exception as e:
            print(f"Warning: Phase environment corrections failed: {str(e)}")
            # Don't fail the entire calculation if corrections fail
    
    def _apply_phase_tail_physics_corrections(self) -> None:
        """
        Apply phase tail corrections to physical quantities.
        
        This method applies corrections based on the phase tail analysis
        according to 7D theory principles.
        """
        try:
            if not hasattr(self, 'phase_tail_corrections') or not self.phase_tail_corrections:
                return
            
            corrections = self.phase_tail_corrections
            
            # Apply metric corrections to charge radius
            if hasattr(self.physical_quantities, 'charge_radius'):
                metric_correction = corrections['metric_correction']
                # Scale charge radius by metric correction
                self.physical_quantities.charge_radius *= (1.0 + metric_correction)
            
            # Apply source enhancement to magnetic moment
            if hasattr(self.physical_quantities, 'magnetic_moment'):
                source_enhancement = corrections['source_enhancement']
                # Scale magnetic moment by source enhancement
                self.physical_quantities.magnetic_moment *= source_enhancement
            
            # Apply coherence effects to mass
            if hasattr(self.physical_quantities, 'mass'):
                coherence_length = corrections['coherence_length']
                interference_strength = corrections['interference_strength']
                # Mass correction based on coherence and interference
                coherence_factor = 1.0 + 0.1 * coherence_length
                interference_factor = 1.0 + 0.05 * interference_strength
                self.physical_quantities.mass *= coherence_factor * interference_factor
            
            # Apply baryon number corrections (if needed)
            if hasattr(self.physical_quantities, 'baryon_number'):
                # Phase tails can affect topological charge
                interference_correction = 1.0 + 0.01 * corrections['interference_strength']
                self.physical_quantities.baryon_number *= interference_correction
            
            print(f"Applied phase tail corrections:")
            print(f"  Metric correction: {corrections['metric_correction']:.6f}")
            print(f"  Source enhancement: {corrections['source_enhancement']:.3f}")
            print(f"  Coherence length: {corrections['coherence_length']:.3f}")
            print(f"  Interference strength: {corrections['interference_strength']:.6f}")
            
        except Exception as e:
            print(f"Warning: Phase tail physics corrections failed: {str(e)}")
            # Don't fail the entire calculation if corrections fail

    def optimize(self) -> bool:
        """
        Optimize model.

        Returns:
            True if successful
        """
        try:
            if self.status != ModelStatus.PHYSICS_CALCULATED:
                raise ValueError("Physics must be calculated first")

            # Functions for optimization
            def energy_function(U: Any) -> float:
                return self.energy_calculator.calculate_total_energy(U)

            def gradient_function(U: Any) -> np.ndarray:
                return self.energy_calculator.calculate_gradient(U)

            constraint_functions = {
                "baryon_number": lambda U: (
                    self.physics_calculator.calculate_baryon_number(U)
                ),
                "electric_charge": lambda U: (
                    self.physics_calculator.calculate_electric_charge(U)
                ),
                "energy_balance": lambda U: (
                    self.energy_calculator.calculate_energy_balance(U)
                ),
            }

            # Relaxation
            optimization_results = self.relaxation_solver.solve(
                U_init=self.su2_field,
                energy_function=energy_function,
                gradient_function=gradient_function,
                constraint_functions=constraint_functions,
            )

            # Update field
            self.su2_field = optimization_results["solution"]

            # Recalculate field derivatives and energy density
            self.field_derivatives = self.energy_calculator.calculate_field_derivatives(
                su2_field=self.su2_field
            )
            self.energy_density = self.energy_calculator.calculate_energy_density(
                su2_field=self.su2_field
            )

            # Recalculate physical quantities
            self.physical_quantities = self.physics_calculator.calculate_quantities(
                su2_field=self.su2_field,
                energy_density=self.energy_density,
                profile=self.profile,
                field_derivatives=self.field_derivatives,
            )

            self.status = ModelStatus.OPTIMIZED
            self.optimization_results = optimization_results
            print(
                f"Model optimized in {optimization_results['iterations']} "
                f"iterations"
            )
            return True

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Optimization error: {str(e)}"
            print(f"Optimization error details: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def get_energy_report(self) -> str:
        """
        Get detailed energy analysis report.

        Returns:
            Energy analysis report string
        """
        if not hasattr(self, 'energy_density') or self.energy_density is None:
            return "Energy density not calculated yet."
        
        from phaze_particles.utils.energy_densities import EnergyAnalyzer
        analyzer = EnergyAnalyzer()
        analysis = analyzer.analyze_energy(self.energy_density)
        
        # Get energy report from EnergyDensities class
        from phaze_particles.utils.energy_densities import EnergyDensities
        energy_densities = EnergyDensities()
        energy_report = energy_densities.get_energy_report(self.energy_density)
        
        # Add additional analysis
        additional_info = f"""
ADDITIONAL ANALYSIS
===================

Virial Residual: {analysis.get('virial_residual', 0.0):.6f}
Positivity Check: {'✓ PASS' if analysis.get('positivity', {}).get('total_energy_positive', True) else '✗ FAIL'}

Energy Components:
  E₂: {analysis['components']['E2']:.6f}
  E₄: {analysis['components']['E4']:.6f}
  E₆: {analysis['components']['E6']:.6f}
  Total: {analysis['components']['E_total']:.6f}

Quality Assessment:
  Overall: {analysis['quality']['overall_quality'].upper()}
  Balance: {analysis['quality']['balance_quality'].upper()}
  Virial: {analysis['quality'].get('virial_quality', 'unknown').upper()}
"""
        
        return energy_report + additional_info

    def optimize_skyrme_constants(
        self,
        target_e2_ratio: float = 0.5,
        target_e4_ratio: float = 0.5,
        target_virial_residual: float = 0.05,
        max_iterations: int = 100,
        verbose: bool = False
    ) -> bool:
        """
        Optimize Skyrme constants for virial balance and energy balance.
        
        Args:
            target_e2_ratio: Target E₂/E_total ratio
            target_e4_ratio: Target E₄/E_total ratio
            target_virial_residual: Target virial residual
            max_iterations: Maximum optimization iterations
            verbose: Verbose output
            
        Returns:
            True if optimization successful
        """
        try:
            if self.status != ModelStatus.ENERGY_CALCULATED:
                raise ValueError("Energy must be calculated first")
            
            if verbose:
                print("Starting Skyrme constants optimization...")
            
            # Create optimization targets
            targets = OptimizationTargets(
                target_e2_ratio=target_e2_ratio,
                target_e4_ratio=target_e4_ratio,
                target_virial_residual=target_virial_residual
            )
            
            # Create optimizer
            optimizer = SkyrmeConstantsOptimizer(
                targets=targets,
                max_iterations=max_iterations,
                learning_rate=0.1,
                convergence_tolerance=1e-4
            )
            
            # Create adaptive optimizer
            adaptive_optimizer = AdaptiveOptimizer(optimizer)
            
            # Optimize constants
            optimization_result = adaptive_optimizer.optimize_with_adaptation(
                energy_calculator=self.energy_calculator,
                su2_field=self.su2_field,
                initial_c2=self.config.c2,
                initial_c4=self.config.c4,
                initial_c6=self.config.c6,
                verbose=verbose
            )
            
            # Update configuration with optimized constants
            self.config.c2 = optimization_result.c2
            self.config.c4 = optimization_result.c4
            self.config.c6 = optimization_result.c6
            
            # Update energy calculator
            self.energy_calculator.c2 = optimization_result.c2
            self.energy_calculator.c4 = optimization_result.c4
            self.energy_calculator.c6 = optimization_result.c6
            
            # Recalculate energy density with optimized constants
            self.energy_density = self.energy_calculator.calculate_energy_density(
                su2_field=self.su2_field
            )
            self.field_derivatives = self.energy_calculator.calculate_field_derivatives(
                su2_field=self.su2_field
            )
            
            # Store optimization result
            self.optimization_result = optimization_result
            
            if verbose:
                print("\n" + "="*60)
                print("SKYRME CONSTANTS OPTIMIZATION COMPLETED")
                print("="*60)
                print(optimizer.get_optimization_report(optimization_result))
            
            return True
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Constants optimization error: {str(e)}"
            print(f"Constants optimization error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def get_optimization_report(self) -> str:
        """
        Get optimization report.
        
        Returns:
            Optimization report string
        """
        if not hasattr(self, 'optimization_result') or self.optimization_result is None:
            return "No optimization performed yet."
        
        from phaze_particles.utils.skyrme_optimizer import SkyrmeConstantsOptimizer
        optimizer = SkyrmeConstantsOptimizer()
        return optimizer.get_optimization_report(self.optimization_result)
    
    def get_phase_environment_report(self) -> str:
        """
        Get phase environment report.
        
        Returns:
            Phase environment analysis report
        """
        if not hasattr(self, 'phase_environment_results') or not self.phase_environment_results:
            return "Phase environment not analyzed. Run calculate_energy() first."
        
        report = []
        report.append("PHASE ENVIRONMENT INTEGRATION REPORT")
        report.append("=" * 50)
        
        # Phase energy contribution
        phase_energy = self.phase_environment_results['phase_energy']
        report.append(f"\nPhase Energy Contribution:")
        report.append(f"  Total phase energy: {phase_energy['total_phase_energy']:.3f}")
        report.append(f"  Interaction energy: {phase_energy['interaction_energy']:.3f}")
        report.append(f"  Total energy: {phase_energy['total_energy']:.3f}")
        
        # Compression-rarefaction balance
        balance = self.phase_environment_results['compression_rarefaction_balance']
        report.append(f"\nCompression-Rarefaction Balance:")
        report.append(f"  Total compression: {balance['total_compression']:.3f}")
        report.append(f"  Total rarefaction: {balance['total_rarefaction']:.3f}")
        report.append(f"  Balance ratio: {balance['balance_ratio']:.3f}")
        report.append(f"  Is stable: {balance['is_stable']}")
        report.append(f"  Stability margin: {balance['stability_margin']:.3f}")
        
        # Scale quantization
        if self.phase_environment and self.phase_environment.scale_quantization:
            quantization = self.phase_environment.scale_quantization
            report.append(f"\nScale Quantization:")
            report.append(f"  Natural radius R*: {quantization.natural_radius_R_star:.3f} fm")
            report.append(f"  Spectral radii Rn: {len(quantization.spectral_radii_Rn)} modes")
            report.append(f"  Allowed radii: {len(quantization.allowed_radii)} modes")
            report.append(f"  ΔR: {quantization.delta_R:.3f} fm")
            
            if quantization.allowed_radii:
                report.append(f"  First few allowed radii: {[f'{r:.3f}' for r in quantization.allowed_radii[:5]]}")
        
        # Physical quantities corrections
        if hasattr(self, 'physical_quantities') and self.physical_quantities:
            report.append(f"\nPhysical Quantities (with phase environment):")
            report.append(f"  Mass: {self.physical_quantities.mass:.1f} MeV")
            report.append(f"  Charge radius: {self.physical_quantities.charge_radius:.3f} fm")
            report.append(f"  Magnetic moment: {self.physical_quantities.magnetic_moment:.3f} μN")
            report.append(f"  Electric charge: {self.physical_quantities.electric_charge:.3f}")
            report.append(f"  Baryon number: {self.physical_quantities.baryon_number:.3f}")
        
        return "\n".join(report)
    
    def analyze_phase_tails(self) -> PhaseTailResult:
        """
        Analyze phase tails and interference patterns
        
        Returns:
            PhaseTailResult: Analysis results
        """
        if not hasattr(self, 'su2_field') or self.su2_field is None:
            raise RuntimeError("SU(2) field not built. Call build_fields() first.")
        
        if not hasattr(self, 'energy_density') or self.energy_density is None:
            raise RuntimeError("Energy density not calculated. Call calculate_energy() first.")
        
        # Prepare field components
        field_components = {
            'u_00': self.su2_field.u_00,
            'u_01': self.su2_field.u_01,
            'u_10': self.su2_field.u_10,
            'u_11': self.su2_field.u_11
        }
        
        # Prepare coordinates
        coordinates = {
            'X': self.su2_field_builder.X,
            'Y': self.su2_field_builder.Y,
            'Z': self.su2_field_builder.Z,
            'R': self.su2_field_builder.R
        }
        
        # Get energy density
        energy_density = self.energy_density.total_density
        
        # Prepare config
        config = {
            'grid_size': self.config.grid_size,
            'box_size': self.config.box_size,
            'c2': self.config.c2,
            'c4': self.config.c4,
            'c6': self.config.c6,
            'phase_density_factor': getattr(self.config, 'phase_density_factor', 1.0),
            'phase_velocity_factor': getattr(self.config, 'phase_velocity_factor', 1.0)
        }
        
        # Run analysis
        return self.phase_tail_analyzer.analyze_phase_tails(
            field_components, coordinates, energy_density, config
        )
    
    def get_phase_tail_report(self) -> str:
        """Get human-readable phase tail analysis report"""
        try:
            result = self.analyze_phase_tails()
            return self.phase_tail_analyzer.get_analysis_report(result)
        except Exception as e:
            return f"Phase tail analysis failed: {str(e)}"

    def solve_with_universal_solver(
        self,
        target_mass: Optional[float] = None,
        target_radius: Optional[float] = None,
        target_magnetic_moment: Optional[float] = None,
        target_bands: Optional[int] = None,
        optimization_strategy: str = 'auto',
        verbose: bool = False
    ) -> SolverOutput:
        """
        Solve proton model using universal solver with advanced optimization.
        
        Args:
            target_mass: Target mass in MeV (e.g., 938.272 for proton)
            target_radius: Target radius in fm (e.g., 0.841 for proton)
            target_magnetic_moment: Target magnetic moment in μN (e.g., 2.793 for proton)
            target_bands: Target number of energy bands
            optimization_strategy: Optimization strategy ('auto', 'energy_balance', 'physical_params', 'quantization')
            verbose: Verbose output
            
        Returns:
            Solver output with comprehensive results
        """
        try:
            if verbose:
                print("Starting universal solver optimization...")
            
            # Create solver input
            solver_input = SolverInput(
                grid_size=self.config.grid_size,
                box_size=self.config.box_size,
                config_type=self.config.config_type,
                c2=self.config.c2,
                c4=self.config.c4,
                c6=self.config.c6,
                F_pi=self.config.F_pi,
                e=self.config.e,
                target_mass=target_mass,
                target_radius=target_radius,
                target_magnetic_moment=target_magnetic_moment,
                target_bands=target_bands,
                optimization_strategy=optimization_strategy
            )
            
            # Create and run universal solver
            solver = UniversalSkyrmeSolver(backend=self.backend)
            
            # Initialize phase environment
            if not hasattr(self, 'phase_environment') or self.phase_environment is None:
                self.phase_environment = PhaseEnvironment(backend=self.backend)
            result = solver.solve(solver_input)
            
            if result.success:
                # Update model with optimized results
                self.config.c2 = result.optimized_constants['c2']
                self.config.c4 = result.optimized_constants['c4']
                self.config.c6 = result.optimized_constants['c6']
                self.config.F_pi = result.optimized_constants['F_pi']
                self.config.e = result.optimized_constants['e']
                
                # Rebuild model with optimized constants
                self._initialize_components()
                self.create_geometry()
                self.build_fields()
                self.calculate_energy()
                self.calculate_physics()
                
                # Store solver result
                self.solver_result = result
                
                if verbose:
                    print("✅ Universal solver optimization completed successfully!")
                    print(f"   Mass: {result.physical_parameters['mass']:.1f} MeV")
                    print(f"   Radius: {result.physical_parameters['charge_radius']:.3f} fm")
                    print(f"   Magnetic moment: {result.physical_parameters['magnetic_moment']:.3f} μN")
                    print(f"   Energy bands: {result.mode_analysis['energy_bands']}")
                    print(f"   Execution time: {result.execution_time:.1f} seconds")
                
                self.status = ModelStatus.PHYSICS_CALCULATED
            else:
                if verbose:
                    print(f"❌ Universal solver failed: {result.convergence_info}")
                
                self.status = ModelStatus.FAILED
                self.error_message = f"Universal solver failed: {result.convergence_info}"
            
            return result
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Universal solver error: {str(e)}"
            if verbose:
                print(f"Universal solver error details: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Return failed result
            return SolverOutput(
                success=False,
                optimized_constants={},
                physical_parameters={},
                energy_analysis={},
                mode_analysis={},
                topological_analysis={},
                interference_analysis={},
                convergence_info={'error': str(e)},
                execution_time=0.0,
                iterations=0
            )
    
    def analyze_phase_environment(self) -> Dict[str, Any]:
        """
        Analyze the phase environment according to 7D theory.
        
        This includes:
        - Phase well around the defect
        - Compression-rarefaction balance
        - Scale quantization
        - Impedance operator
        """
        try:
            if self.phase_environment is None:
                self.phase_environment = PhaseEnvironment(backend=self.backend)
            
            # Get coordinates from SU2FieldBuilder
            coordinates = {
                'X': self.su2_field_builder.X,
                'Y': self.su2_field_builder.Y,
                'Z': self.su2_field_builder.Z,
                'R': self.su2_field_builder.R
            }
            
            # Defect position (center of the field)
            defect_position = np.array([0.0, 0.0, 0.0])
            
            # Compute phase well
            phase_well = self.phase_environment.compute_phase_well(
                defect_position=defect_position,
                field_strength=1.0,  # Normalized
                coordinates=coordinates
            )
            
            # Compute compression-rarefaction balance
            balance = self.phase_environment.compute_compression_rarefaction_balance(phase_well)
            
            # Compute impedance operator
            impedance = self.phase_environment.compute_impedance_operator(
                boundary_radius=self.config.box_size / 2,
                phase_velocity=1.0  # Normalized
            )
            
            # Quantize scales using natural radius from virial conditions
            if hasattr(self, 'physical_quantities') and self.physical_quantities:
                natural_radius = self.physical_quantities.charge_radius
            else:
                natural_radius = 0.841  # Default proton radius
            
            scale_quantization = self.phase_environment.quantize_scales(
                natural_radius_R_star=natural_radius,
                phase_velocity=1.0,
                max_modes=20
            )
            
            # Compute phase field energy
            phase_energy = self.phase_environment.compute_phase_field_energy(
                phase_well, 
                self.field_derivatives if hasattr(self, 'field_derivatives') else {}
            )
            
            return {
                'phase_well': phase_well,
                'compression_rarefaction_balance': balance,
                'impedance_parameters': impedance,
                'scale_quantization': scale_quantization,
                'phase_energy': phase_energy,
                'environment_report': self.phase_environment.get_environment_report()
            }
            
        except Exception as e:
            return {
                'error': f"Phase environment analysis failed: {str(e)}",
                'phase_well': {},
                'compression_rarefaction_balance': {},
                'impedance_parameters': {},
                'scale_quantization': {},
                'phase_energy': {},
                'environment_report': f"Error: {str(e)}"
            }

    def validate(self) -> bool:
        """
        Validate model.

        Returns:
            True if successful
        """
        try:
            if self.status != ModelStatus.OPTIMIZED:
                raise ValueError("Model must be optimized first")

            if not self.validation_system:
                print("Validation disabled")
                return True

            # Prepare data for validation
            calculated_data = CalculatedData(
                proton_mass=self.physical_quantities.mass,
                charge_radius=self.physical_quantities.charge_radius,
                magnetic_moment=self.physical_quantities.magnetic_moment,
                electric_charge=self.physical_quantities.electric_charge,
                baryon_number=self.physical_quantities.baryon_number,
                energy_balance=self.physical_quantities.energy_balance,
                total_energy=self.physical_quantities.energy,
                execution_time=self.optimization_results["execution_time"],
            )

            # Validation
            self.validation_results = self.validation_system.validate_model(
                calculated_data
            )

            # Save reports
            if self.config.save_reports:
                self.validation_system.save_reports(
                    self.validation_results, self.config.output_dir
                )

            self.status = ModelStatus.VALIDATED
            print(
                f"Model validated. Status: "
                f"{self.validation_results['overall_status'].value}"
            )
            return True

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Validation error: {str(e)}"
            print(f"Validation error details: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def run(self) -> ModelResults:
        """
        Run full model cycle.

        Returns:
            Model results
        """
        start_time = time.time()

        try:
            # Create progress bar for the full cycle
            progress_bar = create_progress_bar(6, "Proton Model Execution")

            # Create geometry
            progress_bar.update(1)
            if not self.create_geometry():
                raise RuntimeError("Failed to create geometry")

            # Build fields
            progress_bar.update(1)
            if not self.build_fields():
                raise RuntimeError("Failed to build fields")

            # Calculate energy
            progress_bar.update(1)
            if not self.calculate_energy():
                raise RuntimeError("Failed to calculate energy")

            # Calculate physics
            progress_bar.update(1)
            if not self.calculate_physics():
                raise RuntimeError("Failed to calculate physics")

            # Optimize
            progress_bar.update(1)
            if not self.optimize():
                raise RuntimeError("Failed to optimize")

            # Validate
            progress_bar.update(1)
            if not self.validate():
                raise RuntimeError("Failed to validate")

            # Create results
            self.results = ModelResults(
                status=self.status,
                execution_time=time.time() - start_time,
                iterations=self.optimization_results["iterations"],
                converged=self.optimization_results["converged"],
                proton_mass=self.physical_quantities.mass,
                charge_radius=self.physical_quantities.charge_radius,
                magnetic_moment=self.physical_quantities.magnetic_moment,
                electric_charge=self.physical_quantities.electric_charge,
                baryon_number=self.physical_quantities.baryon_number,
                energy_balance=self.physical_quantities.energy_balance,
                total_energy=self.physical_quantities.energy,
                validation_status=(
                    self.validation_results["overall_status"].value
                    if self.validation_system
                    else None
                ),
                validation_score=(
                    self.validation_results["weighted_score"]
                    if self.validation_system
                    else None
                ),
                config=self.config,
            )

            print("Proton model successfully executed")
            return self.results

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = str(e)

            self.results = ModelResults(
                status=self.status,
                execution_time=time.time() - start_time,
                iterations=0,
                converged=False,
                proton_mass=0.0,
                charge_radius=0.0,
                magnetic_moment=0.0,
                electric_charge=0.0,
                baryon_number=0.0,
                energy_balance=0.0,
                total_energy=0.0,
                config=self.config,
                error_message=self.error_message,
            )

            print(f"Model execution error: {self.error_message}")
            return self.results

    def get_status(self) -> ModelStatus:
        """
        Get current model status.

        Returns:
            Current status
        """
        return self.status

    def get_results(self) -> Optional[ModelResults]:
        """
        Get model results.

        Returns:
            Model results or None
        """
        return self.results

    def save_results(self, output_path: str) -> None:
        """
        Save results to file.

        Args:
            output_path: Path to results file
        """
        if self.results:
            self.results.save_to_file(output_path)
        else:
            raise ValueError("No results to save")

    def reset(self) -> None:
        """Reset model to initial state."""
        self.status = ModelStatus.INITIALIZED
        self.results = None
        self.error_message = None

        # Reset components
        if hasattr(self, "relaxation_solver"):
            self.relaxation_solver.reset()

        print("Model reset to initial state")

    def get_cuda_status(self) -> str:
        """
        Get CUDA status string.

        Returns:
            CUDA status string
        """
        return self.cuda_manager.get_status_string()

    def get_cuda_info(self) -> Dict[str, Any]:
        """
        Get detailed CUDA information.

        Returns:
            Dictionary with CUDA information
        """
        return self.cuda_manager.get_detailed_status()


class ProtonModelFactory:
    """Factory for creating proton models."""

    @staticmethod
    def create_from_config(config_path: str) -> ProtonModel:
        """
        Create model from configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Proton model
        """
        config = ModelConfig.from_file(config_path)
        return ProtonModel(config)

    @staticmethod
    def create_default() -> ProtonModel:
        """
        Create model with default configuration.

        Returns:
            Proton model
        """
        config = ModelConfig()
        return ProtonModel(config)

    @staticmethod
    def create_quick_test() -> ProtonModel:
        """
        Create model for quick testing.

        Returns:
            Proton model
        """
        config = ModelConfig(
            grid_size=32, box_size=2.0, max_iterations=100, validation_enabled=False
        )
        return ProtonModel(config)
