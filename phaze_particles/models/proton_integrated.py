#!/usr/bin/env python3
"""
Integrated proton model implementation.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time

# Import all modules
from ..utils.mathematical_foundations import MathematicalFoundations
from ..utils.torus_geometries import TorusGeometryManager, TorusConfiguration
from ..utils.su2_fields import SU2FieldBuilder
from ..utils.energy_densities import EnergyDensityCalculator
from ..utils.physics import PhysicalQuantitiesCalculator
from ..utils.numerical_methods import (
    RelaxationSolver,
    RelaxationConfig,
    ConstraintConfig,
)
from ..utils.validation import ValidationSystem, ExperimentalData, CalculatedData


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
    r_scale: float = 1.0

    # Skyrme constants
    c2: float = 1.0
    c4: float = 1.0
    c6: float = 1.0

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

        return cls(**config_data)

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

        if self.c2 <= 0 or self.c4 <= 0 or self.c6 <= 0:
            errors.append("Skyrme constants must be positive")

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
            self.torus_geometries = TorusGeometryManager(
                grid_size=self.config.grid_size, box_size=self.config.box_size
            )

            # SU(2) fields
            self.su2_field_builder = SU2FieldBuilder(
                grid_size=self.config.grid_size, box_size=self.config.box_size
            )

            # Energy densities
            self.energy_calculator = EnergyDensityCalculator(
                grid_size=self.config.grid_size,
                box_size=self.config.box_size,
                c2=self.config.c2,
                c4=self.config.c4,
                c6=self.config.c6,
            )

            # Physical quantities
            self.physics_calculator = PhysicalQuantitiesCalculator(
                grid_size=self.config.grid_size, box_size=self.config.box_size
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

            print("All model components successfully initialized")

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

            # Build SU(2) field
            self.su2_field = self.su2_field_builder.build_field(
                field_direction=self.field_direction,
                profile_type=self.config.profile_type,
                f_0=self.config.f_0,
                f_inf=self.config.f_inf,
                r_scale=self.config.r_scale,
            )

            self.status = ModelStatus.FIELDS_BUILT
            print("SU(2) fields built")
            return True

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Field building error: {str(e)}"
            return False

    def calculate_energy(self) -> bool:
        """
        Calculate energy density.

        Returns:
            True if successful
        """
        try:
            if self.status != ModelStatus.FIELDS_BUILT:
                raise ValueError("Fields must be built first")

            # Calculate energy density
            self.energy_density = self.energy_calculator.calculate_energy_density(
                su2_field=self.su2_field
            )

            self.status = ModelStatus.ENERGY_CALCULATED
            print("Energy density calculated")
            return True

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Energy calculation error: {str(e)}"
            return False

    def calculate_physics(self) -> bool:
        """
        Calculate physical quantities.

        Returns:
            True if successful
        """
        try:
            if self.status != ModelStatus.ENERGY_CALCULATED:
                raise ValueError("Energy must be calculated first")

            # Calculate physical quantities
            self.physical_quantities = self.physics_calculator.calculate_quantities(
                su2_field=self.su2_field, energy_density=self.energy_density
            )

            self.status = ModelStatus.PHYSICS_CALCULATED
            print("Physical quantities calculated")
            return True

        except Exception as e:
            self.status = ModelStatus.FAILED
            self.error_message = f"Physics calculation error: {str(e)}"
            return False

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

            def constraint_functions(U: Any) -> Dict[str, float]:
                return {
                    "baryon_number": (
                        self.physics_calculator.calculate_baryon_number(U)
                    ),
                    "electric_charge": (
                        self.physics_calculator.calculate_electric_charge(U)
                    ),
                    "energy_balance": (
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

            # Recalculate physical quantities
            self.physical_quantities = self.physics_calculator.calculate_quantities(
                su2_field=self.su2_field, energy_density=self.energy_density
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
            return False

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
            return False

    def run(self) -> ModelResults:
        """
        Run full model cycle.

        Returns:
            Model results
        """
        start_time = time.time()

        try:
            # Create geometry
            if not self.create_geometry():
                raise RuntimeError("Failed to create geometry")

            # Build fields
            if not self.build_fields():
                raise RuntimeError("Failed to build fields")

            # Calculate energy
            if not self.calculate_energy():
                raise RuntimeError("Failed to calculate energy")

            # Calculate physics
            if not self.calculate_physics():
                raise RuntimeError("Failed to calculate physics")

            # Optimize
            if not self.optimize():
                raise RuntimeError("Failed to optimize")

            # Validate
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
