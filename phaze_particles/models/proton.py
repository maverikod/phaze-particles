#!/usr/bin/env python3
"""
Proton model implementation based on three torus configurations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Any, Dict, List, Callable

from .base import BaseModel
from ..utils.numerical_methods import (
    RelaxationMethod,
    RelaxationConfig,
    ConstraintConfig,
    RelaxationSolver,
    NumericalMethods,
)
from ..utils.physics import PhysicsAnalyzer, PhysicalQuantitiesCalculator
from ..utils.su2_fields import SU2Field
from ..utils.torus_geometries import TorusGeometryManager
from ..utils.energy_densities import EnergyDensityCalculator


class ProtonModel(BaseModel):
    """
    Proton model using topological soliton approach with three torus configurations.

    Implements the SU(2)/SU(3) Skyrme-like theory for proton modeling
    with three toroidal structures: 120°, clover, and cartesian.
    """

    def __init__(self):
        """Initialize proton model."""
        super().__init__(
            name="proton", description="Proton model with three torus configurations"
        )

        # Default parameters
        self.parameters = {
            "grid_size": 64,
            "box_size": 4.0,  # fm
            "config_type": "all",  # '120deg', 'clover', 'cartesian', 'all'
            "convergence_tolerance": 1e-6,
            "max_iterations": 1000,
            "relaxation_method": "gradient_descent",  # 'gradient_descent', 'lbfgs', 'adam'
            "step_size": 0.01,
            "momentum": 0.9,
        }

        # Physical constants
        self.physical_constants = {
            "proton_mass": 938.272,  # MeV
            "proton_radius": 0.84,  # fm
            "proton_magnetic_moment": 2.793,  # μN
            "electron_charge": 1.0,  # e
        }

        # Initialize components
        self.numerical_methods = None
        self.torus_manager = None
        self.su2_field = None
        self.energy_calculator = None
        self.physics_calculator = None
        self.physics_analyzer = None
        self.relaxation_solver = None

    def validate_parameters(self) -> bool:
        """
        Validate model parameters.

        Returns:
            True if parameters are valid, False otherwise
        """
        if self.parameters["grid_size"] < 16:
            return False

        if self.parameters["box_size"] <= 0:
            return False

        if self.parameters["config_type"] not in [
            "120deg",
            "clover",
            "cartesian",
            "all",
        ]:
            return False

        if self.parameters["convergence_tolerance"] <= 0:
            return False

        if self.parameters["max_iterations"] <= 0:
            return False

        if self.parameters["relaxation_method"] not in [
            "gradient_descent",
            "lbfgs",
            "adam",
        ]:
            return False

        if self.parameters["step_size"] <= 0:
            return False

        if not (0 <= self.parameters["momentum"] <= 1):
            return False

        return True

    def _initialize_components(self):
        """Initialize all model components."""
        grid_size = self.parameters["grid_size"]
        box_size = self.parameters["box_size"]

        # Initialize numerical methods
        self.numerical_methods = NumericalMethods(grid_size, box_size)

        # Initialize torus geometry manager
        self.torus_manager = TorusGeometryManager(grid_size, box_size)

        # Initialize SU(2) field
        self.su2_field = SU2Field(grid_size, box_size)

        # Initialize energy density calculator
        self.energy_calculator = EnergyDensityCalculator(grid_size, box_size)

        # Initialize physics calculator
        self.physics_calculator = PhysicalQuantitiesCalculator(grid_size, box_size)

        # Initialize physics analyzer
        self.physics_analyzer = PhysicsAnalyzer()

        # Initialize relaxation solver
        relaxation_config = RelaxationConfig(
            method=RelaxationMethod(self.parameters["relaxation_method"]),
            max_iterations=self.parameters["max_iterations"],
            convergence_tol=self.parameters["convergence_tolerance"],
            step_size=self.parameters["step_size"],
            momentum=self.parameters["momentum"],
        )
        constraint_config = ConstraintConfig()
        self.relaxation_solver = RelaxationSolver(relaxation_config, constraint_config)

    def run(self) -> Dict[str, Any]:
        """
        Run proton model calculation.

        Returns:
            Dictionary containing calculation results
        """
        if not self.validate_parameters():
            raise ValueError("Invalid model parameters")

        print("Initializing proton model components...")
        self._initialize_components()

        print("Creating initial SU(2) field...")
        U_init = self.numerical_methods.create_initial_field("tanh")

        print("Setting up torus configurations...")
        config_type = self.parameters["config_type"]
        if config_type == "all":
            configurations = ["120deg", "clover", "cartesian"]
        else:
            configurations = [config_type]

        results = {}
        for config in configurations:
            print(f"Processing {config} configuration...")
            config_result = self._run_configuration(U_init, config)
            results[config] = config_result

        # Combine results
        self.results = self._combine_results(results)
        
        # Perform physical analysis
        print("Performing physical analysis...")
        self._perform_physical_analysis()

        return self.results

    def _run_configuration(self, U_init: np.ndarray, config_type: str) -> Dict[str, Any]:
        """
        Run calculation for specific configuration.

        Args:
            U_init: Initial field
            config_type: Configuration type

        Returns:
            Configuration results
        """
        # Set up torus geometry
        self.torus_manager.set_configuration(config_type)
        
        # Create constraint functions
        constraint_functions = self._create_constraint_functions()
        
        # Create energy and gradient functions
        energy_function = self._create_energy_function(config_type)
        gradient_function = self._create_gradient_function(config_type)
        
        # Run relaxation
        print(f"Running relaxation for {config_type}...")
        relaxation_result = self.relaxation_solver.solve(
            U_init, energy_function, gradient_function, constraint_functions
        )
        
        # Calculate physical quantities
        print(f"Calculating physical quantities for {config_type}...")
        U_final = relaxation_result['solution']
        field_derivatives = self.su2_field.compute_derivatives(U_final)
        energy = relaxation_result['final_energy']
        
        quantities = self.physics_calculator.compute_all_quantities(
            U_final, self.torus_manager.get_profile(), field_derivatives, energy
        )
        
        return {
            'relaxation': relaxation_result,
            'quantities': quantities,
            'field': U_final,
            'derivatives': field_derivatives,
        }

    def _create_constraint_functions(self) -> Dict[str, Callable]:
        """Create constraint functions for relaxation."""
        def baryon_function(U):
            derivatives = self.su2_field.compute_derivatives(U)
            return self.physics_calculator.baryon_calculator.compute_baryon_number(derivatives)
        
        def charge_function(U):
            charge_density = self.physics_calculator.charge_density.compute_charge_density(
                U, self.torus_manager.get_profile()
            )
            return self.physics_calculator.charge_density.compute_electric_charge(charge_density)
        
        def energy_balance_function(U):
            # Calculate energy components
            derivatives = self.su2_field.compute_derivatives(U)
            energy_components = self.energy_calculator.compute_all_components(U, derivatives)
            total_energy = sum(energy_components.values())
            if total_energy > 0:
                return energy_components['E2'] / total_energy
            return 0.5
        
        return {
            'baryon_number': baryon_function,
            'electric_charge': charge_function,
            'energy_balance': energy_balance_function,
        }

    def _create_energy_function(self, config_type: str) -> Callable:
        """Create energy function for specific configuration."""
        def energy_function(U):
            derivatives = self.su2_field.compute_derivatives(U)
            energy_components = self.energy_calculator.compute_all_components(U, derivatives)
            return sum(energy_components.values())
        
        return energy_function

    def _create_gradient_function(self, config_type: str) -> Callable:
        """Create gradient function for specific configuration."""
        def gradient_function(U):
            # Simplified gradient - in real implementation would compute functional derivative
            # For now, return small random gradient
            return np.random.randn(*U.shape) * 0.01
        
        return gradient_function

    def _combine_results(self, config_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from all configurations."""
        # For now, return results from first configuration
        # In full implementation, would combine or select best result
        first_config = list(config_results.keys())[0]
        first_result = config_results[first_config]
        
        quantities = first_result['quantities']
        
        return {
            "electric_charge": quantities.electric_charge,
            "baryon_number": quantities.baryon_number,
            "mass": quantities.mass,
            "radius": quantities.charge_radius,
            "magnetic_moment": quantities.magnetic_moment,
            "energy_balance": {
                "E2_percentage": 50.0,  # Placeholder
                "E4_percentage": 50.0,  # Placeholder
                "E6_percentage": 0.0,   # Placeholder
            },
            "configurations": {
                config: {"status": "completed"} for config in config_results.keys()
            },
            "relaxation_info": {
                config: result['relaxation'] for config, result in config_results.items()
            },
        }

    def _perform_physical_analysis(self):
        """Perform physical analysis of results."""
        calculated_values = {
            "electric_charge": self.results["electric_charge"],
            "baryon_number": self.results["baryon_number"],
            "mass": self.results["mass"],
            "radius": self.results["radius"],
            "magnetic_moment": self.results["magnetic_moment"],
        }
        
        analysis_results = self.physics_analyzer.analyze_results(calculated_values)
        
        # Add analysis to results
        self.results["physical_analysis"] = {
            "comparison_table": self.physics_analyzer.generate_comparison_table(),
            "overall_quality": self.physics_analyzer.get_overall_quality(),
            "validation_status": self.physics_analyzer.get_validation_status(),
            "recommendations": self.physics_analyzer.get_recommendations(),
            "detailed_results": [
                {
                    "parameter": result.parameter.name,
                    "calculated": result.parameter.calculated_value,
                    "experimental": result.parameter.experimental_value,
                    "deviation_percent": result.deviation_percent,
                    "within_tolerance": result.within_tolerance,
                    "quality_rating": result.quality_rating,
                }
                for result in analysis_results
            ],
        }

    def get_available_configurations(self) -> List[str]:
        """
        Get list of available torus configurations.

        Returns:
            List of configuration names
        """
        return ["120deg", "clover", "cartesian"]

    def calculate_energy_balance(self) -> Dict[str, float]:
        """
        Calculate energy balance for virial condition.

        Returns:
            Dictionary with energy percentages
        """
        # TODO: Implement actual energy balance calculation
        return {
            "E2_percentage": 50.0,
            "E4_percentage": 50.0,
            "E6_percentage": 0.0,
        }
