#!/usr/bin/env python3
"""
Tests for integrated proton model.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch

import numpy as np

from phaze_particles.models.proton_integrated import (
    ProtonModel,
    ProtonModelFactory,
    ModelConfig,
    ModelResults,
    ModelStatus,
)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = ModelConfig()

        self.assertEqual(config.grid_size, 64)
        self.assertEqual(config.box_size, 4.0)
        self.assertEqual(config.torus_config, "120deg")
        self.assertEqual(config.R_torus, 1.0)
        self.assertEqual(config.r_torus, 0.2)
        self.assertEqual(config.profile_type, "tanh")
        self.assertEqual(config.f_0, np.pi)
        self.assertEqual(config.f_inf, 0.0)
        self.assertEqual(config.r_scale, 1.0)
        self.assertEqual(config.c2, 1.0)
        self.assertEqual(config.c4, 1.0)
        self.assertEqual(config.c6, 1.0)
        self.assertEqual(config.max_iterations, 1000)
        self.assertEqual(config.convergence_tol, 1e-6)
        self.assertEqual(config.step_size, 0.01)
        self.assertEqual(config.relaxation_method, "gradient_descent")
        self.assertEqual(config.lambda_B, 1000.0)
        self.assertEqual(config.lambda_Q, 1000.0)
        self.assertEqual(config.lambda_virial, 1000.0)
        self.assertTrue(config.validation_enabled)
        self.assertTrue(config.save_reports)
        self.assertEqual(config.output_dir, "results")

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ModelConfig()
        errors = config.validate()
        self.assertEqual(len(errors), 0)

        # Invalid grid_size
        config.grid_size = -1
        errors = config.validate()
        self.assertIn("grid_size must be positive", errors)

        # Invalid box_size
        config = ModelConfig()
        config.box_size = 0
        errors = config.validate()
        self.assertIn("box_size must be positive", errors)

        # Invalid torus_config
        config = ModelConfig()
        config.torus_config = "invalid"
        errors = config.validate()
        self.assertIn("torus_config must be one of: 120deg, clover, cartesian", errors)

        # Invalid R_torus
        config = ModelConfig()
        config.R_torus = -1
        errors = config.validate()
        self.assertIn("R_torus must be positive", errors)

        # Invalid r_torus
        config = ModelConfig()
        config.r_torus = 0
        errors = config.validate()
        self.assertIn("r_torus must be positive", errors)

        # Invalid Skyrme constants
        config = ModelConfig()
        config.c2 = 0
        errors = config.validate()
        self.assertIn("Skyrme constants must be positive", errors)

        # Invalid max_iterations
        config = ModelConfig()
        config.max_iterations = 0
        errors = config.validate()
        self.assertIn("max_iterations must be positive", errors)

        # Invalid convergence_tol
        config = ModelConfig()
        config.convergence_tol = 0
        errors = config.validate()
        self.assertIn("convergence_tol must be positive", errors)

    def test_config_file_operations(self):
        """Test configuration file save/load operations."""
        config = ModelConfig()
        config.grid_size = 128
        config.box_size = 6.0
        config.torus_config = "clover"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name

        try:
            # Save config
            config.save_to_file(config_path)

            # Load config
            loaded_config = ModelConfig.from_file(config_path)

            self.assertEqual(loaded_config.grid_size, 128)
            self.assertEqual(loaded_config.box_size, 6.0)
            self.assertEqual(loaded_config.torus_config, "clover")

        finally:
            os.unlink(config_path)


class TestModelResults(unittest.TestCase):
    """Test ModelResults class."""

    def test_results_creation(self):
        """Test model results creation."""
        results = ModelResults(
            status=ModelStatus.VALIDATED,
            execution_time=10.5,
            iterations=500,
            converged=True,
            proton_mass=938.272,
            charge_radius=0.841,
            magnetic_moment=2.793,
            electric_charge=1.0,
            baryon_number=1.0,
            energy_balance=0.5,
            total_energy=938.272,
            validation_status="excellent",
            validation_score=0.95,
        )

        self.assertEqual(results.status, ModelStatus.VALIDATED)
        self.assertEqual(results.execution_time, 10.5)
        self.assertEqual(results.iterations, 500)
        self.assertTrue(results.converged)
        self.assertEqual(results.proton_mass, 938.272)
        self.assertEqual(results.charge_radius, 0.841)
        self.assertEqual(results.magnetic_moment, 2.793)
        self.assertEqual(results.electric_charge, 1.0)
        self.assertEqual(results.baryon_number, 1.0)
        self.assertEqual(results.energy_balance, 0.5)
        self.assertEqual(results.total_energy, 938.272)
        self.assertEqual(results.validation_status, "excellent")
        self.assertEqual(results.validation_score, 0.95)

    def test_results_to_dict(self):
        """Test results to dictionary conversion."""
        results = ModelResults(
            status=ModelStatus.VALIDATED,
            execution_time=10.5,
            iterations=500,
            converged=True,
            proton_mass=938.272,
            charge_radius=0.841,
            magnetic_moment=2.793,
            electric_charge=1.0,
            baryon_number=1.0,
            energy_balance=0.5,
            total_energy=938.272,
        )

        result_dict = results.to_dict()

        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["status"], ModelStatus.VALIDATED)
        self.assertEqual(result_dict["execution_time"], 10.5)
        self.assertEqual(result_dict["iterations"], 500)
        self.assertTrue(result_dict["converged"])
        self.assertEqual(result_dict["proton_mass"], 938.272)

    def test_results_save_to_file(self):
        """Test results save to file."""
        results = ModelResults(
            status=ModelStatus.VALIDATED,
            execution_time=10.5,
            iterations=500,
            converged=True,
            proton_mass=938.272,
            charge_radius=0.841,
            magnetic_moment=2.793,
            electric_charge=1.0,
            baryon_number=1.0,
            energy_balance=0.5,
            total_energy=938.272,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            results_path = f.name

        try:
            # Save results
            results.save_to_file(results_path)

            # Load and verify
            with open(results_path, "r") as f:
                loaded_data = json.load(f)

            self.assertEqual(loaded_data["status"], "validated")
            self.assertEqual(loaded_data["execution_time"], 10.5)
            self.assertEqual(loaded_data["iterations"], 500)
            self.assertTrue(loaded_data["converged"])
            self.assertEqual(loaded_data["proton_mass"], 938.272)

        finally:
            os.unlink(results_path)


class TestProtonModelFactory(unittest.TestCase):
    """Test ProtonModelFactory class."""

    def test_create_default(self):
        """Test creating model with default configuration."""
        model = ProtonModelFactory.create_default()

        self.assertIsInstance(model, ProtonModel)
        self.assertEqual(model.config.grid_size, 64)
        self.assertEqual(model.config.box_size, 4.0)
        self.assertEqual(model.config.torus_config, "120deg")

    def test_create_quick_test(self):
        """Test creating model for quick testing."""
        model = ProtonModelFactory.create_quick_test()

        self.assertIsInstance(model, ProtonModel)
        self.assertEqual(model.config.grid_size, 32)
        self.assertEqual(model.config.box_size, 2.0)
        self.assertEqual(model.config.max_iterations, 100)
        self.assertFalse(model.config.validation_enabled)

    def test_create_from_config_file(self):
        """Test creating model from configuration file."""
        config = ModelConfig()
        config.grid_size = 128
        config.box_size = 6.0
        config.torus_config = "clover"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name

        try:
            # Save config
            config.save_to_file(config_path)

            # Create model from config file
            model = ProtonModelFactory.create_from_config(config_path)

            self.assertIsInstance(model, ProtonModel)
            self.assertEqual(model.config.grid_size, 128)
            self.assertEqual(model.config.box_size, 6.0)
            self.assertEqual(model.config.torus_config, "clover")

        finally:
            os.unlink(config_path)


class TestProtonModel(unittest.TestCase):
    """Test ProtonModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            grid_size=32, box_size=2.0, max_iterations=10, validation_enabled=False
        )

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    def test_model_initialization(
        self, mock_solver, mock_physics, mock_energy, mock_su2, mock_torus, mock_math
    ):
        """Test model initialization."""
        # Mock all components
        mock_math.return_value = Mock()
        mock_torus.return_value = Mock()
        mock_su2.return_value = Mock()
        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        # Create model
        model = ProtonModel(self.config)

        # Verify initialization
        self.assertEqual(model.status, ModelStatus.INITIALIZED)
        self.assertEqual(model.config, self.config)
        self.assertIsNone(model.results)
        self.assertIsNone(model.error_message)

        # Verify components were initialized
        mock_math.assert_called_once_with(grid_size=32, box_size=2.0)
        mock_torus.assert_called_once_with(grid_size=32, box_size=2.0)
        mock_su2.assert_called_once_with(grid_size=32, box_size=2.0)
        mock_energy.assert_called_once_with(
            grid_size=32, box_size=2.0, c2=1.0, c4=1.0, c6=1.0
        )
        mock_physics.assert_called_once_with(grid_size=32, box_size=2.0)
        mock_solver.assert_called_once()

    def test_invalid_config_initialization(self):
        """Test model initialization with invalid configuration."""
        invalid_config = ModelConfig()
        invalid_config.grid_size = -1  # Invalid

        with self.assertRaises(ValueError) as context:
            ProtonModel(invalid_config)

        self.assertIn("Configuration validation failed", str(context.exception))

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    def test_create_geometry(
        self, mock_solver, mock_physics, mock_energy, mock_su2, mock_torus, mock_math
    ):
        """Test geometry creation."""
        # Mock components
        mock_math.return_value = Mock()
        mock_torus.return_value = Mock()
        mock_su2.return_value = Mock()
        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        # Mock torus geometry creation
        mock_torus_instance = mock_torus.return_value
        mock_torus_instance.create_field_direction.return_value = Mock()

        # Create model
        model = ProtonModel(self.config)

        # Test geometry creation
        result = model.create_geometry()

        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.GEOMETRY_CREATED)
        mock_torus_instance.create_field_direction.assert_called_once()

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    def test_build_fields(
        self, mock_solver, mock_physics, mock_energy, mock_su2, mock_torus, mock_math
    ):
        """Test field building."""
        # Mock components
        mock_math.return_value = Mock()
        mock_torus.return_value = Mock()
        mock_su2.return_value = Mock()
        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        # Mock field building
        mock_su2_instance = mock_su2.return_value
        mock_su2_instance.build_field.return_value = Mock()

        # Create model and set up geometry
        model = ProtonModel(self.config)
        model.status = ModelStatus.GEOMETRY_CREATED
        model.field_direction = Mock()

        # Test field building
        result = model.build_fields()

        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.FIELDS_BUILT)
        mock_su2_instance.build_field.assert_called_once()

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    def test_calculate_energy(
        self, mock_solver, mock_physics, mock_energy, mock_su2, mock_torus, mock_math
    ):
        """Test energy calculation."""
        # Mock components
        mock_math.return_value = Mock()
        mock_torus.return_value = Mock()
        mock_su2.return_value = Mock()
        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        # Mock energy calculation
        mock_energy_instance = mock_energy.return_value
        mock_energy_instance.calculate_energy_density.return_value = Mock()

        # Create model and set up fields
        model = ProtonModel(self.config)
        model.status = ModelStatus.FIELDS_BUILT
        model.su2_field = Mock()

        # Test energy calculation
        result = model.calculate_energy()

        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.ENERGY_CALCULATED)
        mock_energy_instance.calculate_energy_density.assert_called_once()

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    def test_calculate_physics(
        self, mock_solver, mock_physics, mock_energy, mock_su2, mock_torus, mock_math
    ):
        """Test physics calculation."""
        # Mock components
        mock_math.return_value = Mock()
        mock_torus.return_value = Mock()
        mock_su2.return_value = Mock()
        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        # Mock physics calculation
        mock_physics_instance = mock_physics.return_value
        mock_physics_instance.calculate_quantities.return_value = Mock()

        # Create model and set up energy
        model = ProtonModel(self.config)
        model.status = ModelStatus.ENERGY_CALCULATED
        model.su2_field = Mock()
        model.energy_density = Mock()

        # Test physics calculation
        result = model.calculate_physics()

        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.PHYSICS_CALCULATED)
        mock_physics_instance.calculate_quantities.assert_called_once()

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    def test_optimize(
        self, mock_solver, mock_physics, mock_energy, mock_su2, mock_torus, mock_math
    ):
        """Test model optimization."""
        # Mock components
        mock_math.return_value = Mock()
        mock_torus.return_value = Mock()
        mock_su2.return_value = Mock()
        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        # Mock optimization
        mock_solver_instance = mock_solver.return_value
        mock_solver_instance.solve.return_value = {
            "solution": Mock(),
            "iterations": 50,
            "converged": True,
            "execution_time": 5.0,
        }

        # Mock physics calculation
        mock_physics_instance = mock_physics.return_value
        mock_physics_instance.calculate_quantities.return_value = Mock()

        # Create model and set up physics
        model = ProtonModel(self.config)
        model.status = ModelStatus.PHYSICS_CALCULATED
        model.su2_field = Mock()
        model.energy_density = Mock()

        # Test optimization
        result = model.optimize()

        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.OPTIMIZED)
        mock_solver_instance.solve.assert_called_once()

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    def test_validate(
        self, mock_solver, mock_physics, mock_energy, mock_su2, mock_torus, mock_math
    ):
        """Test model validation."""
        # Mock components
        mock_math.return_value = Mock()
        mock_torus.return_value = Mock()
        mock_su2.return_value = Mock()
        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        # Mock validation system
        mock_validation = Mock()
        mock_validation.validate_model.return_value = {
            "overall_status": Mock(value="excellent"),
            "weighted_score": 0.95,
        }

        # Create model and set up optimization
        model = ProtonModel(self.config)
        model.status = ModelStatus.OPTIMIZED
        model.validation_system = mock_validation
        model.physical_quantities = Mock()
        model.physical_quantities.mass = 938.272
        model.physical_quantities.charge_radius = 0.841
        model.physical_quantities.magnetic_moment = 2.793
        model.physical_quantities.electric_charge = 1.0
        model.physical_quantities.baryon_number = 1.0
        model.physical_quantities.energy_balance = 0.5
        model.physical_quantities.energy = 938.272
        model.optimization_results = {"execution_time": 5.0}

        # Test validation
        result = model.validate()

        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.VALIDATED)
        mock_validation.validate_model.assert_called_once()

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    def test_full_run(
        self, mock_solver, mock_physics, mock_energy, mock_su2, mock_torus, mock_math
    ):
        """Test full model run."""
        # Mock components
        mock_math.return_value = Mock()
        mock_torus.return_value = Mock()
        mock_su2.return_value = Mock()
        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        # Mock all methods
        mock_torus_instance = mock_torus.return_value
        mock_torus_instance.create_field_direction.return_value = Mock()

        mock_su2_instance = mock_su2.return_value
        mock_su2_instance.build_field.return_value = Mock()

        mock_energy_instance = mock_energy.return_value
        mock_energy_instance.calculate_energy_density.return_value = Mock()

        mock_physics_instance = mock_physics.return_value
        mock_physics_instance.calculate_quantities.return_value = Mock()
        mock_physics_instance.calculate_baryon_number.return_value = 1.0
        mock_physics_instance.calculate_electric_charge.return_value = 1.0

        mock_energy_instance.calculate_total_energy.return_value = 938.272
        mock_energy_instance.calculate_gradient.return_value = Mock()
        mock_energy_instance.calculate_energy_balance.return_value = 0.5

        mock_solver_instance = mock_solver.return_value
        mock_solver_instance.solve.return_value = {
            "solution": Mock(),
            "iterations": 50,
            "converged": True,
            "execution_time": 5.0,
        }

        # Create model
        model = ProtonModel(self.config)

        # Run model
        results = model.run()

        # Verify results
        self.assertIsInstance(results, ModelResults)
        self.assertEqual(results.status, ModelStatus.OPTIMIZED)
        self.assertGreater(results.execution_time, 0)
        self.assertEqual(results.iterations, 50)
        self.assertTrue(results.converged)

    def test_get_status(self):
        """Test getting model status."""
        model = ProtonModel(self.config)
        self.assertEqual(model.get_status(), ModelStatus.INITIALIZED)

    def test_get_results(self):
        """Test getting model results."""
        model = ProtonModel(self.config)
        self.assertIsNone(model.get_results())

    def test_reset(self):
        """Test model reset."""
        model = ProtonModel(self.config)
        model.status = ModelStatus.VALIDATED
        model.results = Mock()
        model.error_message = "test error"

        model.reset()

        self.assertEqual(model.status, ModelStatus.INITIALIZED)
        self.assertIsNone(model.results)
        self.assertIsNone(model.error_message)


if __name__ == "__main__":
    unittest.main()
