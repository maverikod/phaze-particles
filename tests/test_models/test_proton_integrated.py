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
from unittest.mock import Mock, patch, MagicMock

import numpy as np

from phaze_particles.models.proton_integrated import (
    ProtonModel,
    ModelConfig,
    ModelResults,
    ModelStatus,
    ProtonModelFactory,
)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ModelConfig()

        # Check default values
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

        # Invalid grid size
        config.grid_size = -1
        errors = config.validate()
        self.assertIn("grid_size must be positive", errors)

        # Invalid box size
        config = ModelConfig()
        config.box_size = 0
        errors = config.validate()
        self.assertIn("box_size must be positive", errors)

        # Invalid torus config
        config = ModelConfig()
        config.torus_config = "invalid"
        errors = config.validate()
        self.assertIn("torus_config must be one of: 120deg, clover, cartesian", errors)

        # Invalid Skyrme constants
        config = ModelConfig()
        config.c2 = 0
        errors = config.validate()
        self.assertIn("Skyrme constants must be positive", errors)

    def test_config_file_operations(self):
        """Test configuration file operations."""
        config = ModelConfig(grid_size=32, box_size=2.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.save_to_file(f.name)

            # Load config from file
            loaded_config = ModelConfig.from_file(f.name)

            self.assertEqual(loaded_config.grid_size, 32)
            self.assertEqual(loaded_config.box_size, 2.0)

            # Clean up
            os.unlink(f.name)


class TestModelResults(unittest.TestCase):
    """Test ModelResults class."""

    def test_model_results_creation(self):
        """Test ModelResults creation."""
        results = ModelResults(
            status=ModelStatus.VALIDATED,
            execution_time=10.5,
            iterations=100,
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
        self.assertEqual(results.iterations, 100)
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

    def test_model_results_to_dict(self):
        """Test ModelResults to_dict method."""
        results = ModelResults(
            status=ModelStatus.VALIDATED,
            execution_time=10.5,
            iterations=100,
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
        self.assertEqual(result_dict["iterations"], 100)
        self.assertTrue(result_dict["converged"])
        self.assertEqual(result_dict["proton_mass"], 938.272)

    def test_model_results_save_to_file(self):
        """Test ModelResults save_to_file method."""
        results = ModelResults(
            status=ModelStatus.VALIDATED,
            execution_time=10.5,
            iterations=100,
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
            results.save_to_file(f.name)

            # Check file was created and contains data
            self.assertTrue(os.path.exists(f.name))

            with open(f.name, "r") as file:
                data = json.load(file)

            self.assertEqual(data["status"], "validated")
            self.assertEqual(data["execution_time"], 10.5)
            self.assertEqual(data["iterations"], 100)
            self.assertTrue(data["converged"])
            self.assertEqual(data["proton_mass"], 938.272)

            # Clean up
            os.unlink(f.name)


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
    @patch("phaze_particles.models.proton_integrated.get_cuda_manager")
    def test_proton_model_initialization(
        self,
        mock_cuda_manager,
        mock_solver,
        mock_physics,
        mock_energy,
        mock_su2,
        mock_torus,
        mock_math,
    ):
        """Test ProtonModel initialization."""
        # Mock CUDA manager
        mock_cuda_manager.return_value.get_status_string.return_value = (
            "CUDA: ❌ Not Available (CPU mode)"
        )

        # Mock components
        mock_math.return_value = Mock()
        mock_torus.return_value = Mock()
        mock_su2.return_value = Mock()
        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        model = ProtonModel(self.config)

        self.assertEqual(model.status, ModelStatus.INITIALIZED)
        self.assertEqual(model.config, self.config)
        self.assertIsNotNone(model.cuda_manager)

        # Check that all components were initialized
        mock_math.assert_called_once()
        # Note: TorusGeometryManager is not directly instantiated in ProtonModel
        # It's used internally by TorusGeometries
        mock_su2.assert_called_once()
        mock_energy.assert_called_once()
        mock_physics.assert_called_once()
        mock_solver.assert_called_once()

    def test_proton_model_invalid_config(self):
        """Test ProtonModel with invalid configuration."""
        invalid_config = ModelConfig(grid_size=-1)  # Invalid grid size

        with self.assertRaises(ValueError) as context:
            ProtonModel(invalid_config)

        self.assertIn("Configuration validation failed", str(context.exception))

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    @patch("phaze_particles.models.proton_integrated.get_cuda_manager")
    def test_create_geometry(
        self,
        mock_cuda_manager,
        mock_solver,
        mock_physics,
        mock_energy,
        mock_su2,
        mock_torus,
        mock_math,
    ):
        """Test geometry creation."""
        # Mock CUDA manager
        mock_cuda_manager.return_value.get_status_string.return_value = (
            "CUDA: ❌ Not Available (CPU mode)"
        )

        # Mock torus geometry manager
        mock_torus_instance = Mock()
        mock_torus_instance.create_field_direction.return_value = Mock()
        mock_torus.return_value = mock_torus_instance

        # Mock other components
        mock_math.return_value = Mock()
        mock_su2.return_value = Mock()
        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        model = ProtonModel(self.config)

        # Test successful geometry creation
        result = model.create_geometry()
        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.GEOMETRY_CREATED)

        # Test with invalid torus config
        model.config.torus_config = "invalid"
        result = model.create_geometry()
        self.assertFalse(result)
        self.assertEqual(model.status, ModelStatus.FAILED)

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    @patch("phaze_particles.models.proton_integrated.get_cuda_manager")
    def test_build_fields(
        self,
        mock_cuda_manager,
        mock_solver,
        mock_physics,
        mock_energy,
        mock_su2,
        mock_torus,
        mock_math,
    ):
        """Test field building."""
        # Mock CUDA manager
        mock_cuda_manager.return_value.get_status_string.return_value = (
            "CUDA: ❌ Not Available (CPU mode)"
        )

        # Mock components
        mock_math.return_value = Mock()
        mock_torus_instance = Mock()
        mock_torus_instance.create_field_direction.return_value = Mock()
        mock_torus.return_value = mock_torus_instance

        mock_su2_instance = Mock()
        mock_su2_instance.build_field.return_value = Mock()
        mock_su2.return_value = mock_su2_instance

        mock_energy.return_value = Mock()
        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        model = ProtonModel(self.config)

        # First create geometry
        model.create_geometry()

        # Then build fields
        result = model.build_fields()
        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.FIELDS_BUILT)

        # Test without geometry
        model.status = ModelStatus.INITIALIZED
        result = model.build_fields()
        self.assertFalse(result)
        self.assertEqual(model.status, ModelStatus.FAILED)

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    @patch("phaze_particles.models.proton_integrated.get_cuda_manager")
    def test_calculate_energy(
        self,
        mock_cuda_manager,
        mock_solver,
        mock_physics,
        mock_energy,
        mock_su2,
        mock_torus,
        mock_math,
    ):
        """Test energy calculation."""
        # Mock CUDA manager
        mock_cuda_manager.return_value.get_status_string.return_value = (
            "CUDA: ❌ Not Available (CPU mode)"
        )

        # Mock components
        mock_math.return_value = Mock()
        mock_torus_instance = Mock()
        mock_torus_instance.create_field_direction.return_value = Mock()
        mock_torus.return_value = mock_torus_instance

        mock_su2_instance = Mock()
        mock_su2_instance.build_field.return_value = Mock()
        mock_su2.return_value = mock_su2_instance

        mock_energy_instance = Mock()
        mock_energy_instance.calculate_energy_density.return_value = Mock()
        mock_energy.return_value = mock_energy_instance

        mock_physics.return_value = Mock()
        mock_solver.return_value = Mock()

        model = ProtonModel(self.config)

        # Set up model state
        model.create_geometry()
        model.build_fields()

        # Calculate energy
        result = model.calculate_energy()
        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.ENERGY_CALCULATED)

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    @patch("phaze_particles.models.proton_integrated.get_cuda_manager")
    def test_calculate_physics(
        self,
        mock_cuda_manager,
        mock_solver,
        mock_physics,
        mock_energy,
        mock_su2,
        mock_torus,
        mock_math,
    ):
        """Test physics calculation."""
        # Mock CUDA manager
        mock_cuda_manager.return_value.get_status_string.return_value = (
            "CUDA: ❌ Not Available (CPU mode)"
        )

        # Mock components
        mock_math.return_value = Mock()
        mock_torus_instance = Mock()
        mock_torus_instance.create_field_direction.return_value = Mock()
        mock_torus.return_value = mock_torus_instance

        mock_su2_instance = Mock()
        mock_su2_instance.build_field.return_value = Mock()
        mock_su2.return_value = mock_su2_instance

        mock_energy_instance = Mock()
        mock_energy_instance.calculate_energy_density.return_value = Mock()
        mock_energy.return_value = mock_energy_instance

        mock_physics_instance = Mock()
        mock_physics_instance.calculate_quantities.return_value = Mock()
        mock_physics.return_value = mock_physics_instance

        mock_solver.return_value = Mock()

        model = ProtonModel(self.config)

        # Set up model state
        model.create_geometry()
        model.build_fields()
        model.calculate_energy()

        # Calculate physics
        result = model.calculate_physics()
        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.PHYSICS_CALCULATED)

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    @patch("phaze_particles.models.proton_integrated.get_cuda_manager")
    def test_optimize(
        self,
        mock_cuda_manager,
        mock_solver,
        mock_physics,
        mock_energy,
        mock_su2,
        mock_torus,
        mock_math,
    ):
        """Test model optimization."""
        # Mock CUDA manager
        mock_cuda_manager.return_value.get_status_string.return_value = (
            "CUDA: ❌ Not Available (CPU mode)"
        )

        # Mock components
        mock_math.return_value = Mock()
        mock_torus_instance = Mock()
        mock_torus_instance.create_field_direction.return_value = Mock()
        mock_torus.return_value = mock_torus_instance

        mock_su2_instance = Mock()
        mock_su2_instance.build_field.return_value = Mock()
        mock_su2.return_value = mock_su2_instance

        mock_energy_instance = Mock()
        mock_energy_instance.calculate_energy_density.return_value = Mock()
        mock_energy_instance.calculate_total_energy.return_value = 100.0
        mock_energy_instance.calculate_gradient.return_value = np.zeros(
            (32, 32, 32, 2, 2)
        )
        mock_energy_instance.calculate_energy_balance.return_value = 0.5
        mock_energy.return_value = mock_energy_instance

        mock_physics_instance = Mock()
        mock_physics_instance.calculate_quantities.return_value = Mock()
        mock_physics_instance.calculate_baryon_number.return_value = 1.0
        mock_physics_instance.calculate_electric_charge.return_value = 1.0
        mock_physics.return_value = mock_physics_instance

        mock_solver_instance = Mock()
        mock_solver_instance.solve.return_value = {
            "solution": Mock(),
            "iterations": 50,
            "converged": True,
            "execution_time": 5.0,
        }
        mock_solver.return_value = mock_solver_instance

        model = ProtonModel(self.config)

        # Set up model state
        model.create_geometry()
        model.build_fields()
        model.calculate_energy()
        model.calculate_physics()

        # Optimize
        result = model.optimize()
        self.assertTrue(result)
        self.assertEqual(model.status, ModelStatus.OPTIMIZED)

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    @patch("phaze_particles.models.proton_integrated.get_cuda_manager")
    def test_validate(
        self,
        mock_cuda_manager,
        mock_solver,
        mock_physics,
        mock_energy,
        mock_su2,
        mock_torus,
        mock_math,
    ):
        """Test model validation."""
        # Mock CUDA manager
        mock_cuda_manager.return_value.get_status_string.return_value = (
            "CUDA: ❌ Not Available (CPU mode)"
        )

        # Mock components
        mock_math.return_value = Mock()
        mock_torus_instance = Mock()
        mock_torus_instance.create_field_direction.return_value = Mock()
        mock_torus.return_value = mock_torus_instance

        mock_su2_instance = Mock()
        mock_su2_instance.build_field.return_value = Mock()
        mock_su2.return_value = mock_su2_instance

        mock_energy_instance = Mock()
        mock_energy_instance.calculate_energy_density.return_value = Mock()
        mock_energy_instance.calculate_total_energy.return_value = 100.0
        mock_energy_instance.calculate_gradient.return_value = np.zeros(
            (32, 32, 32, 2, 2)
        )
        mock_energy_instance.calculate_energy_balance.return_value = 0.5
        mock_energy.return_value = mock_energy_instance

        mock_physics_instance = Mock()
        mock_physics_instance.calculate_quantities.return_value = Mock()
        mock_physics_instance.calculate_baryon_number.return_value = 1.0
        mock_physics_instance.calculate_electric_charge.return_value = 1.0
        mock_physics.return_value = mock_physics_instance

        mock_solver_instance = Mock()
        mock_solver_instance.solve.return_value = {
            "solution": Mock(),
            "iterations": 50,
            "converged": True,
            "execution_time": 5.0,
        }
        mock_solver.return_value = mock_solver_instance

        model = ProtonModel(self.config)

        # Set up model state
        model.create_geometry()
        model.build_fields()
        model.calculate_energy()
        model.calculate_physics()
        model.optimize()

        # Validate
        result = model.validate()
        self.assertTrue(result)
        # Since validation is disabled in config, status should remain OPTIMIZED
        self.assertEqual(model.status, ModelStatus.OPTIMIZED)

    @patch("phaze_particles.models.proton_integrated.MathematicalFoundations")
    @patch("phaze_particles.models.proton_integrated.TorusGeometryManager")
    @patch("phaze_particles.models.proton_integrated.SU2FieldBuilder")
    @patch("phaze_particles.models.proton_integrated.EnergyDensityCalculator")
    @patch("phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator")
    @patch("phaze_particles.models.proton_integrated.RelaxationSolver")
    @patch("phaze_particles.models.proton_integrated.get_cuda_manager")
    def test_run_full_cycle(
        self,
        mock_cuda_manager,
        mock_solver,
        mock_physics,
        mock_energy,
        mock_su2,
        mock_torus,
        mock_math,
    ):
        """Test full model cycle."""
        # Mock CUDA manager
        mock_cuda_manager.return_value.get_status_string.return_value = (
            "CUDA: ❌ Not Available (CPU mode)"
        )

        # Mock components
        mock_math.return_value = Mock()
        mock_torus_instance = Mock()
        mock_torus_instance.create_field_direction.return_value = Mock()
        mock_torus.return_value = mock_torus_instance

        mock_su2_instance = Mock()
        mock_su2_instance.build_field.return_value = Mock()
        mock_su2.return_value = mock_su2_instance

        mock_energy_instance = Mock()
        mock_energy_instance.calculate_energy_density.return_value = Mock()
        mock_energy_instance.calculate_total_energy.return_value = 100.0
        mock_energy_instance.calculate_gradient.return_value = np.zeros(
            (32, 32, 32, 2, 2)
        )
        mock_energy_instance.calculate_energy_balance.return_value = 0.5
        mock_energy.return_value = mock_energy_instance

        mock_physics_instance = Mock()
        mock_physics_instance.calculate_quantities.return_value = Mock()
        mock_physics_instance.calculate_baryon_number.return_value = 1.0
        mock_physics_instance.calculate_electric_charge.return_value = 1.0
        mock_physics.return_value = mock_physics_instance

        mock_solver_instance = Mock()
        mock_solver_instance.solve.return_value = {
            "solution": Mock(),
            "iterations": 50,
            "converged": True,
            "execution_time": 5.0,
        }
        mock_solver.return_value = mock_solver_instance

        model = ProtonModel(self.config)

        # Run full cycle
        results = model.run()

        self.assertIsInstance(results, ModelResults)
        # Since validation is disabled in config, status should be OPTIMIZED
        self.assertEqual(results.status, ModelStatus.OPTIMIZED)
        self.assertGreater(results.execution_time, 0)
        self.assertEqual(results.iterations, 50)
        self.assertTrue(results.converged)

    def test_get_cuda_status(self):
        """Test CUDA status retrieval."""
        with patch(
            "phaze_particles.models.proton_integrated.get_cuda_manager"
        ) as mock_cuda_manager:
            mock_cuda_instance = Mock()
            mock_cuda_instance.get_status_string.return_value = "CUDA: ✅ Available"
            mock_cuda_instance.get_detailed_status.return_value = {
                "available": True,
                "device_count": 1,
            }
            mock_cuda_manager.return_value = mock_cuda_instance

            # Mock other components
            with patch(
                "phaze_particles.models.proton_integrated.MathematicalFoundations"
            ), patch(
                "phaze_particles.models.proton_integrated.TorusGeometryManager"
            ), patch(
                "phaze_particles.models.proton_integrated.SU2FieldBuilder"
            ), patch(
                "phaze_particles.models.proton_integrated.EnergyDensityCalculator"
            ), patch(
                "phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator"
            ), patch(
                "phaze_particles.models.proton_integrated.RelaxationSolver"
            ):

                model = ProtonModel(self.config)

                status = model.get_cuda_status()
                self.assertEqual(status, "CUDA: ✅ Available")

                info = model.get_cuda_info()
                self.assertIsInstance(info, dict)

    def test_reset(self):
        """Test model reset."""
        with patch(
            "phaze_particles.models.proton_integrated.get_cuda_manager"
        ) as mock_cuda_manager:
            mock_cuda_instance = Mock()
            mock_cuda_instance.get_status_string.return_value = (
                "CUDA: ❌ Not Available (CPU mode)"
            )
            mock_cuda_manager.return_value = mock_cuda_instance

            # Mock other components
            with patch(
                "phaze_particles.models.proton_integrated.MathematicalFoundations"
            ), patch(
                "phaze_particles.models.proton_integrated.TorusGeometryManager"
            ), patch(
                "phaze_particles.models.proton_integrated.SU2FieldBuilder"
            ), patch(
                "phaze_particles.models.proton_integrated.EnergyDensityCalculator"
            ), patch(
                "phaze_particles.models.proton_integrated.PhysicalQuantitiesCalculator"
            ), patch(
                "phaze_particles.models.proton_integrated.RelaxationSolver"
            ) as mock_solver:

                mock_solver_instance = Mock()
                mock_solver_instance.reset.return_value = None
                mock_solver.return_value = mock_solver_instance

                model = ProtonModel(self.config)
                model.status = ModelStatus.VALIDATED
                model.results = Mock()
                model.error_message = "Test error"

                model.reset()

                self.assertEqual(model.status, ModelStatus.INITIALIZED)
                self.assertIsNone(model.results)
                self.assertIsNone(model.error_message)


class TestProtonModelFactory(unittest.TestCase):
    """Test ProtonModelFactory class."""

    def test_create_default(self):
        """Test creating model with default configuration."""
        with patch(
            "phaze_particles.models.proton_integrated.ProtonModel"
        ) as mock_model:
            mock_instance = Mock()
            mock_model.return_value = mock_instance

            model = ProtonModelFactory.create_default()

            self.assertEqual(model, mock_instance)
            mock_model.assert_called_once()

            # Check that default config was used
            args, kwargs = mock_model.call_args
            config = args[0]
            self.assertIsInstance(config, ModelConfig)
            self.assertEqual(config.grid_size, 64)
            self.assertEqual(config.box_size, 4.0)

    def test_create_quick_test(self):
        """Test creating model for quick testing."""
        with patch(
            "phaze_particles.models.proton_integrated.ProtonModel"
        ) as mock_model:
            mock_instance = Mock()
            mock_model.return_value = mock_instance

            model = ProtonModelFactory.create_quick_test()

            self.assertEqual(model, mock_instance)
            mock_model.assert_called_once()

            # Check that quick test config was used
            args, kwargs = mock_model.call_args
            config = args[0]
            self.assertIsInstance(config, ModelConfig)
            self.assertEqual(config.grid_size, 32)
            self.assertEqual(config.box_size, 2.0)
            self.assertEqual(config.max_iterations, 100)
            self.assertFalse(config.validation_enabled)

    def test_create_from_config(self):
        """Test creating model from configuration file."""
        config_data = {
            "grid_size": 128,
            "box_size": 6.0,
            "torus_config": "clover",
            "max_iterations": 2000,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch(
                "phaze_particles.models.proton_integrated.ProtonModel"
            ) as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance

                model = ProtonModelFactory.create_from_config(config_path)

                self.assertEqual(model, mock_instance)
                mock_model.assert_called_once()

                # Check that config from file was used
                args, kwargs = mock_model.call_args
                config = args[0]
                self.assertIsInstance(config, ModelConfig)
                self.assertEqual(config.grid_size, 128)
                self.assertEqual(config.box_size, 6.0)
                self.assertEqual(config.torus_config, "clover")
                self.assertEqual(config.max_iterations, 2000)

        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()
