#!/usr/bin/env python3
"""
Integration tests for proton model with numerical methods.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np

from phaze_particles.models.proton import ProtonModel


class TestProtonModelIntegration(unittest.TestCase):
    """Test proton model integration with numerical methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ProtonModel()

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        self.assertTrue(self.model.validate_parameters())
        
        # Invalid grid size
        self.model.parameters["grid_size"] = 8
        self.assertFalse(self.model.validate_parameters())
        self.model.parameters["grid_size"] = 64  # Reset
        
        # Invalid relaxation method
        self.model.parameters["relaxation_method"] = "invalid"
        self.assertFalse(self.model.validate_parameters())
        self.model.parameters["relaxation_method"] = "gradient_descent"  # Reset

    @patch('phaze_particles.models.proton.NumericalMethods')
    @patch('phaze_particles.models.proton.TorusGeometryManager')
    @patch('phaze_particles.models.proton.SU2Field')
    @patch('phaze_particles.models.proton.EnergyDensityCalculator')
    @patch('phaze_particles.models.proton.PhysicalQuantitiesCalculator')
    @patch('phaze_particles.models.proton.PhysicsAnalyzer')
    @patch('phaze_particles.models.proton.RelaxationSolver')
    def test_component_initialization(self, mock_solver, mock_analyzer, 
                                    mock_physics_calc, mock_energy_calc,
                                    mock_su2_field, mock_torus, mock_numerical):
        """Test component initialization."""
        # Mock the components
        mock_numerical.return_value = Mock()
        mock_torus.return_value = Mock()
        mock_su2_field.return_value = Mock()
        mock_energy_calc.return_value = Mock()
        mock_physics_calc.return_value = Mock()
        mock_analyzer.return_value = Mock()
        mock_solver.return_value = Mock()
        
        # Initialize components
        self.model._initialize_components()
        
        # Check that all components were initialized
        self.assertIsNotNone(self.model.numerical_methods)
        self.assertIsNotNone(self.model.torus_manager)
        self.assertIsNotNone(self.model.su2_field)
        self.assertIsNotNone(self.model.energy_calculator)
        self.assertIsNotNone(self.model.physics_calculator)
        self.assertIsNotNone(self.model.physics_analyzer)
        self.assertIsNotNone(self.model.relaxation_solver)

    def test_available_configurations(self):
        """Test available configurations."""
        configs = self.model.get_available_configurations()
        expected = ["120deg", "clover", "cartesian"]
        self.assertEqual(configs, expected)

    def test_energy_balance_calculation(self):
        """Test energy balance calculation."""
        balance = self.model.calculate_energy_balance()
        
        # Check structure
        self.assertIn("E2_percentage", balance)
        self.assertIn("E4_percentage", balance)
        self.assertIn("E6_percentage", balance)
        
        # Check values are reasonable
        self.assertGreaterEqual(balance["E2_percentage"], 0)
        self.assertGreaterEqual(balance["E4_percentage"], 0)
        self.assertGreaterEqual(balance["E6_percentage"], 0)

    @patch('phaze_particles.models.proton.ProtonModel._initialize_components')
    @patch('phaze_particles.models.proton.ProtonModel._run_configuration')
    @patch('phaze_particles.models.proton.ProtonModel._combine_results')
    @patch('phaze_particles.models.proton.ProtonModel._perform_physical_analysis')
    def test_run_method_structure(self, mock_analysis, mock_combine, 
                                 mock_run_config, mock_init):
        """Test run method structure."""
        # Mock the methods
        mock_init.return_value = None
        mock_run_config.return_value = {"test": "result"}
        mock_combine.return_value = {"combined": "result"}
        mock_analysis.return_value = None
        
        # Mock numerical methods
        self.model.numerical_methods = Mock()
        self.model.numerical_methods.create_initial_field.return_value = np.ones((2, 2, 2, 2, 2))
        
        # Run the model
        result = self.model.run()
        
        # Check that methods were called
        mock_init.assert_called_once()
        mock_combine.assert_called_once()
        mock_analysis.assert_called_once()
        
        # Check result structure
        self.assertIn("combined", result)

    def test_constraint_functions_creation(self):
        """Test constraint functions creation."""
        # Mock components
        self.model.su2_field = Mock()
        self.model.su2_field.compute_derivatives.return_value = {}
        self.model.physics_calculator = Mock()
        self.model.physics_calculator.baryon_calculator = Mock()
        self.model.physics_calculator.baryon_calculator.compute_baryon_number.return_value = 1.0
        self.model.physics_calculator.charge_density = Mock()
        self.model.physics_calculator.charge_density.compute_charge_density.return_value = np.ones((10, 10, 10))
        self.model.physics_calculator.charge_density.compute_electric_charge.return_value = 1.0
        self.model.torus_manager = Mock()
        self.model.torus_manager.get_profile.return_value = Mock()
        self.model.energy_calculator = Mock()
        self.model.energy_calculator.compute_all_components.return_value = {"E2": 0.5, "E4": 0.5}
        
        # Create constraint functions
        constraints = self.model._create_constraint_functions()
        
        # Check that all constraint functions are created
        self.assertIn('baryon_number', constraints)
        self.assertIn('electric_charge', constraints)
        self.assertIn('energy_balance', constraints)
        
        # Test that functions are callable
        test_field = np.ones((2, 2, 2, 2, 2))
        self.assertIsInstance(constraints['baryon_number'](test_field), float)
        self.assertIsInstance(constraints['electric_charge'](test_field), float)
        self.assertIsInstance(constraints['energy_balance'](test_field), float)

    def test_energy_function_creation(self):
        """Test energy function creation."""
        # Mock components
        self.model.su2_field = Mock()
        self.model.su2_field.compute_derivatives.return_value = {}
        self.model.energy_calculator = Mock()
        self.model.energy_calculator.compute_all_components.return_value = {"E2": 0.5, "E4": 0.5}
        
        # Create energy function
        energy_func = self.model._create_energy_function("120deg")
        
        # Test that function is callable and returns a number
        test_field = np.ones((2, 2, 2, 2, 2))
        energy = energy_func(test_field)
        self.assertIsInstance(energy, (int, float))

    def test_gradient_function_creation(self):
        """Test gradient function creation."""
        # Create gradient function
        gradient_func = self.model._create_gradient_function("120deg")
        
        # Test that function is callable and returns array
        test_field = np.ones((2, 2, 2, 2, 2))
        gradient = gradient_func(test_field)
        self.assertIsInstance(gradient, np.ndarray)
        self.assertEqual(gradient.shape, test_field.shape)

    def test_results_combination(self):
        """Test results combination."""
        # Mock configuration results
        config_results = {
            "120deg": {
                "quantities": Mock(
                    electric_charge=1.0,
                    baryon_number=1.0,
                    mass=938.272,
                    charge_radius=0.84,
                    magnetic_moment=2.793
                ),
                "relaxation": {"iterations": 100, "converged": True}
            }
        }
        
        # Combine results
        combined = self.model._combine_results(config_results)
        
        # Check structure
        self.assertIn("electric_charge", combined)
        self.assertIn("baryon_number", combined)
        self.assertIn("mass", combined)
        self.assertIn("radius", combined)
        self.assertIn("magnetic_moment", combined)
        self.assertIn("energy_balance", combined)
        self.assertIn("configurations", combined)
        self.assertIn("relaxation_info", combined)

    @patch('phaze_particles.models.proton.PhysicsAnalyzer')
    def test_physical_analysis(self, mock_analyzer_class):
        """Test physical analysis."""
        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_results.return_value = []
        mock_analyzer.generate_comparison_table.return_value = "table"
        mock_analyzer.get_overall_quality.return_value = "good"
        mock_analyzer.get_validation_status.return_value = "pass"
        mock_analyzer.get_recommendations.return_value = ["recommendation"]
        
        # Set up model
        self.model.physics_analyzer = mock_analyzer
        self.model.results = {
            "electric_charge": 1.0,
            "baryon_number": 1.0,
            "mass": 938.272,
            "radius": 0.84,
            "magnetic_moment": 2.793,
        }
        
        # Perform analysis
        self.model._perform_physical_analysis()
        
        # Check that analysis was added to results
        self.assertIn("physical_analysis", self.model.results)
        analysis = self.model.results["physical_analysis"]
        self.assertIn("comparison_table", analysis)
        self.assertIn("overall_quality", analysis)
        self.assertIn("validation_status", analysis)
        self.assertIn("recommendations", analysis)
        self.assertIn("detailed_results", analysis)


if __name__ == '__main__':
    unittest.main()
