#!/usr/bin/env python3
"""
Tests for physical quantities and analysis utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
import math
from unittest.mock import Mock, MagicMock

from phaze_particles.utils.physics import (
    PhysicalParameter,
    AnalysisResult,
    PhysicsAnalyzer,
    PhysicalQuantities,
    ChargeDensity,
    BaryonNumberCalculator,
    MagneticMomentCalculator,
    MassCalculator,
    PhysicalQuantitiesCalculator,
)


class TestPhysicalParameter(unittest.TestCase):
    """Test PhysicalParameter dataclass."""

    def test_physical_parameter_creation(self):
        """Test PhysicalParameter creation."""
        param = PhysicalParameter(
            name="test_param",
            calculated_value=1.0,
            experimental_value=1.0,
            tolerance=0.1,
            unit="test_unit",
            description="Test parameter",
        )

        self.assertEqual(param.name, "test_param")
        self.assertEqual(param.calculated_value, 1.0)
        self.assertEqual(param.experimental_value, 1.0)
        self.assertEqual(param.tolerance, 0.1)
        self.assertEqual(param.unit, "test_unit")
        self.assertEqual(param.description, "Test parameter")


class TestAnalysisResult(unittest.TestCase):
    """Test AnalysisResult dataclass."""

    def test_analysis_result_creation(self):
        """Test AnalysisResult creation."""
        param = PhysicalParameter(
            name="test_param",
            calculated_value=1.0,
            experimental_value=1.0,
            tolerance=0.1,
            unit="test_unit",
            description="Test parameter",
        )

        result = AnalysisResult(
            parameter=param,
            deviation_percent=5.0,
            within_tolerance=True,
            quality_rating="good",
        )

        self.assertEqual(result.parameter, param)
        self.assertEqual(result.deviation_percent, 5.0)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.quality_rating, "good")


class TestPhysicsAnalyzer(unittest.TestCase):
    """Test PhysicsAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PhysicsAnalyzer()

    def test_experimental_values_structure(self):
        """Test experimental values structure."""
        exp_values = self.analyzer.EXPERIMENTAL_VALUES

        # Check required parameters
        required_params = [
            "electric_charge",
            "baryon_number",
            "mass",
            "radius",
            "magnetic_moment",
            "energy_balance_e2",
            "energy_balance_e4",
        ]

        for param in required_params:
            self.assertIn(param, exp_values)
            self.assertIn("value", exp_values[param])
            self.assertIn("tolerance", exp_values[param])
            self.assertIn("unit", exp_values[param])
            self.assertIn("description", exp_values[param])

    def test_analyze_results(self):
        """Test analyze_results method."""
        calculated_values = {
            "electric_charge": 1.0,
            "baryon_number": 1.0,
            "mass": 938.272,
            "radius": 0.841,
            "magnetic_moment": 2.793,
        }

        results = self.analyzer.analyze_results(calculated_values)

        self.assertEqual(len(results), 5)
        self.assertIsInstance(results[0], AnalysisResult)

    def test_analyze_parameter(self):
        """Test _analyze_parameter method."""
        param = PhysicalParameter(
            name="test_param",
            calculated_value=1.0,
            experimental_value=1.0,
            tolerance=0.1,
            unit="test_unit",
            description="Test parameter",
        )

        result = self.analyzer._analyze_parameter(param)

        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.deviation_percent, 0.0)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.quality_rating, "excellent")

    def test_determine_quality_rating(self):
        """Test _determine_quality_rating method."""
        # Test excellent rating
        rating = self.analyzer._determine_quality_rating(0.5, 10.0)
        self.assertEqual(rating, "excellent")

        # Test good rating
        rating = self.analyzer._determine_quality_rating(3.0, 10.0)
        self.assertEqual(rating, "good")

        # Test fair rating
        rating = self.analyzer._determine_quality_rating(8.0, 10.0)
        self.assertEqual(rating, "fair")

        # Test poor rating
        rating = self.analyzer._determine_quality_rating(15.0, 10.0)
        self.assertEqual(rating, "poor")

    def test_get_overall_quality(self):
        """Test get_overall_quality method."""
        # Test with no results
        quality = self.analyzer.get_overall_quality()
        self.assertEqual(quality, "unknown")

        # Test with results
        calculated_values = {
            "electric_charge": 1.0,
            "baryon_number": 1.0,
        }
        self.analyzer.analyze_results(calculated_values)
        quality = self.analyzer.get_overall_quality()
        self.assertIn(quality, ["excellent", "good", "fair", "poor"])

    def test_get_validation_status(self):
        """Test get_validation_status method."""
        # Test with no results
        status = self.analyzer.get_validation_status()
        self.assertEqual(status, "fail")

        # Test with results
        calculated_values = {
            "electric_charge": 1.0,
            "baryon_number": 1.0,
        }
        self.analyzer.analyze_results(calculated_values)
        status = self.analyzer.get_validation_status()
        self.assertIn(status, ["pass", "fail"])

    def test_generate_comparison_table(self):
        """Test generate_comparison_table method."""
        # Test with no results
        table = self.analyzer.generate_comparison_table()
        self.assertIn("No analysis results available", table)

        # Test with results
        calculated_values = {
            "electric_charge": 1.0,
            "baryon_number": 1.0,
        }
        self.analyzer.analyze_results(calculated_values)
        table = self.analyzer.generate_comparison_table()
        self.assertIn("PHYSICAL PARAMETER ANALYSIS", table)
        self.assertIn("Electric charge", table)

    def test_get_recommendations(self):
        """Test get_recommendations method."""
        # Test with no results
        recommendations = self.analyzer.get_recommendations()
        self.assertIn("No analysis results available", recommendations[0])

        # Test with results
        calculated_values = {
            "electric_charge": 1.0,
            "baryon_number": 1.0,
        }
        self.analyzer.analyze_results(calculated_values)
        recommendations = self.analyzer.get_recommendations()
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)


class TestPhysicalQuantities(unittest.TestCase):
    """Test PhysicalQuantities dataclass."""

    def test_physical_quantities_creation(self):
        """Test PhysicalQuantities creation."""
        quantities = PhysicalQuantities(
            electric_charge=1.0,
            baryon_number=1.0,
            charge_radius=0.841,
            magnetic_moment=2.793,
            mass=938.272,
            energy=938.272,
            grid_size=64,
            box_size=4.0,
            dx=0.0625,
        )

        self.assertEqual(quantities.electric_charge, 1.0)
        self.assertEqual(quantities.baryon_number, 1.0)
        self.assertEqual(quantities.charge_radius, 0.841)
        self.assertEqual(quantities.magnetic_moment, 2.793)
        self.assertEqual(quantities.mass, 938.272)
        self.assertEqual(quantities.energy, 938.272)
        self.assertEqual(quantities.grid_size, 64)
        self.assertEqual(quantities.box_size, 4.0)
        self.assertEqual(quantities.dx, 0.0625)

    def test_validate_charge(self):
        """Test validate_charge method."""
        quantities = PhysicalQuantities(
            electric_charge=1.0,
            baryon_number=1.0,
            charge_radius=0.841,
            magnetic_moment=2.793,
            mass=938.272,
            energy=938.272,
            grid_size=64,
            box_size=4.0,
            dx=0.0625,
        )

        self.assertTrue(quantities.validate_charge())

        # Test with invalid charge
        quantities.electric_charge = 0.5
        self.assertFalse(quantities.validate_charge())

    def test_validate_baryon_number(self):
        """Test validate_baryon_number method."""
        quantities = PhysicalQuantities(
            electric_charge=1.0,
            baryon_number=1.0,
            charge_radius=0.841,
            magnetic_moment=2.793,
            mass=938.272,
            energy=938.272,
            grid_size=64,
            box_size=4.0,
            dx=0.0625,
        )

        self.assertTrue(quantities.validate_baryon_number())

        # Test with invalid baryon number
        quantities.baryon_number = 0.5
        self.assertFalse(quantities.validate_baryon_number())

    def test_get_validation_status(self):
        """Test get_validation_status method."""
        quantities = PhysicalQuantities(
            electric_charge=1.0,
            baryon_number=1.0,
            charge_radius=0.841,
            magnetic_moment=2.793,
            mass=938.272,
            energy=938.272,
            grid_size=64,
            box_size=4.0,
            dx=0.0625,
        )

        status = quantities.get_validation_status()
        self.assertIsInstance(status, dict)
        self.assertIn("electric_charge", status)
        self.assertIn("baryon_number", status)
        self.assertIn("charge_radius", status)
        self.assertIn("magnetic_moment", status)


class TestChargeDensity(unittest.TestCase):
    """Test ChargeDensity class."""

    def setUp(self):
        """Set up test fixtures."""
        self.charge_density = ChargeDensity(grid_size=32, box_size=4.0)

    def test_initialization(self):
        """Test ChargeDensity initialization."""
        self.assertEqual(self.charge_density.grid_size, 32)
        self.assertEqual(self.charge_density.box_size, 4.0)
        self.assertEqual(self.charge_density.dx, 4.0 / 32)

        # Check coordinate grids
        self.assertEqual(self.charge_density.X.shape, (32, 32, 32))
        self.assertEqual(self.charge_density.Y.shape, (32, 32, 32))
        self.assertEqual(self.charge_density.Z.shape, (32, 32, 32))
        self.assertEqual(self.charge_density.R.shape, (32, 32, 32))

    def test_compute_charge_density(self):
        """Test compute_charge_density method."""
        # Mock field and profile
        field = Mock()
        profile = Mock()
        profile.evaluate.return_value = np.ones((32, 32, 32))

        charge_density = self.charge_density.compute_charge_density(field, profile)

        self.assertEqual(charge_density.shape, (32, 32, 32))
        self.assertTrue(np.all(charge_density >= 0))

    def test_compute_electric_charge(self):
        """Test compute_electric_charge method."""
        # Create test charge density
        charge_density = np.ones((32, 32, 32))

        electric_charge = self.charge_density.compute_electric_charge(charge_density)

        expected_charge = 32**3 * (4.0 / 32) ** 3
        self.assertAlmostEqual(electric_charge, expected_charge, places=6)

    def test_compute_charge_radius(self):
        """Test compute_charge_radius method."""
        # Create test charge density
        charge_density = np.ones((32, 32, 32))

        charge_radius = self.charge_density.compute_charge_radius(charge_density)

        self.assertGreater(charge_radius, 0)
        self.assertIsInstance(charge_radius, float)

    def test_compute_charge_radius_zero_density(self):
        """Test compute_charge_radius with zero density."""
        # Create zero charge density
        charge_density = np.zeros((32, 32, 32))

        charge_radius = self.charge_density.compute_charge_radius(charge_density)

        self.assertEqual(charge_radius, 0.0)


class TestBaryonNumberCalculator(unittest.TestCase):
    """Test BaryonNumberCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = BaryonNumberCalculator(grid_size=32, box_size=4.0)

    def test_initialization(self):
        """Test BaryonNumberCalculator initialization."""
        self.assertEqual(self.calculator.grid_size, 32)
        self.assertEqual(self.calculator.box_size, 4.0)
        self.assertEqual(self.calculator.dx, 4.0 / 32)

    def test_get_epsilon_tensor(self):
        """Test _get_epsilon_tensor method."""
        epsilon = self.calculator._get_epsilon_tensor()

        self.assertEqual(epsilon.shape, (3, 3, 3))
        self.assertEqual(epsilon[0, 1, 2], 1)
        self.assertEqual(epsilon[1, 2, 0], 1)
        self.assertEqual(epsilon[2, 0, 1], 1)
        self.assertEqual(epsilon[0, 2, 1], -1)
        self.assertEqual(epsilon[2, 1, 0], -1)
        self.assertEqual(epsilon[1, 0, 2], -1)

    def test_compute_triple_trace(self):
        """Test _compute_triple_trace method."""
        # Create mock left currents
        l1 = {
            "l_00": np.ones((32, 32, 32)),
            "l_01": np.ones((32, 32, 32)),
            "l_10": np.ones((32, 32, 32)),
            "l_11": np.ones((32, 32, 32)),
        }
        l2 = {
            "l_00": np.ones((32, 32, 32)),
            "l_01": np.ones((32, 32, 32)),
            "l_10": np.ones((32, 32, 32)),
            "l_11": np.ones((32, 32, 32)),
        }
        l3 = {
            "l_00": np.ones((32, 32, 32)),
            "l_01": np.ones((32, 32, 32)),
            "l_10": np.ones((32, 32, 32)),
            "l_11": np.ones((32, 32, 32)),
        }

        trace = self.calculator._compute_triple_trace(l1, l2, l3)

        self.assertEqual(trace.shape, (32, 32, 32))
        self.assertTrue(np.all(trace == 8.0))  # All ones, so trace = 8

    def test_compute_baryon_number(self):
        """Test compute_baryon_number method."""
        # Create mock field derivatives
        field_derivatives = {
            "left_currents": {
                "x": {
                    "l_00": np.ones((32, 32, 32)),
                    "l_01": np.ones((32, 32, 32)),
                    "l_10": np.ones((32, 32, 32)),
                    "l_11": np.ones((32, 32, 32)),
                },
                "y": {
                    "l_00": np.ones((32, 32, 32)),
                    "l_01": np.ones((32, 32, 32)),
                    "l_10": np.ones((32, 32, 32)),
                    "l_11": np.ones((32, 32, 32)),
                },
                "z": {
                    "l_00": np.ones((32, 32, 32)),
                    "l_01": np.ones((32, 32, 32)),
                    "l_10": np.ones((32, 32, 32)),
                    "l_11": np.ones((32, 32, 32)),
                },
            }
        }

        baryon_number = self.calculator.compute_baryon_number(field_derivatives)

        self.assertIsInstance(baryon_number, float)
        self.assertNotEqual(baryon_number, 0)


class TestMagneticMomentCalculator(unittest.TestCase):
    """Test MagneticMomentCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = MagneticMomentCalculator(grid_size=32, box_size=4.0)

    def test_initialization(self):
        """Test MagneticMomentCalculator initialization."""
        self.assertEqual(self.calculator.grid_size, 32)
        self.assertEqual(self.calculator.box_size, 4.0)
        self.assertEqual(self.calculator.dx, 4.0 / 32)

        # Check coordinate grids
        self.assertEqual(self.calculator.X.shape, (32, 32, 32))
        self.assertEqual(self.calculator.Y.shape, (32, 32, 32))
        self.assertEqual(self.calculator.Z.shape, (32, 32, 32))

    def test_get_field_direction(self):
        """Test _get_field_direction method."""
        # Mock field
        field = Mock()
        field.u_00 = np.ones((32, 32, 32))
        field.u_01 = np.ones((32, 32, 32))
        field.u_10 = np.ones((32, 32, 32))
        field.u_11 = np.ones((32, 32, 32))

        n_x, n_y, n_z = self.calculator._get_field_direction(field)

        self.assertEqual(n_x.shape, (32, 32, 32))
        self.assertEqual(n_y.shape, (32, 32, 32))
        self.assertEqual(n_z.shape, (32, 32, 32))

    def test_compute_current_density(self):
        """Test _compute_current_density method."""
        # Mock field and profile
        field = Mock()
        field.u_00 = np.ones((32, 32, 32))
        field.u_01 = np.ones((32, 32, 32))
        field.u_10 = np.ones((32, 32, 32))
        field.u_11 = np.ones((32, 32, 32))

        profile = Mock()
        profile.evaluate.return_value = np.ones((32, 32, 32))

        current_density = self.calculator._compute_current_density(field, profile)

        self.assertIn("x", current_density)
        self.assertIn("y", current_density)
        self.assertIn("z", current_density)
        self.assertEqual(current_density["x"].shape, (32, 32, 32))

    def test_compute_moment_integral(self):
        """Test _compute_moment_integral method."""
        # Create mock current density
        current_density = {
            "x": np.ones((32, 32, 32)),
            "y": np.ones((32, 32, 32)),
            "z": np.ones((32, 32, 32)),
        }

        magnetic_moment = self.calculator._compute_moment_integral(current_density)

        self.assertIsInstance(magnetic_moment, float)

    def test_compute_magnetic_moment(self):
        """Test compute_magnetic_moment method."""
        # Mock field and profile
        field = Mock()
        field.u_00 = np.ones((32, 32, 32))
        field.u_01 = np.ones((32, 32, 32))
        field.u_10 = np.ones((32, 32, 32))
        field.u_11 = np.ones((32, 32, 32))

        profile = Mock()
        profile.evaluate.return_value = np.ones((32, 32, 32))

        magnetic_moment = self.calculator.compute_magnetic_moment(
            field, profile, 938.272
        )

        self.assertIsInstance(magnetic_moment, float)


class TestMassCalculator(unittest.TestCase):
    """Test MassCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = MassCalculator(energy_scale=1.0)

    def test_initialization(self):
        """Test MassCalculator initialization."""
        self.assertEqual(self.calculator.energy_scale, 1.0)

    def test_compute_mass(self):
        """Test compute_mass method."""
        energy = 938.272
        mass = self.calculator.compute_mass(energy)

        self.assertEqual(mass, energy)

    def test_compute_energy_from_mass(self):
        """Test compute_energy_from_mass method."""
        mass = 938.272
        energy = self.calculator.compute_energy_from_mass(mass)

        self.assertEqual(energy, mass)

    def test_different_energy_scale(self):
        """Test with different energy scale."""
        calculator = MassCalculator(energy_scale=2.0)
        energy = 938.272
        mass = calculator.compute_mass(energy)

        self.assertEqual(mass, 2.0 * energy)


class TestPhysicalQuantitiesCalculator(unittest.TestCase):
    """Test PhysicalQuantitiesCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = PhysicalQuantitiesCalculator(
            grid_size=32, box_size=4.0, energy_scale=1.0
        )

    def test_initialization(self):
        """Test PhysicalQuantitiesCalculator initialization."""
        self.assertEqual(self.calculator.grid_size, 32)
        self.assertEqual(self.calculator.box_size, 4.0)
        self.assertEqual(self.calculator.dx, 4.0 / 32)

        # Check sub-calculators
        self.assertIsInstance(self.calculator.charge_density, ChargeDensity)
        self.assertIsInstance(self.calculator.baryon_calculator, BaryonNumberCalculator)
        self.assertIsInstance(
            self.calculator.magnetic_calculator, MagneticMomentCalculator
        )
        self.assertIsInstance(self.calculator.mass_calculator, MassCalculator)

    def test_compute_all_quantities(self):
        """Test compute_all_quantities method."""
        # Mock field and profile
        field = Mock()
        field.u_00 = np.ones((32, 32, 32))
        field.u_01 = np.ones((32, 32, 32))
        field.u_10 = np.ones((32, 32, 32))
        field.u_11 = np.ones((32, 32, 32))

        profile = Mock()
        profile.evaluate.return_value = np.ones((32, 32, 32))

        # Mock field derivatives
        field_derivatives = {
            "left_currents": {
                "x": {
                    "l_00": np.ones((32, 32, 32)),
                    "l_01": np.ones((32, 32, 32)),
                    "l_10": np.ones((32, 32, 32)),
                    "l_11": np.ones((32, 32, 32)),
                },
                "y": {
                    "l_00": np.ones((32, 32, 32)),
                    "l_01": np.ones((32, 32, 32)),
                    "l_10": np.ones((32, 32, 32)),
                    "l_11": np.ones((32, 32, 32)),
                },
                "z": {
                    "l_00": np.ones((32, 32, 32)),
                    "l_01": np.ones((32, 32, 32)),
                    "l_10": np.ones((32, 32, 32)),
                    "l_11": np.ones((32, 32, 32)),
                },
            }
        }

        energy = 938.272

        quantities = self.calculator.compute_all_quantities(
            field, profile, field_derivatives, energy
        )

        self.assertIsInstance(quantities, PhysicalQuantities)
        self.assertEqual(quantities.grid_size, 32)
        self.assertEqual(quantities.box_size, 4.0)
        self.assertEqual(quantities.energy, energy)

    def test_validate_quantities(self):
        """Test validate_quantities method."""
        quantities = PhysicalQuantities(
            electric_charge=1.0,
            baryon_number=1.0,
            charge_radius=0.841,
            magnetic_moment=2.793,
            mass=938.272,
            energy=938.272,
            grid_size=32,
            box_size=4.0,
            dx=0.125,
        )

        validation = self.calculator.validate_quantities(quantities)

        self.assertIn("validation", validation)
        self.assertIn("experimental", validation)
        self.assertIn("calculated", validation)
        self.assertIn("deviations", validation)
        self.assertIn("total_deviation", validation)
        self.assertIn("overall_quality", validation)

    def test_get_quantities_report(self):
        """Test get_quantities_report method."""
        quantities = PhysicalQuantities(
            electric_charge=1.0,
            baryon_number=1.0,
            charge_radius=0.841,
            magnetic_moment=2.793,
            mass=938.272,
            energy=938.272,
            grid_size=32,
            box_size=4.0,
            dx=0.125,
        )

        report = self.calculator.get_quantities_report(quantities)

        self.assertIsInstance(report, str)
        self.assertIn("PHYSICAL QUANTITIES ANALYSIS", report)
        self.assertIn("Electric Charge", report)
        self.assertIn("Baryon Number", report)


if __name__ == "__main__":
    unittest.main()
