#!/usr/bin/env python3
"""
Tests for physical quantities calculation.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
import math
from unittest.mock import Mock, MagicMock

from phaze_particles.utils.physics import (
    PhysicalQuantities,
    ChargeDensity,
    BaryonNumberCalculator,
    MagneticMomentCalculator,
    MassCalculator,
    PhysicalQuantitiesCalculator,
    PhysicsAnalyzer,
    PhysicalParameter,
    AnalysisResult,
)


class TestPhysicalQuantities(unittest.TestCase):
    """Test PhysicalQuantities dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.quantities = PhysicalQuantities(
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

    def test_validate_charge(self):
        """Test electric charge validation."""
        # Valid charge
        self.assertTrue(self.quantities.validate_charge())

        # Invalid charge
        self.quantities.electric_charge = 0.5
        self.assertFalse(self.quantities.validate_charge())

    def test_validate_baryon_number(self):
        """Test baryon number validation."""
        # Valid baryon number
        self.assertTrue(self.quantities.validate_baryon_number())

        # Invalid baryon number
        self.quantities.baryon_number = 0.5
        self.assertFalse(self.quantities.validate_baryon_number())

    def test_get_validation_status(self):
        """Test validation status for all quantities."""
        status = self.quantities.get_validation_status()

        self.assertIn("electric_charge", status)
        self.assertIn("baryon_number", status)
        self.assertIn("charge_radius", status)
        self.assertIn("magnetic_moment", status)

        # All should be valid for default values
        self.assertTrue(all(status.values()))


class TestChargeDensity(unittest.TestCase):
    """Test ChargeDensity calculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.grid_size = 32
        self.box_size = 4.0
        self.charge_density = ChargeDensity(self.grid_size, self.box_size)

        # Mock profile
        self.profile = Mock()
        self.profile.evaluate.return_value = np.ones(
            (self.grid_size, self.grid_size, self.grid_size)
        )

        # Mock field
        self.field = Mock()

    def test_initialization(self):
        """Test ChargeDensity initialization."""
        self.assertEqual(self.charge_density.grid_size, self.grid_size)
        self.assertEqual(self.charge_density.box_size, self.box_size)
        self.assertEqual(self.charge_density.dx, self.box_size / self.grid_size)

        # Check coordinate grids
        self.assertEqual(
            self.charge_density.X.shape,
            (self.grid_size, self.grid_size, self.grid_size),
        )
        self.assertEqual(
            self.charge_density.Y.shape,
            (self.grid_size, self.grid_size, self.grid_size),
        )
        self.assertEqual(
            self.charge_density.Z.shape,
            (self.grid_size, self.grid_size, self.grid_size),
        )
        self.assertEqual(
            self.charge_density.R.shape,
            (self.grid_size, self.grid_size, self.grid_size),
        )

    def test_compute_charge_density(self):
        """Test charge density computation."""
        charge_density = self.charge_density.compute_charge_density(
            self.field, self.profile
        )

        self.assertEqual(
            charge_density.shape, (self.grid_size, self.grid_size, self.grid_size)
        )
        self.assertTrue(np.all(charge_density >= 0))  # Non-negative

    def test_compute_electric_charge(self):
        """Test electric charge computation."""
        # Create test charge density
        charge_density = np.ones((self.grid_size, self.grid_size, self.grid_size))
        # Normalize to get total charge = 1.0
        total_volume = self.grid_size**3 * self.charge_density.dx**3
        charge_density *= 1.0 / total_volume

        electric_charge = self.charge_density.compute_electric_charge(charge_density)

        self.assertAlmostEqual(electric_charge, 1.0, places=6)

    def test_compute_charge_radius(self):
        """Test charge radius computation."""
        # Create test charge density (Gaussian-like)
        r = self.charge_density.R
        sigma = 0.5
        charge_density = np.exp(-(r**2) / (2 * sigma**2))

        charge_radius = self.charge_density.compute_charge_radius(charge_density)

        self.assertGreater(charge_radius, 0)
        self.assertLess(charge_radius, self.box_size)


class TestBaryonNumberCalculator(unittest.TestCase):
    """Test BaryonNumberCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.grid_size = 32
        self.box_size = 4.0
        self.calculator = BaryonNumberCalculator(self.grid_size, self.box_size)

        # Mock field derivatives
        self.field_derivatives = {
            "left_currents": {
                "x": {
                    "l_00": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                    "l_01": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_10": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_11": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                },
                "y": {
                    "l_00": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                    "l_01": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_10": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_11": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                },
                "z": {
                    "l_00": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                    "l_01": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_10": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_11": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                },
            }
        }

    def test_initialization(self):
        """Test BaryonNumberCalculator initialization."""
        self.assertEqual(self.calculator.grid_size, self.grid_size)
        self.assertEqual(self.calculator.box_size, self.box_size)
        self.assertEqual(self.calculator.dx, self.box_size / self.grid_size)

    def test_get_epsilon_tensor(self):
        """Test epsilon tensor generation."""
        epsilon = self.calculator._get_epsilon_tensor()

        self.assertEqual(epsilon.shape, (3, 3, 3))

        # Check antisymmetry
        self.assertEqual(epsilon[0, 1, 2], 1)
        self.assertEqual(epsilon[1, 2, 0], 1)
        self.assertEqual(epsilon[2, 0, 1], 1)
        self.assertEqual(epsilon[0, 2, 1], -1)
        self.assertEqual(epsilon[2, 1, 0], -1)
        self.assertEqual(epsilon[1, 0, 2], -1)

    def test_compute_triple_trace(self):
        """Test triple trace computation."""
        l1 = self.field_derivatives["left_currents"]["x"]
        l2 = self.field_derivatives["left_currents"]["y"]
        l3 = self.field_derivatives["left_currents"]["z"]

        trace = self.calculator._compute_triple_trace(l1, l2, l3)

        self.assertEqual(trace.shape, (self.grid_size, self.grid_size, self.grid_size))

    def test_compute_baryon_number(self):
        """Test baryon number computation."""
        baryon_number = self.calculator.compute_baryon_number(self.field_derivatives)

        self.assertIsInstance(baryon_number, float)
        self.assertIsNotNone(baryon_number)


class TestMagneticMomentCalculator(unittest.TestCase):
    """Test MagneticMomentCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.grid_size = 32
        self.box_size = 4.0
        self.calculator = MagneticMomentCalculator(self.grid_size, self.box_size)

        # Mock field
        self.field = Mock()
        self.field.u_00 = np.ones((self.grid_size, self.grid_size, self.grid_size))
        self.field.u_01 = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        self.field.u_10 = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        self.field.u_11 = np.ones((self.grid_size, self.grid_size, self.grid_size))

        # Mock profile
        self.profile = Mock()
        self.profile.evaluate.return_value = np.ones(
            (self.grid_size, self.grid_size, self.grid_size)
        )

        self.mass = 938.272

    def test_initialization(self):
        """Test MagneticMomentCalculator initialization."""
        self.assertEqual(self.calculator.grid_size, self.grid_size)
        self.assertEqual(self.calculator.box_size, self.box_size)
        self.assertEqual(self.calculator.dx, self.box_size / self.grid_size)

        # Check coordinate grids
        self.assertEqual(
            self.calculator.X.shape, (self.grid_size, self.grid_size, self.grid_size)
        )
        self.assertEqual(
            self.calculator.Y.shape, (self.grid_size, self.grid_size, self.grid_size)
        )
        self.assertEqual(
            self.calculator.Z.shape, (self.grid_size, self.grid_size, self.grid_size)
        )

    def test_get_field_direction(self):
        """Test field direction extraction."""
        n_x, n_y, n_z = self.calculator._get_field_direction(self.field)

        self.assertEqual(n_x.shape, (self.grid_size, self.grid_size, self.grid_size))
        self.assertEqual(n_y.shape, (self.grid_size, self.grid_size, self.grid_size))
        self.assertEqual(n_z.shape, (self.grid_size, self.grid_size, self.grid_size))

    def test_compute_current_density(self):
        """Test current density computation."""
        current_density = self.calculator._compute_current_density(
            self.field, self.profile
        )

        self.assertIn("x", current_density)
        self.assertIn("y", current_density)
        self.assertIn("z", current_density)

        for component in current_density.values():
            self.assertEqual(
                component.shape, (self.grid_size, self.grid_size, self.grid_size)
            )

    def test_compute_moment_integral(self):
        """Test moment integral computation."""
        current_density = {
            "x": np.ones((self.grid_size, self.grid_size, self.grid_size)),
            "y": np.ones((self.grid_size, self.grid_size, self.grid_size)),
            "z": np.ones((self.grid_size, self.grid_size, self.grid_size)),
        }

        moment = self.calculator._compute_moment_integral(current_density)

        self.assertIsInstance(moment, float)

    def test_compute_magnetic_moment(self):
        """Test magnetic moment computation."""
        magnetic_moment = self.calculator.compute_magnetic_moment(
            self.field, self.profile, self.mass
        )

        self.assertIsInstance(magnetic_moment, float)


class TestMassCalculator(unittest.TestCase):
    """Test MassCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.energy_scale = 1.0
        self.calculator = MassCalculator(self.energy_scale)

    def test_initialization(self):
        """Test MassCalculator initialization."""
        self.assertEqual(self.calculator.energy_scale, self.energy_scale)

    def test_compute_mass(self):
        """Test mass computation."""
        energy = 938.272
        mass = self.calculator.compute_mass(energy)

        self.assertAlmostEqual(mass, energy, places=6)

    def test_compute_energy_from_mass(self):
        """Test energy computation from mass."""
        mass = 938.272
        energy = self.calculator.compute_energy_from_mass(mass)

        self.assertAlmostEqual(energy, mass, places=6)

    def test_energy_scale_factor(self):
        """Test energy scale factor."""
        energy_scale = 2.0
        calculator = MassCalculator(energy_scale)

        energy = 100.0
        mass = calculator.compute_mass(energy)

        self.assertAlmostEqual(mass, 200.0, places=6)


class TestPhysicalQuantitiesCalculator(unittest.TestCase):
    """Test PhysicalQuantitiesCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.grid_size = 32
        self.box_size = 4.0
        self.energy_scale = 1.0
        self.calculator = PhysicalQuantitiesCalculator(
            self.grid_size, self.box_size, self.energy_scale
        )

        # Mock field
        self.field = Mock()
        self.field.u_00 = np.ones((self.grid_size, self.grid_size, self.grid_size))
        self.field.u_01 = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        self.field.u_10 = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        self.field.u_11 = np.ones((self.grid_size, self.grid_size, self.grid_size))

        # Mock profile
        self.profile = Mock()
        self.profile.evaluate.return_value = np.ones(
            (self.grid_size, self.grid_size, self.grid_size)
        )

        # Mock field derivatives
        self.field_derivatives = {
            "left_currents": {
                "x": {
                    "l_00": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                    "l_01": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_10": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_11": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                },
                "y": {
                    "l_00": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                    "l_01": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_10": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_11": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                },
                "z": {
                    "l_00": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                    "l_01": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_10": np.zeros((self.grid_size, self.grid_size, self.grid_size)),
                    "l_11": np.ones((self.grid_size, self.grid_size, self.grid_size)),
                },
            }
        }

        self.energy = 938.272

    def test_initialization(self):
        """Test PhysicalQuantitiesCalculator initialization."""
        self.assertEqual(self.calculator.grid_size, self.grid_size)
        self.assertEqual(self.calculator.box_size, self.box_size)
        self.assertEqual(self.calculator.dx, self.box_size / self.grid_size)

        # Check sub-calculators
        self.assertIsInstance(self.calculator.charge_density, ChargeDensity)
        self.assertIsInstance(self.calculator.baryon_calculator, BaryonNumberCalculator)
        self.assertIsInstance(
            self.calculator.magnetic_calculator, MagneticMomentCalculator
        )
        self.assertIsInstance(self.calculator.mass_calculator, MassCalculator)

    def test_compute_all_quantities(self):
        """Test computation of all physical quantities."""
        quantities = self.calculator.compute_all_quantities(
            self.field, self.profile, self.field_derivatives, self.energy
        )

        self.assertIsInstance(quantities, PhysicalQuantities)
        self.assertIsInstance(quantities.electric_charge, float)
        self.assertIsInstance(quantities.baryon_number, float)
        self.assertIsInstance(quantities.charge_radius, float)
        self.assertIsInstance(quantities.magnetic_moment, float)
        self.assertIsInstance(quantities.mass, float)
        self.assertIsInstance(quantities.energy, float)

    def test_validate_quantities(self):
        """Test quantities validation."""
        quantities = self.calculator.compute_all_quantities(
            self.field, self.profile, self.field_derivatives, self.energy
        )

        validation = self.calculator.validate_quantities(quantities)

        self.assertIn("validation", validation)
        self.assertIn("experimental", validation)
        self.assertIn("calculated", validation)
        self.assertIn("deviations", validation)
        self.assertIn("total_deviation", validation)
        self.assertIn("overall_quality", validation)

    def test_get_quantities_report(self):
        """Test quantities report generation."""
        quantities = self.calculator.compute_all_quantities(
            self.field, self.profile, self.field_derivatives, self.energy
        )

        report = self.calculator.get_quantities_report(quantities)

        self.assertIsInstance(report, str)
        self.assertIn("PHYSICAL QUANTITIES ANALYSIS", report)
        self.assertIn("Electric Charge:", report)
        self.assertIn("Baryon Number:", report)
        self.assertIn("Charge Radius:", report)
        self.assertIn("Magnetic Moment:", report)


class TestPhysicsAnalyzer(unittest.TestCase):
    """Test PhysicsAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PhysicsAnalyzer()

    def test_initialization(self):
        """Test PhysicsAnalyzer initialization."""
        self.assertEqual(len(self.analyzer.results), 0)
        self.assertIsInstance(self.analyzer.EXPERIMENTAL_VALUES, dict)

    def test_analyze_results(self):
        """Test results analysis."""
        calculated_values = {
            "electric_charge": 1.0,
            "baryon_number": 1.0,
            "mass": 938.272,
            "radius": 0.841,
            "magnetic_moment": 2.793,
        }

        results = self.analyzer.analyze_results(calculated_values)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for result in results:
            self.assertIsInstance(result, AnalysisResult)

    def test_get_overall_quality(self):
        """Test overall quality assessment."""
        # Test with no results
        quality = self.analyzer.get_overall_quality()
        self.assertEqual(quality, "unknown")

        # Test with mock results
        param = PhysicalParameter(
            name="test",
            calculated_value=1.0,
            experimental_value=1.0,
            tolerance=0.1,
            unit="test",
            description="Test parameter",
        )

        result = AnalysisResult(
            parameter=param,
            deviation_percent=0.05,
            within_tolerance=True,
            quality_rating="excellent",
        )

        self.analyzer.results = [result]
        quality = self.analyzer.get_overall_quality()
        self.assertIn(quality, ["excellent", "good", "fair", "poor"])

    def test_get_validation_status(self):
        """Test validation status."""
        # Test with no results
        status = self.analyzer.get_validation_status()
        self.assertEqual(status, "fail")

        # Test with mock results
        param = PhysicalParameter(
            name="test",
            calculated_value=1.0,
            experimental_value=1.0,
            tolerance=0.1,
            unit="test",
            description="Test parameter",
        )

        result = AnalysisResult(
            parameter=param,
            deviation_percent=0.05,
            within_tolerance=True,
            quality_rating="excellent",
        )

        self.analyzer.results = [result]
        status = self.analyzer.get_validation_status()
        self.assertIn(status, ["pass", "fail"])

    def test_generate_comparison_table(self):
        """Test comparison table generation."""
        # Test with no results
        table = self.analyzer.generate_comparison_table()
        self.assertIn("No analysis results available", table)

        # Test with mock results
        param = PhysicalParameter(
            name="test",
            calculated_value=1.0,
            experimental_value=1.0,
            tolerance=0.1,
            unit="test",
            description="Test parameter",
        )

        result = AnalysisResult(
            parameter=param,
            deviation_percent=0.05,
            within_tolerance=True,
            quality_rating="excellent",
        )

        self.analyzer.results = [result]
        table = self.analyzer.generate_comparison_table()
        self.assertIn("PHYSICAL PARAMETER ANALYSIS", table)
        self.assertIn("Test parameter", table)

    def test_get_recommendations(self):
        """Test recommendations generation."""
        recommendations = self.analyzer.get_recommendations()

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)


if __name__ == "__main__":
    unittest.main()
