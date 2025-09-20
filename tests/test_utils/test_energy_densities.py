#!/usr/bin/env python3
"""
Unit tests for energy densities module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np

from phaze_particles.utils.energy_densities import (
    EnergyDensity,
    BaryonDensity,
    EnergyDensityCalculator,
    EnergyAnalyzer,
    EnergyOptimizer,
    EnergyDensities,
)


class TestEnergyDensity(unittest.TestCase):
    """Test EnergyDensity class."""

    def setUp(self):
        """Set up test fixtures."""
        self.grid_size = 4
        self.box_size = 2.0
        self.dx = 0.5
        self.c2 = 1.0
        self.c4 = 1.0
        self.c6 = 1.0

        # Create test arrays
        shape = (self.grid_size, self.grid_size, self.grid_size)
        self.c2_term = np.ones(shape)
        self.c4_term = 2.0 * np.ones(shape)
        self.c6_term = 0.5 * np.ones(shape)
        self.total_density = self.c2_term + self.c4_term + self.c6_term

        self.energy_density = EnergyDensity(
            c2_term=self.c2_term,
            c4_term=self.c4_term,
            c6_term=self.c6_term,
            total_density=self.total_density,
            grid_size=self.grid_size,
            box_size=self.box_size,
            dx=self.dx,
            c2=self.c2,
            c4=self.c4,
            c6=self.c6,
        )

    def test_get_total_energy(self):
        """Test total energy calculation."""
        expected_energy = np.sum(self.total_density) * self.dx**3
        actual_energy = self.energy_density.get_total_energy()
        self.assertAlmostEqual(actual_energy, expected_energy, places=10)

    def test_get_energy_components(self):
        """Test energy components calculation."""
        components = self.energy_density.get_energy_components()

        expected_e2 = np.sum(self.c2_term) * self.dx**3
        expected_e4 = np.sum(self.c4_term) * self.dx**3
        expected_e6 = np.sum(self.c6_term) * self.dx**3
        expected_total = expected_e2 + expected_e4 + expected_e6

        self.assertAlmostEqual(components["E2"], expected_e2, places=10)
        self.assertAlmostEqual(components["E4"], expected_e4, places=10)
        self.assertAlmostEqual(components["E6"], expected_e6, places=10)
        self.assertAlmostEqual(components["E_total"], expected_total, places=10)

    def test_get_energy_balance(self):
        """Test energy balance calculation."""
        balance = self.energy_density.get_energy_balance()

        components = self.energy_density.get_energy_components()
        total = components["E_total"]

        expected_e2_ratio = components["E2"] / total
        expected_e4_ratio = components["E4"] / total
        expected_e6_ratio = components["E6"] / total

        self.assertAlmostEqual(balance["E2_ratio"], expected_e2_ratio, places=10)
        self.assertAlmostEqual(balance["E4_ratio"], expected_e4_ratio, places=10)
        self.assertAlmostEqual(balance["E6_ratio"], expected_e6_ratio, places=10)

    def test_check_virial_condition_pass(self):
        """Test virial condition check when condition is satisfied."""
        # Create energy density with E2 = E4
        c2_term = np.ones((2, 2, 2))
        c4_term = np.ones((2, 2, 2))
        c6_term = np.zeros((2, 2, 2))
        total_density = c2_term + c4_term + c6_term

        energy_density = EnergyDensity(
            c2_term=c2_term,
            c4_term=c4_term,
            c6_term=c6_term,
            total_density=total_density,
            grid_size=2,
            box_size=1.0,
            dx=0.5,
            c2=1.0,
            c4=1.0,
            c6=1.0,
        )

        self.assertTrue(energy_density.check_virial_condition(tolerance=0.1))

    def test_check_virial_condition_fail(self):
        """Test virial condition check when condition is not satisfied."""
        # Create energy density with E2 >> E4
        c2_term = 10.0 * np.ones((2, 2, 2))
        c4_term = np.ones((2, 2, 2))
        c6_term = np.zeros((2, 2, 2))
        total_density = c2_term + c4_term + c6_term

        energy_density = EnergyDensity(
            c2_term=c2_term,
            c4_term=c4_term,
            c6_term=c6_term,
            total_density=total_density,
            grid_size=2,
            box_size=1.0,
            dx=0.5,
            c2=1.0,
            c4=1.0,
            c6=1.0,
        )

        self.assertFalse(energy_density.check_virial_condition(tolerance=0.1))


class TestBaryonDensity(unittest.TestCase):
    """Test BaryonDensity class."""

    def setUp(self):
        """Set up test fixtures."""
        self.baryon_density = BaryonDensity()

        # Create mock left currents
        self.left_currents = {
            "x": {
                "l_00": np.ones((2, 2, 2)),
                "l_01": np.zeros((2, 2, 2)),
                "l_10": np.zeros((2, 2, 2)),
                "l_11": np.ones((2, 2, 2)),
            },
            "y": {
                "l_00": np.ones((2, 2, 2)),
                "l_01": np.zeros((2, 2, 2)),
                "l_10": np.zeros((2, 2, 2)),
                "l_11": np.ones((2, 2, 2)),
            },
            "z": {
                "l_00": np.ones((2, 2, 2)),
                "l_01": np.zeros((2, 2, 2)),
                "l_10": np.zeros((2, 2, 2)),
                "l_11": np.ones((2, 2, 2)),
            },
        }

    def test_get_epsilon_tensor(self):
        """Test epsilon tensor generation."""
        epsilon = self.baryon_density._get_epsilon_tensor()

        # Check antisymmetric properties
        self.assertEqual(epsilon[0, 1, 2], 1)
        self.assertEqual(epsilon[1, 2, 0], 1)
        self.assertEqual(epsilon[2, 0, 1], 1)
        self.assertEqual(epsilon[0, 2, 1], -1)
        self.assertEqual(epsilon[2, 1, 0], -1)
        self.assertEqual(epsilon[1, 0, 2], -1)

        # Check that other elements are zero
        self.assertEqual(epsilon[0, 0, 0], 0)
        self.assertEqual(epsilon[1, 1, 1], 0)
        self.assertEqual(epsilon[2, 2, 2], 0)

    def test_compute_triple_trace(self):
        """Test triple trace computation."""
        l1 = {
            "l_00": np.ones((2, 2, 2)),
            "l_01": np.zeros((2, 2, 2)),
            "l_10": np.zeros((2, 2, 2)),
            "l_11": np.ones((2, 2, 2)),
        }
        l2 = {
            "l_00": np.ones((2, 2, 2)),
            "l_01": np.zeros((2, 2, 2)),
            "l_10": np.zeros((2, 2, 2)),
            "l_11": np.ones((2, 2, 2)),
        }
        l3 = {
            "l_00": np.ones((2, 2, 2)),
            "l_01": np.zeros((2, 2, 2)),
            "l_10": np.zeros((2, 2, 2)),
            "l_11": np.ones((2, 2, 2)),
        }

        trace = self.baryon_density._compute_triple_trace(l1, l2, l3)

        # For identity matrices, trace should be 2
        expected = 2.0 * np.ones((2, 2, 2))
        np.testing.assert_array_almost_equal(trace, expected)

    def test_compute_baryon_density(self):
        """Test baryon density computation."""
        baryon_density = self.baryon_density.compute_baryon_density(self.left_currents)

        # Check that result has correct shape
        self.assertEqual(baryon_density.shape, (2, 2, 2))

        # Check that result is not all zeros (for non-trivial input)
        self.assertFalse(np.allclose(baryon_density, 0))


class TestEnergyDensityCalculator(unittest.TestCase):
    """Test EnergyDensityCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.grid_size = 4
        self.box_size = 2.0
        self.c2 = 1.0
        self.c4 = 1.0
        self.c6 = 1.0

        self.calculator = EnergyDensityCalculator(
            self.grid_size, self.box_size, self.c2, self.c4, self.c6
        )

        # Create mock field derivatives
        shape = (self.grid_size, self.grid_size, self.grid_size)
        self.field_derivatives = {
            "traces": {
                "l_squared": np.ones(shape),
                "comm_squared": 2.0 * np.ones(shape),
            },
            "left_currents": {
                "x": {
                    "l_00": np.ones(shape),
                    "l_01": np.zeros(shape),
                    "l_10": np.zeros(shape),
                    "l_11": np.ones(shape),
                },
                "y": {
                    "l_00": np.ones(shape),
                    "l_01": np.zeros(shape),
                    "l_10": np.zeros(shape),
                    "l_11": np.ones(shape),
                },
                "z": {
                    "l_00": np.ones(shape),
                    "l_01": np.zeros(shape),
                    "l_10": np.zeros(shape),
                    "l_11": np.ones(shape),
                },
            },
        }

    def test_init(self):
        """Test calculator initialization."""
        self.assertEqual(self.calculator.grid_size, self.grid_size)
        self.assertEqual(self.calculator.box_size, self.box_size)
        self.assertEqual(self.calculator.dx, self.box_size / self.grid_size)
        self.assertEqual(self.calculator.c2, self.c2)
        self.assertEqual(self.calculator.c4, self.c4)
        self.assertEqual(self.calculator.c6, self.c6)
        self.assertIsInstance(self.calculator.baryon_density, BaryonDensity)

    def test_compute_energy_density(self):
        """Test energy density computation."""
        energy_density = self.calculator.compute_energy_density(self.field_derivatives)

        # Check that result is EnergyDensity instance
        self.assertIsInstance(energy_density, EnergyDensity)

        # Check that all components have correct shape
        shape = (self.grid_size, self.grid_size, self.grid_size)
        self.assertEqual(energy_density.c2_term.shape, shape)
        self.assertEqual(energy_density.c4_term.shape, shape)
        self.assertEqual(energy_density.c6_term.shape, shape)
        self.assertEqual(energy_density.total_density.shape, shape)

        # Check that total density is sum of components
        expected_total = (
            energy_density.c2_term + energy_density.c4_term + energy_density.c6_term
        )
        np.testing.assert_array_almost_equal(
            energy_density.total_density, expected_total
        )

    def test_compute_baryon_number(self):
        """Test baryon number computation."""
        baryon_number = self.calculator.compute_baryon_number(self.field_derivatives)

        # Check that result is a float
        self.assertIsInstance(baryon_number, float)

        # Check that result is not NaN or infinite
        self.assertFalse(np.isnan(baryon_number))
        self.assertFalse(np.isinf(baryon_number))


class TestEnergyAnalyzer(unittest.TestCase):
    """Test EnergyAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = EnergyAnalyzer(tolerance=0.05)

        # Create test energy density
        self.grid_size = 4
        self.box_size = 2.0
        self.dx = 0.5

        shape = (self.grid_size, self.grid_size, self.grid_size)
        self.c2_term = np.ones(shape)
        self.c4_term = np.ones(shape)
        self.c6_term = np.zeros(shape)
        self.total_density = self.c2_term + self.c4_term + self.c6_term

        self.energy_density = EnergyDensity(
            c2_term=self.c2_term,
            c4_term=self.c4_term,
            c6_term=self.c6_term,
            total_density=self.total_density,
            grid_size=self.grid_size,
            box_size=self.box_size,
            dx=self.dx,
            c2=1.0,
            c4=1.0,
            c6=1.0,
        )

    def test_init(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.tolerance, 0.05)

    def test_analyze_energy(self):
        """Test energy analysis."""
        analysis = self.analyzer.analyze_energy(self.energy_density)

        # Check that all required keys are present
        required_keys = [
            "components",
            "balance",
            "virial_condition",
            "density_stats",
            "quality",
        ]
        for key in required_keys:
            self.assertIn(key, analysis)

        # Check components
        self.assertIn("E2", analysis["components"])
        self.assertIn("E4", analysis["components"])
        self.assertIn("E6", analysis["components"])
        self.assertIn("E_total", analysis["components"])

        # Check balance
        self.assertIn("E2_ratio", analysis["balance"])
        self.assertIn("E4_ratio", analysis["balance"])
        self.assertIn("E6_ratio", analysis["balance"])

        # Check quality
        self.assertIn("overall_quality", analysis["quality"])
        self.assertIn("balance_quality", analysis["quality"])
        self.assertIn("virial_condition", analysis["quality"])
        self.assertIn("recommendations", analysis["quality"])

    def test_compute_density_statistics(self):
        """Test density statistics computation."""
        stats = self.analyzer._compute_density_statistics(self.energy_density)

        required_keys = [
            "total_mean",
            "total_std",
            "total_max",
            "total_min",
            "c2_mean",
            "c4_mean",
            "c6_mean",
        ]
        for key in required_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], float)

    def test_assess_energy_quality(self):
        """Test energy quality assessment."""
        quality = self.analyzer._assess_energy_quality(self.energy_density)

        # Check that quality assessment has required keys
        required_keys = [
            "overall_quality",
            "balance_quality",
            "virial_condition",
            "recommendations",
        ]
        for key in required_keys:
            self.assertIn(key, quality)

        # Check that quality levels are valid
        valid_qualities = ["excellent", "good", "fair", "poor"]
        self.assertIn(quality["overall_quality"], valid_qualities)
        self.assertIn(quality["balance_quality"], valid_qualities)

        # Check that recommendations is a list
        self.assertIsInstance(quality["recommendations"], list)

    def test_get_energy_recommendations(self):
        """Test energy recommendations generation."""
        balance = {"E2_ratio": 0.7, "E4_ratio": 0.3, "E6_ratio": 0.0}
        virial_ok = False

        positivity = {"total_energy_positive": True, "energy_density_positive": True}
        recommendations = self.analyzer._get_energy_recommendations(balance, virial_ok, positivity)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)


class TestEnergyOptimizer(unittest.TestCase):
    """Test EnergyOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = EnergyOptimizer(
            target_e2_ratio=0.5, target_e4_ratio=0.5, tolerance=0.05
        )

        # Create mock field derivatives
        self.field_derivatives = {
            "traces": {
                "l_squared": np.ones((2, 2, 2)),
                "comm_squared": np.ones((2, 2, 2)),
            },
            "left_currents": {
                "x": {
                    "l_00": np.ones((2, 2, 2)),
                    "l_01": np.zeros((2, 2, 2)),
                    "l_10": np.zeros((2, 2, 2)),
                    "l_11": np.ones((2, 2, 2)),
                },
                "y": {
                    "l_00": np.ones((2, 2, 2)),
                    "l_01": np.zeros((2, 2, 2)),
                    "l_10": np.zeros((2, 2, 2)),
                    "l_11": np.ones((2, 2, 2)),
                },
                "z": {
                    "l_00": np.ones((2, 2, 2)),
                    "l_01": np.zeros((2, 2, 2)),
                    "l_10": np.zeros((2, 2, 2)),
                    "l_11": np.ones((2, 2, 2)),
                },
            },
        }

    def test_init(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.target_e2_ratio, 0.5)
        self.assertEqual(self.optimizer.target_e4_ratio, 0.5)
        self.assertEqual(self.optimizer.tolerance, 0.05)

    def test_optimize_constants(self):
        """Test constants optimization."""
        initial_c2 = 1.0
        initial_c4 = 1.0
        initial_c6 = 1.0

        optimized = self.optimizer.optimize_constants(
            initial_c2,
            initial_c4,
            initial_c6,
            self.field_derivatives,
            max_iterations=10,
        )

        # Check that result has required keys
        required_keys = ["c2", "c4", "c6"]
        for key in required_keys:
            self.assertIn(key, optimized)
            self.assertIsInstance(optimized[key], float)
            self.assertGreater(optimized[key], 0)


class TestEnergyDensities(unittest.TestCase):
    """Test EnergyDensities main class."""

    def setUp(self):
        """Set up test fixtures."""
        self.energy_densities = EnergyDensities(
            grid_size=4, box_size=2.0, c2=1.0, c4=1.0, c6=1.0
        )

        # Create mock field derivatives
        self.field_derivatives = {
            "traces": {
                "l_squared": np.ones((4, 4, 4)),
                "comm_squared": np.ones((4, 4, 4)),
            },
            "left_currents": {
                "x": {
                    "l_00": np.ones((4, 4, 4)),
                    "l_01": np.zeros((4, 4, 4)),
                    "l_10": np.zeros((4, 4, 4)),
                    "l_11": np.ones((4, 4, 4)),
                },
                "y": {
                    "l_00": np.ones((4, 4, 4)),
                    "l_01": np.zeros((4, 4, 4)),
                    "l_10": np.zeros((4, 4, 4)),
                    "l_11": np.ones((4, 4, 4)),
                },
                "z": {
                    "l_00": np.ones((4, 4, 4)),
                    "l_01": np.zeros((4, 4, 4)),
                    "l_10": np.zeros((4, 4, 4)),
                    "l_11": np.ones((4, 4, 4)),
                },
            },
        }

    def test_init(self):
        """Test main class initialization."""
        self.assertEqual(self.energy_densities.grid_size, 4)
        self.assertEqual(self.energy_densities.box_size, 2.0)
        self.assertEqual(self.energy_densities.c2, 1.0)
        self.assertEqual(self.energy_densities.c4, 1.0)
        self.assertEqual(self.energy_densities.c6, 1.0)

        # Check that sub-components are initialized
        self.assertIsInstance(self.energy_densities.calculator, EnergyDensityCalculator)
        self.assertIsInstance(self.energy_densities.analyzer, EnergyAnalyzer)
        self.assertIsInstance(self.energy_densities.optimizer, EnergyOptimizer)

    def test_compute_energy(self):
        """Test energy computation."""
        energy_density = self.energy_densities.compute_energy(self.field_derivatives)

        self.assertIsInstance(energy_density, EnergyDensity)

    def test_compute_baryon_number(self):
        """Test baryon number computation."""
        baryon_number = self.energy_densities.compute_baryon_number(
            self.field_derivatives
        )

        self.assertIsInstance(baryon_number, float)
        self.assertFalse(np.isnan(baryon_number))
        self.assertFalse(np.isinf(baryon_number))

    def test_analyze_energy(self):
        """Test energy analysis."""
        energy_density = self.energy_densities.compute_energy(self.field_derivatives)
        analysis = self.energy_densities.analyze_energy(energy_density)

        self.assertIsInstance(analysis, dict)
        self.assertIn("components", analysis)
        self.assertIn("balance", analysis)
        self.assertIn("quality", analysis)

    def test_optimize_constants(self):
        """Test constants optimization."""
        optimized = self.energy_densities.optimize_constants(self.field_derivatives)

        self.assertIsInstance(optimized, dict)
        self.assertIn("c2", optimized)
        self.assertIn("c4", optimized)
        self.assertIn("c6", optimized)

    def test_get_energy_report(self):
        """Test energy report generation."""
        energy_density = self.energy_densities.compute_energy(self.field_derivatives)
        report = self.energy_densities.get_energy_report(energy_density)

        self.assertIsInstance(report, str)
        self.assertIn("ENERGY DENSITY ANALYSIS", report)
        self.assertIn("Energy Components:", report)
        self.assertIn("Energy Balance:", report)
        self.assertIn("Quality Assessment:", report)


if __name__ == "__main__":
    unittest.main()
