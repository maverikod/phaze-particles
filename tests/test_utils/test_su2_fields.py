#!/usr/bin/env python3
"""
Tests for SU(2) fields implementation.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
import math
from typing import Dict, Any

from phaze_particles.utils.su2_fields import (
    SU2Field,
    RadialProfile,
    SU2FieldBuilder,
    SU2FieldOperations,
    SU2FieldValidator,
    SU2Fields,
)
from phaze_particles.utils.mathematical_foundations import ArrayBackend


class TestSU2Field(unittest.TestCase):
    """Test SU2Field class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.backend = ArrayBackend()
        self.grid_size = 8
        self.box_size = 2.0

        # Create simple test field
        xp = self.backend.get_array_module()
        self.u_00 = xp.ones(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )
        self.u_01 = xp.zeros(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )
        self.u_10 = xp.zeros(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )
        self.u_11 = xp.ones(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )

    def test_su2_field_creation(self) -> None:
        """Test SU2Field creation."""
        field = SU2Field(
            u_00=self.u_00,
            u_01=self.u_01,
            u_10=self.u_10,
            u_11=self.u_11,
            grid_size=self.grid_size,
            box_size=self.box_size,
            backend=self.backend,
        )

        self.assertEqual(field.grid_size, self.grid_size)
        self.assertEqual(field.box_size, self.box_size)
        self.assertIsInstance(field.backend, ArrayBackend)

    def test_su2_field_validation(self) -> None:
        """Test SU2Field validation."""
        field = SU2Field(
            u_00=self.u_00,
            u_01=self.u_01,
            u_10=self.u_10,
            u_11=self.u_11,
            grid_size=self.grid_size,
            box_size=self.box_size,
            backend=self.backend,
        )

        # Should be valid SU(2) field (identity matrix)
        self.assertTrue(field._is_su2_field())

    def test_get_matrix_at_point(self) -> None:
        """Test getting matrix at specific point."""
        field = SU2Field(
            u_00=self.u_00,
            u_01=self.u_01,
            u_10=self.u_10,
            u_11=self.u_11,
            grid_size=self.grid_size,
            box_size=self.box_size,
            backend=self.backend,
        )

        matrix = field.get_matrix_at_point(0, 0, 0)
        xp = self.backend.get_array_module()
        expected = xp.eye(2, dtype=complex)

        self.assertTrue(xp.allclose(matrix, expected))

    def test_get_determinant(self) -> None:
        """Test determinant calculation."""
        field = SU2Field(
            u_00=self.u_00,
            u_01=self.u_01,
            u_10=self.u_10,
            u_11=self.u_11,
            grid_size=self.grid_size,
            box_size=self.box_size,
            backend=self.backend,
        )

        det = field.get_determinant()
        xp = self.backend.get_array_module()

        # For identity matrix, determinant should be 1
        self.assertTrue(xp.allclose(det, 1.0))


class TestRadialProfile(unittest.TestCase):
    """Test RadialProfile class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.backend = ArrayBackend()
        self.scale = 1.0
        self.center_value = math.pi

    def test_skyrmion_profile(self) -> None:
        """Test skyrmion profile."""
        profile = RadialProfile("skyrmion", self.scale, self.center_value, self.backend)

        xp = self.backend.get_array_module()
        r = xp.array([0.0, 1.0, 2.0])
        f_r = profile.evaluate(r)

        # At r=0, should be center_value
        self.assertAlmostEqual(float(f_r[0]), self.center_value, places=10)

        # Should be decreasing
        self.assertLess(float(f_r[1]), float(f_r[0]))
        self.assertLess(float(f_r[2]), float(f_r[1]))

    def test_exponential_profile(self) -> None:
        """Test exponential profile."""
        profile = RadialProfile(
            "exponential", self.scale, self.center_value, self.backend
        )

        xp = self.backend.get_array_module()
        r = xp.array([0.0, 1.0, 2.0])
        f_r = profile.evaluate(r)

        # At r=0, should be center_value
        self.assertAlmostEqual(float(f_r[0]), self.center_value, places=10)

        # Should be decreasing
        self.assertLess(float(f_r[1]), float(f_r[0]))
        self.assertLess(float(f_r[2]), float(f_r[1]))

    def test_polynomial_profile(self) -> None:
        """Test polynomial profile."""
        profile = RadialProfile(
            "polynomial", self.scale, self.center_value, self.backend
        )

        xp = self.backend.get_array_module()
        r = xp.array([0.0, 1.0, 2.0])
        f_r = profile.evaluate(r)

        # At r=0, should be center_value
        self.assertAlmostEqual(float(f_r[0]), self.center_value, places=10)

        # Should be decreasing
        self.assertLess(float(f_r[1]), float(f_r[0]))
        self.assertLess(float(f_r[2]), float(f_r[1]))

    def test_invalid_profile_type(self) -> None:
        """Test invalid profile type."""
        profile = RadialProfile("invalid", self.scale, self.center_value, self.backend)
        xp = self.backend.get_array_module()
        r = xp.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            profile.evaluate(r)

    def test_get_derivative(self) -> None:
        """Test derivative calculation."""
        profile = RadialProfile("skyrmion", self.scale, self.center_value, self.backend)

        xp = self.backend.get_array_module()
        r = xp.array([1.0, 2.0, 3.0])
        dr = 0.01

        derivative = profile.get_derivative(r, dr)

        # Should be negative (decreasing function)
        self.assertTrue(xp.all(derivative < 0))


class TestSU2FieldBuilder(unittest.TestCase):
    """Test SU2FieldBuilder class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.backend = ArrayBackend()
        self.grid_size = 8
        self.box_size = 2.0
        self.builder = SU2FieldBuilder(self.grid_size, self.box_size, self.backend)

    def test_builder_initialization(self) -> None:
        """Test builder initialization."""
        self.assertEqual(self.builder.grid_size, self.grid_size)
        self.assertEqual(self.builder.box_size, self.box_size)
        self.assertEqual(self.builder.dx, self.box_size / self.grid_size)

    def test_coordinate_grids(self) -> None:
        """Test coordinate grid creation."""
        xp = self.backend.get_array_module()

        # Check grid shapes
        self.assertEqual(
            self.builder.X.shape, (self.grid_size, self.grid_size, self.grid_size)
        )
        self.assertEqual(
            self.builder.Y.shape, (self.grid_size, self.grid_size, self.grid_size)
        )
        self.assertEqual(
            self.builder.Z.shape, (self.grid_size, self.grid_size, self.grid_size)
        )
        self.assertEqual(
            self.builder.R.shape, (self.grid_size, self.grid_size, self.grid_size)
        )

        # Check R is non-negative
        self.assertTrue(xp.all(self.builder.R >= 0))

    def test_build_field(self) -> None:
        """Test field building."""
        xp = self.backend.get_array_module()

        # Create simple direction field
        n_x = xp.zeros((self.grid_size, self.grid_size, self.grid_size))
        n_y = xp.zeros((self.grid_size, self.grid_size, self.grid_size))
        n_z = xp.ones((self.grid_size, self.grid_size, self.grid_size))

        profile = RadialProfile("skyrmion", 1.0, math.pi, self.backend)

        field = self.builder.build_field((n_x, n_y, n_z), profile=profile)

        self.assertIsInstance(field, SU2Field)
        self.assertEqual(field.grid_size, self.grid_size)
        self.assertEqual(field.box_size, self.box_size)


class TestSU2FieldOperations(unittest.TestCase):
    """Test SU2FieldOperations class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.backend = ArrayBackend()
        self.dx = 0.1
        self.operations = SU2FieldOperations(self.dx, self.backend)

        # Create test field
        self.grid_size = 8
        self.box_size = 2.0
        xp = self.backend.get_array_module()

        # Simple field: U = I (identity)
        self.u_00 = xp.ones(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )
        self.u_01 = xp.zeros(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )
        self.u_10 = xp.zeros(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )
        self.u_11 = xp.ones(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )

        self.field = SU2Field(
            u_00=self.u_00,
            u_01=self.u_01,
            u_10=self.u_10,
            u_11=self.u_11,
            grid_size=self.grid_size,
            box_size=self.box_size,
            backend=self.backend,
        )

    def test_compute_left_currents(self) -> None:
        """Test left currents computation."""
        l_x, l_y, l_z = self.operations.compute_left_currents(self.field)

        # For identity field, left currents should be zero
        xp = self.backend.get_array_module()
        self.assertTrue(xp.allclose(l_x["l_00"], 0.0))
        self.assertTrue(xp.allclose(l_x["l_01"], 0.0))
        self.assertTrue(xp.allclose(l_x["l_10"], 0.0))
        self.assertTrue(xp.allclose(l_x["l_11"], 0.0))

    def test_compute_commutators(self) -> None:
        """Test commutators computation."""
        l_x, l_y, l_z = self.operations.compute_left_currents(self.field)
        commutators = self.operations.compute_commutators(l_x, l_y, l_z)

        # For zero currents, commutators should be zero
        xp = self.backend.get_array_module()
        for comm_name in ["xy", "yz", "zx"]:
            self.assertTrue(xp.allclose(commutators[comm_name]["comm_00"], 0.0))
            self.assertTrue(xp.allclose(commutators[comm_name]["comm_01"], 0.0))
            self.assertTrue(xp.allclose(commutators[comm_name]["comm_10"], 0.0))
            self.assertTrue(xp.allclose(commutators[comm_name]["comm_11"], 0.0))

    def test_compute_traces(self) -> None:
        """Test traces computation."""
        l_x, l_y, l_z = self.operations.compute_left_currents(self.field)
        commutators = self.operations.compute_commutators(l_x, l_y, l_z)
        traces = self.operations.compute_traces(l_x, l_y, l_z, commutators)

        # For zero currents, traces should be zero
        xp = self.backend.get_array_module()
        self.assertTrue(xp.allclose(traces["l_squared"], 0.0))
        self.assertTrue(xp.allclose(traces["comm_squared"], 0.0))


class TestSU2FieldValidator(unittest.TestCase):
    """Test SU2FieldValidator class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.backend = ArrayBackend()
        self.validator = SU2FieldValidator(backend=self.backend)

        # Create test field
        self.grid_size = 8
        self.box_size = 2.0
        xp = self.backend.get_array_module()

        # Identity field
        self.u_00 = xp.ones(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )
        self.u_01 = xp.zeros(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )
        self.u_10 = xp.zeros(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )
        self.u_11 = xp.ones(
            (self.grid_size, self.grid_size, self.grid_size), dtype=complex
        )

        self.field = SU2Field(
            u_00=self.u_00,
            u_01=self.u_01,
            u_10=self.u_10,
            u_11=self.u_11,
            grid_size=self.grid_size,
            box_size=self.box_size,
            backend=self.backend,
        )

    def test_validate_field(self) -> None:
        """Test field validation."""
        results = self.validator.validate_field(self.field)

        self.assertIn("unitary", results)
        self.assertIn("determinant", results)
        self.assertIn("continuity", results)
        self.assertIn("boundary_conditions", results)

        # Identity field should pass all checks
        self.assertTrue(results["unitary"])
        self.assertTrue(results["determinant"])
        self.assertTrue(results["continuity"])

    def test_check_unitarity(self) -> None:
        """Test unitarity check."""
        result = self.validator._check_unitarity(self.field)
        self.assertTrue(result)

    def test_check_determinant(self) -> None:
        """Test determinant check."""
        result = self.validator._check_determinant(self.field)
        self.assertTrue(result)

    def test_check_continuity(self) -> None:
        """Test continuity check."""
        result = self.validator._check_continuity(self.field)
        self.assertTrue(result)


class TestSU2Fields(unittest.TestCase):
    """Test SU2Fields main class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.grid_size = 8
        self.box_size = 2.0
        self.su2_fields = SU2Fields(self.grid_size, self.box_size, use_cuda=False)

    def test_initialization(self) -> None:
        """Test SU2Fields initialization."""
        self.assertEqual(self.su2_fields.grid_size, self.grid_size)
        self.assertEqual(self.su2_fields.box_size, self.box_size)
        self.assertEqual(self.su2_fields.dx, self.box_size / self.grid_size)

        self.assertIsInstance(self.su2_fields.builder, SU2FieldBuilder)
        self.assertIsInstance(self.su2_fields.operations, SU2FieldOperations)
        self.assertIsInstance(self.su2_fields.validator, SU2FieldValidator)

    def test_get_cuda_status(self) -> None:
        """Test CUDA status."""
        status = self.su2_fields.get_cuda_status()
        self.assertIsInstance(status, str)
        self.assertGreater(len(status), 0)

    def test_is_cuda_available(self) -> None:
        """Test CUDA availability check."""
        available = self.su2_fields.is_cuda_available()
        self.assertIsInstance(available, bool)

    def test_get_backend_info(self) -> None:
        """Test backend information."""
        info = self.su2_fields.get_backend_info()

        self.assertIn("backend", info)
        self.assertIn("cuda_status", info)
        self.assertIn("cuda_available", info)

        self.assertIsInstance(info["backend"], str)
        self.assertIsInstance(info["cuda_status"], str)
        self.assertIsInstance(info["cuda_available"], str)

    def test_get_field_statistics(self) -> None:
        """Test field statistics."""
        # Create simple test field
        xp = self.su2_fields.backend.get_array_module()
        u_00 = xp.ones((self.grid_size, self.grid_size, self.grid_size), dtype=complex)
        u_01 = xp.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=complex)
        u_10 = xp.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=complex)
        u_11 = xp.ones((self.grid_size, self.grid_size, self.grid_size), dtype=complex)

        field = SU2Field(
            u_00=u_00,
            u_01=u_01,
            u_10=u_10,
            u_11=u_11,
            grid_size=self.grid_size,
            box_size=self.box_size,
            backend=self.su2_fields.backend,
        )

        stats = self.su2_fields.get_field_statistics(field)

        self.assertIn("mean_determinant", stats)
        self.assertIn("std_determinant", stats)
        self.assertIn("min_determinant", stats)
        self.assertIn("max_determinant", stats)
        self.assertIn("field_norm_mean", stats)
        self.assertIn("field_norm_std", stats)

        # For identity field, determinant should be 1
        self.assertAlmostEqual(stats["mean_determinant"], 1.0, places=10)
        self.assertAlmostEqual(stats["std_determinant"], 0.0, places=10)
        self.assertAlmostEqual(stats["min_determinant"], 1.0, places=10)
        self.assertAlmostEqual(stats["max_determinant"], 1.0, places=10)


class TestSU2FieldsIntegration(unittest.TestCase):
    """Integration tests for SU2Fields."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.grid_size = 16
        self.box_size = 4.0
        self.su2_fields = SU2Fields(self.grid_size, self.box_size, use_cuda=False)

    def test_full_workflow(self) -> None:
        """Test complete workflow from field creation to validation."""

        # Create mock torus configuration
        class MockTorusConfig:
            def __init__(self, backend):
                self.backend = backend

            def get_field_direction(self, X, Y, Z):
                xp = self.backend.get_array_module()
                # Simple radial field
                R = xp.sqrt(X**2 + Y**2 + Z**2)
                n_x = X / (R + 1e-10)
                n_y = Y / (R + 1e-10)
                n_z = Z / (R + 1e-10)
                return n_x, n_y, n_z

        torus_config = MockTorusConfig(self.su2_fields.backend)

        # Create field from torus configuration
        field = self.su2_fields.create_field_from_torus(torus_config, "skyrmion", 1.0)

        # Validate field
        validation_results = self.su2_fields.validate_field(field)

        # Compute derivatives
        derivatives = self.su2_fields.compute_field_derivatives(field)

        # Get statistics
        stats = self.su2_fields.get_field_statistics(field)

        # Check results
        self.assertIsInstance(field, SU2Field)
        self.assertIsInstance(validation_results, dict)
        self.assertIsInstance(derivatives, dict)
        self.assertIsInstance(stats, dict)

        # Field should be valid
        self.assertTrue(validation_results["unitary"])
        self.assertTrue(validation_results["determinant"])
        self.assertTrue(validation_results["continuity"])


if __name__ == "__main__":
    unittest.main()
