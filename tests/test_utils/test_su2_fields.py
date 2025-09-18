#!/usr/bin/env python3
"""
Tests for SU(2) fields.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
from phaze_particles.utils.su2_fields import (
    SU2Field,
    RadialProfile,
    SU2FieldBuilder,
    SU2FieldOperations,
    SU2FieldValidator,
    SU2Fields,
)
from phaze_particles.utils.torus_geometries import Torus120Degrees


class TestSU2Field(unittest.TestCase):
    """Test SU2Field class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False  # Force CPU mode for tests

        # Create a simple SU(2) field
        grid_size = 8
        box_size = 2.0
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        z = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)

        # Simple skyrmion-like field
        f_r = np.pi * np.exp(-R)
        cos_f = np.cos(f_r)
        sin_f = np.sin(f_r)

        # U = cos f(r) 1 + i sin f(r) σ_z
        u_00 = cos_f + 1j * sin_f
        u_01 = np.zeros_like(cos_f)
        u_10 = np.zeros_like(cos_f)
        u_11 = cos_f - 1j * sin_f

        self.field = SU2Field(
            u_00=u_00,
            u_01=u_01,
            u_10=u_10,
            u_11=u_11,
            grid_size=grid_size,
            box_size=box_size,
            backend=backend,
        )

    def test_initialization(self) -> None:
        """Test SU2Field initialization."""
        self.assertEqual(self.field.grid_size, 8)
        self.assertEqual(self.field.box_size, 2.0)
        self.assertIsNotNone(self.field.u_00)
        self.assertIsNotNone(self.field.u_01)
        self.assertIsNotNone(self.field.u_10)
        self.assertIsNotNone(self.field.u_11)

    def test_su2_validation(self) -> None:
        """Test SU(2) field validation."""
        # The field should be valid SU(2)
        self.assertTrue(self.field._is_su2_field())

    def test_get_matrix_at_point(self) -> None:
        """Test getting matrix at specific point."""
        matrix = self.field.get_matrix_at_point(4, 4, 4)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.iscomplexobj(matrix))

    def test_get_determinant(self) -> None:
        """Test determinant computation."""
        det = self.field.get_determinant()
        self.assertEqual(det.shape, (8, 8, 8))
        # Determinant should be close to 1 for SU(2) field
        self.assertTrue(np.allclose(det, 1.0, atol=1e-10))


class TestRadialProfile(unittest.TestCase):
    """Test RadialProfile class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False  # Force CPU mode for tests
        self.profile = RadialProfile("skyrmion", scale=1.0, backend=backend)

    def test_initialization(self) -> None:
        """Test RadialProfile initialization."""
        self.assertEqual(self.profile.profile_type, "skyrmion")
        self.assertEqual(self.profile.scale, 1.0)
        self.assertEqual(self.profile.center_value, np.pi)

    def test_skyrmion_profile(self) -> None:
        """Test skyrmion profile evaluation."""
        r = np.array([0.0, 1.0, 2.0])
        f_r = self.profile.evaluate(r)

        # At r=0, f should be π
        self.assertAlmostEqual(f_r[0], np.pi, places=10)
        # At r>0, f should be smaller
        self.assertLess(f_r[1], f_r[0])
        self.assertLess(f_r[2], f_r[1])

    def test_exponential_profile(self) -> None:
        """Test exponential profile."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False
        profile = RadialProfile("exponential", scale=1.0, backend=backend)
        r = np.array([0.0, 1.0, 2.0])
        f_r = profile.evaluate(r)

        # At r=0, f should be π
        self.assertAlmostEqual(f_r[0], np.pi, places=10)
        # Should decrease with r
        self.assertLess(f_r[1], f_r[0])

    def test_polynomial_profile(self) -> None:
        """Test polynomial profile."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False
        profile = RadialProfile("polynomial", scale=1.0, backend=backend)
        r = np.array([0.0, 1.0, 2.0])
        f_r = profile.evaluate(r)

        # At r=0, f should be π
        self.assertAlmostEqual(f_r[0], np.pi, places=10)
        # Should decrease with r
        self.assertLess(f_r[1], f_r[0])

    def test_derivative(self) -> None:
        """Test profile derivative."""
        r = np.array([0.0, 1.0, 2.0])
        dr = 0.01
        df_dr = self.profile.get_derivative(r, dr)

        # Derivative should be negative (decreasing function)
        self.assertTrue(np.all(df_dr < 0))


class TestSU2FieldBuilder(unittest.TestCase):
    """Test SU2FieldBuilder class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False  # Force CPU mode for tests
        self.builder = SU2FieldBuilder(8, 2.0, backend)

    def test_initialization(self) -> None:
        """Test builder initialization."""
        self.assertEqual(self.builder.grid_size, 8)
        self.assertEqual(self.builder.box_size, 2.0)
        self.assertEqual(self.builder.dx, 0.25)

    def test_coordinate_grids(self) -> None:
        """Test coordinate grid creation."""
        X, Y, Z = self.builder.X, self.builder.Y, self.builder.Z
        R = self.builder.R

        # Check shapes
        expected_shape = (8, 8, 8)
        self.assertEqual(X.shape, expected_shape)
        self.assertEqual(Y.shape, expected_shape)
        self.assertEqual(Z.shape, expected_shape)
        self.assertEqual(R.shape, expected_shape)

        # Check ranges
        self.assertAlmostEqual(X.min(), -1.0, places=10)
        self.assertAlmostEqual(X.max(), 1.0, places=10)

    def test_build_field(self) -> None:
        """Test field building."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False
        profile = RadialProfile("skyrmion", scale=1.0, backend=backend)

        # Simple field direction (z-axis)
        n_x = np.zeros_like(self.builder.X)
        n_y = np.zeros_like(self.builder.Y)
        n_z = np.ones_like(self.builder.Z)

        field = self.builder.build_field(n_x, n_y, n_z, profile)

        self.assertIsInstance(field, SU2Field)
        self.assertEqual(field.grid_size, 8)
        self.assertEqual(field.box_size, 2.0)

    def test_build_from_torus_config(self) -> None:
        """Test building from torus configuration."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False
        profile = RadialProfile("skyrmion", scale=1.0, backend=backend)

        # Create a simple torus configuration
        torus_config = Torus120Degrees(
            radius=1.0, thickness=0.2, backend=backend
        )

        field = self.builder.build_from_torus_config(torus_config, profile)

        self.assertIsInstance(field, SU2Field)
        self.assertEqual(field.grid_size, 8)
        self.assertEqual(field.box_size, 2.0)


class TestSU2FieldOperations(unittest.TestCase):
    """Test SU2FieldOperations class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False  # Force CPU mode for tests
        self.operations = SU2FieldOperations(0.25, backend)

        # Create a simple test field
        grid_size = 8
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        z = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)

        f_r = np.pi * np.exp(-R)
        cos_f = np.cos(f_r)
        sin_f = np.sin(f_r)

        u_00 = cos_f + 1j * sin_f
        u_01 = np.zeros_like(cos_f)
        u_10 = np.zeros_like(cos_f)
        u_11 = cos_f - 1j * sin_f

        self.field = SU2Field(
            u_00=u_00,
            u_01=u_01,
            u_10=u_10,
            u_11=u_11,
            grid_size=grid_size,
            box_size=2.0,
            backend=backend,
        )

    def test_initialization(self) -> None:
        """Test operations initialization."""
        self.assertEqual(self.operations.dx, 0.25)

    def test_compute_left_currents(self) -> None:
        """Test left current computation."""
        l_x, l_y, l_z = self.operations.compute_left_currents(self.field)

        # Check that currents are dictionaries with correct keys
        for current in [l_x, l_y, l_z]:
            self.assertIn("l_00", current)
            self.assertIn("l_01", current)
            self.assertIn("l_10", current)
            self.assertIn("l_11", current)

        # Check shapes
        expected_shape = (8, 8, 8)
        for current in [l_x, l_y, l_z]:
            for key in current:
                self.assertEqual(current[key].shape, expected_shape)

    def test_compute_commutators(self) -> None:
        """Test commutator computation."""
        l_x, l_y, l_z = self.operations.compute_left_currents(self.field)
        commutators = self.operations.compute_commutators(l_x, l_y, l_z)

        # Check that all commutators are present
        self.assertIn("xy", commutators)
        self.assertIn("yz", commutators)
        self.assertIn("zx", commutators)

        # Check commutator structure
        for comm in commutators.values():
            self.assertIn("comm_00", comm)
            self.assertIn("comm_01", comm)
            self.assertIn("comm_10", comm)
            self.assertIn("comm_11", comm)

    def test_compute_traces(self) -> None:
        """Test trace computation."""
        l_x, l_y, l_z = self.operations.compute_left_currents(self.field)
        commutators = self.operations.compute_commutators(l_x, l_y, l_z)
        traces = self.operations.compute_traces(l_x, l_y, l_z, commutators)

        # Check that traces are present
        self.assertIn("l_squared", traces)
        self.assertIn("comm_squared", traces)

        # Check shapes
        expected_shape = (8, 8, 8)
        self.assertEqual(traces["l_squared"].shape, expected_shape)
        self.assertEqual(traces["comm_squared"].shape, expected_shape)


class TestSU2FieldValidator(unittest.TestCase):
    """Test SU2FieldValidator class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False  # Force CPU mode for tests
        self.validator = SU2FieldValidator(backend=backend)

        # Create a simple test field
        grid_size = 8
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        z = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)

        f_r = np.pi * np.exp(-R)
        cos_f = np.cos(f_r)
        sin_f = np.sin(f_r)

        u_00 = cos_f + 1j * sin_f
        u_01 = np.zeros_like(cos_f)
        u_10 = np.zeros_like(cos_f)
        u_11 = cos_f - 1j * sin_f

        self.field = SU2Field(
            u_00=u_00,
            u_01=u_01,
            u_10=u_10,
            u_11=u_11,
            grid_size=grid_size,
            box_size=2.0,
            backend=backend,
        )

    def test_initialization(self) -> None:
        """Test validator initialization."""
        self.assertEqual(self.validator.tolerance, 1e-10)

    def test_validate_field(self) -> None:
        """Test field validation."""
        results = self.validator.validate_field(self.field)

        # Check that all validation checks are present
        self.assertIn("unitary", results)
        self.assertIn("determinant", results)
        self.assertIn("continuity", results)
        self.assertIn("boundary_conditions", results)

        # Check that results are boolean
        for key, value in results.items():
            self.assertIsInstance(value, bool)

    def test_check_unitarity(self) -> None:
        """Test unitarity check."""
        result = self.validator._check_unitarity(self.field)
        self.assertIsInstance(result, bool)

    def test_check_determinant(self) -> None:
        """Test determinant check."""
        result = self.validator._check_determinant(self.field)
        self.assertIsInstance(result, bool)

    def test_check_continuity(self) -> None:
        """Test continuity check."""
        result = self.validator._check_continuity(self.field)
        self.assertIsInstance(result, bool)

    def test_check_boundary_conditions(self) -> None:
        """Test boundary conditions check."""
        result = self.validator._check_boundary_conditions(self.field)
        self.assertIsInstance(result, bool)


class TestSU2Fields(unittest.TestCase):
    """Test main SU2Fields class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.su2_fields = SU2Fields(grid_size=8, box_size=2.0, use_cuda=False)

    def test_initialization(self) -> None:
        """Test SU2Fields initialization."""
        self.assertEqual(self.su2_fields.grid_size, 8)
        self.assertEqual(self.su2_fields.box_size, 2.0)
        self.assertEqual(self.su2_fields.dx, 0.25)
        self.assertIsNotNone(self.su2_fields.builder)
        self.assertIsNotNone(self.su2_fields.operations)
        self.assertIsNotNone(self.su2_fields.validator)

    def test_create_field_from_torus(self) -> None:
        """Test field creation from torus configuration."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False
        torus_config = Torus120Degrees(
            radius=1.0, thickness=0.2, backend=backend
        )

        field = self.su2_fields.create_field_from_torus(
            torus_config, profile_type="skyrmion", scale=1.0
        )

        self.assertIsInstance(field, SU2Field)
        self.assertEqual(field.grid_size, 8)
        self.assertEqual(field.box_size, 2.0)

    def test_compute_field_derivatives(self) -> None:
        """Test field derivative computation."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False
        torus_config = Torus120Degrees(
            radius=1.0, thickness=0.2, backend=backend
        )

        field = self.su2_fields.create_field_from_torus(torus_config)
        derivatives = self.su2_fields.compute_field_derivatives(field)

        # Check structure
        self.assertIn("left_currents", derivatives)
        self.assertIn("commutators", derivatives)
        self.assertIn("traces", derivatives)

        # Check left currents
        left_currents = derivatives["left_currents"]
        self.assertIn("x", left_currents)
        self.assertIn("y", left_currents)
        self.assertIn("z", left_currents)

    def test_validate_field(self) -> None:
        """Test field validation."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False
        torus_config = Torus120Degrees(
            radius=1.0, thickness=0.2, backend=backend
        )

        field = self.su2_fields.create_field_from_torus(torus_config)
        validation_results = self.su2_fields.validate_field(field)

        # Check that validation results are present
        self.assertIn("unitary", validation_results)
        self.assertIn("determinant", validation_results)
        self.assertIn("continuity", validation_results)
        self.assertIn("boundary_conditions", validation_results)

    def test_get_field_statistics(self) -> None:
        """Test field statistics."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False
        torus_config = Torus120Degrees(
            radius=1.0, thickness=0.2, backend=backend
        )

        field = self.su2_fields.create_field_from_torus(torus_config)
        stats = self.su2_fields.get_field_statistics(field)

        # Check that statistics are present
        self.assertIn("mean_determinant", stats)
        self.assertIn("std_determinant", stats)
        self.assertIn("min_determinant", stats)
        self.assertIn("max_determinant", stats)
        self.assertIn("field_norm_mean", stats)
        self.assertIn("field_norm_std", stats)

        # Check that statistics are numeric
        for key, value in stats.items():
            self.assertIsInstance(value, float)

    def test_cuda_integration(self) -> None:
        """Test CUDA integration."""
        # Test CUDA status
        cuda_status = self.su2_fields.get_cuda_status()
        self.assertIsInstance(cuda_status, str)

        # Test backend info
        backend_info = self.su2_fields.get_backend_info()
        self.assertIn("backend", backend_info)
        self.assertIn("cuda_status", backend_info)
        self.assertIn("cuda_available", backend_info)

        # Test CUDA availability
        cuda_available = self.su2_fields.is_cuda_available()
        self.assertIsInstance(cuda_available, bool)


if __name__ == "__main__":
    unittest.main()
