#!/usr/bin/env python3
"""
Tests for torus geometries.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
from phaze_particles.utils.torus_geometries import (
    TorusConfiguration,
    TorusParameters,
    Torus120Degrees,
    TorusClover,
    TorusCartesian,
    TorusGeometryManager,
    TorusGeometries,
)


class TestTorusParameters(unittest.TestCase):
    """Test TorusParameters dataclass."""

    def test_torus_parameters_creation(self) -> None:
        """Test creation of TorusParameters."""
        params = TorusParameters(
            center=(0.0, 0.0, 0.0),
            radius=1.0,
            axis=(0.0, 0.0, 1.0),
            thickness=0.2,
            strength=1.0,
        )

        self.assertEqual(params.center, (0.0, 0.0, 0.0))
        self.assertEqual(params.radius, 1.0)
        self.assertEqual(params.axis, (0.0, 0.0, 1.0))
        self.assertEqual(params.thickness, 0.2)
        self.assertEqual(params.strength, 1.0)


class TestTorus120Degrees(unittest.TestCase):
    """Test 120° torus configuration."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False  # Force CPU mode for tests
        self.config = Torus120Degrees(
            radius=1.0, thickness=0.2, strength=1.0, backend=backend
        )

    def test_initialization(self) -> None:
        """Test initialization of 120° configuration."""
        self.assertEqual(len(self.config.tori), 3)
        self.assertEqual(self.config.radius, 1.0)
        self.assertEqual(self.config.thickness, 0.2)
        self.assertEqual(self.config.strength, 1.0)

    def test_torus_parameters(self) -> None:
        """Test torus parameters."""
        # Check first torus (z-axis)
        torus1 = self.config.tori[0]
        self.assertEqual(torus1.center, (0.0, 0.0, 0.0))
        self.assertEqual(torus1.radius, 1.0)
        self.assertEqual(torus1.axis, (0.0, 0.0, 1.0))

        # Check second torus (120° rotation)
        torus2 = self.config.tori[1]
        self.assertEqual(torus2.center, (0.0, 0.0, 0.0))
        self.assertEqual(torus2.radius, 1.0)
        # Check that axis is normalized
        axis_norm = np.sqrt(sum(a**2 for a in torus2.axis))
        self.assertAlmostEqual(axis_norm, 1.0, places=10)

    def test_field_direction(self) -> None:
        """Test field direction calculation."""
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-2, 2, 10)
        z = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        n_x, n_y, n_z = self.config.get_field_direction(X, Y, Z)

        # Check shapes
        self.assertEqual(n_x.shape, X.shape)
        self.assertEqual(n_y.shape, Y.shape)
        self.assertEqual(n_z.shape, Z.shape)

        # Check normalization (allow for small deviations near zero)
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        # Check that norm is either close to 1.0 or close to 0.0
        valid_norm = np.logical_or(
            np.abs(norm - 1.0) < 1e-10, np.abs(norm) < 1e-10
        )
        self.assertTrue(np.all(valid_norm), "Field normalization failed")

    def test_distance_to_torus(self) -> None:
        """Test distance calculation to torus."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        torus = self.config.tori[0]  # z-axis torus
        distance = self.config._distance_to_torus(X, Y, Z, torus)

        # Check that distance is non-negative
        self.assertTrue(np.all(distance >= 0))


class TestTorusClover(unittest.TestCase):
    """Test clover torus configuration."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False  # Force CPU mode for tests
        self.config = TorusClover(
            radius=1.0, thickness=0.2, strength=1.0, backend=backend
        )

    def test_initialization(self) -> None:
        """Test initialization of clover configuration."""
        self.assertEqual(len(self.config.tori), 3)
        self.assertEqual(self.config.radius, 1.0)
        self.assertEqual(self.config.thickness, 0.2)
        self.assertEqual(self.config.strength, 1.0)

    def test_torus_parameters(self) -> None:
        """Test torus parameters."""
        # Check first torus (x-axis)
        torus1 = self.config.tori[0]
        self.assertEqual(torus1.axis, (1.0, 0.0, 0.0))

        # Check second torus (y-axis)
        torus2 = self.config.tori[1]
        self.assertEqual(torus2.axis, (0.0, 1.0, 0.0))

        # Check third torus (diagonal)
        torus3 = self.config.tori[2]
        expected_axis = (1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0)
        np.testing.assert_array_almost_equal(torus3.axis, expected_axis)

    def test_field_direction(self) -> None:
        """Test field direction calculation."""
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-2, 2, 10)
        z = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        n_x, n_y, n_z = self.config.get_field_direction(X, Y, Z)

        # Check shapes
        self.assertEqual(n_x.shape, X.shape)
        self.assertEqual(n_y.shape, Y.shape)
        self.assertEqual(n_z.shape, Z.shape)

        # Check normalization (allow for small deviations near zero)
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        # Check that norm is either close to 1.0 or close to 0.0
        valid_norm = np.logical_or(
            np.abs(norm - 1.0) < 1e-10, np.abs(norm) < 1e-10
        )
        self.assertTrue(np.all(valid_norm), "Field normalization failed")


class TestTorusCartesian(unittest.TestCase):
    """Test cartesian torus configuration."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False  # Force CPU mode for tests
        self.config = TorusCartesian(
            radius=1.0, thickness=0.2, strength=1.0, backend=backend
        )

    def test_initialization(self) -> None:
        """Test initialization of cartesian configuration."""
        self.assertEqual(len(self.config.tori), 3)
        self.assertEqual(self.config.radius, 1.0)
        self.assertEqual(self.config.thickness, 0.2)
        self.assertEqual(self.config.strength, 1.0)

    def test_torus_parameters(self) -> None:
        """Test torus parameters."""
        # Check all three tori along axes
        expected_axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

        for i, expected_axis in enumerate(expected_axes):
            torus = self.config.tori[i]
            self.assertEqual(torus.axis, expected_axis)

    def test_field_direction(self) -> None:
        """Test field direction calculation."""
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-2, 2, 10)
        z = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        n_x, n_y, n_z = self.config.get_field_direction(X, Y, Z)

        # Check shapes
        self.assertEqual(n_x.shape, X.shape)
        self.assertEqual(n_y.shape, Y.shape)
        self.assertEqual(n_z.shape, Z.shape)

        # Check normalization (allow for small deviations near zero)
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        # Check that norm is either close to 1.0 or close to 0.0
        valid_norm = np.logical_or(
            np.abs(norm - 1.0) < 1e-10, np.abs(norm) < 1e-10
        )
        self.assertTrue(np.all(valid_norm), "Field normalization failed")


class TestTorusGeometryManager(unittest.TestCase):
    """Test torus geometry manager."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend

        backend = ArrayBackend()
        backend._use_cuda = False  # Force CPU mode for tests
        self.manager = TorusGeometryManager(backend=backend)

    def test_available_configurations(self) -> None:
        """Test available configurations."""
        configs = self.manager.get_available_configurations()
        expected_configs = [
            TorusConfiguration.CONFIG_120_DEG,
            TorusConfiguration.CONFIG_CLOVER,
            TorusConfiguration.CONFIG_CARTESIAN,
        ]

        self.assertEqual(len(configs), 3)
        for config in expected_configs:
            self.assertIn(config, configs)

    def test_create_configuration(self) -> None:
        """Test configuration creation."""
        # Test 120° configuration
        config_120 = self.manager.create_configuration(
            TorusConfiguration.CONFIG_120_DEG
        )
        self.assertIsInstance(config_120, Torus120Degrees)

        # Test clover configuration
        config_clover = self.manager.create_configuration(
            TorusConfiguration.CONFIG_CLOVER
        )
        self.assertIsInstance(config_clover, TorusClover)

        # Test cartesian configuration
        config_cartesian = self.manager.create_configuration(
            TorusConfiguration.CONFIG_CARTESIAN
        )
        self.assertIsInstance(config_cartesian, TorusCartesian)

    def test_configuration_info(self) -> None:
        """Test configuration information."""
        info_120 = self.manager.get_configuration_info(
            TorusConfiguration.CONFIG_120_DEG
        )
        self.assertEqual(info_120["name"], "120° Configuration")
        self.assertEqual(info_120["symmetry_group"], "C₃")
        self.assertEqual(info_120["num_tori"], 3)

        info_clover = self.manager.get_configuration_info(
            TorusConfiguration.CONFIG_CLOVER
        )
        self.assertEqual(info_clover["name"], "Clover Configuration")
        self.assertEqual(info_clover["symmetry_group"], "C₃")

        info_cartesian = self.manager.get_configuration_info(
            TorusConfiguration.CONFIG_CARTESIAN
        )
        self.assertEqual(info_cartesian["name"], "Cartesian Configuration")
        self.assertEqual(info_cartesian["symmetry_group"], "D₄")

    def test_validate_configuration(self) -> None:
        """Test configuration validation."""
        # Valid configuration
        config = self.manager.create_configuration(
            TorusConfiguration.CONFIG_120_DEG
        )
        self.assertTrue(self.manager.validate_configuration(config))

        # Invalid configuration (no tori attribute)
        invalid_config = object()
        self.assertFalse(self.manager.validate_configuration(invalid_config))


class TestTorusGeometries(unittest.TestCase):
    """Test main torus geometries class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.geometries = TorusGeometries(
            grid_size=16, box_size=2.0, use_cuda=False
        )

    def test_initialization(self) -> None:
        """Test initialization."""
        self.assertEqual(self.geometries.grid_size, 16)
        self.assertEqual(self.geometries.box_size, 2.0)
        self.assertIsNotNone(self.geometries.manager)

    def test_coordinate_grids(self) -> None:
        """Test coordinate grid creation."""
        X, Y, Z = self.geometries.X, self.geometries.Y, self.geometries.Z

        # Check shapes
        expected_shape = (16, 16, 16)
        self.assertEqual(X.shape, expected_shape)
        self.assertEqual(Y.shape, expected_shape)
        self.assertEqual(Z.shape, expected_shape)

        # Check ranges
        self.assertAlmostEqual(X.min(), -1.0, places=10)
        self.assertAlmostEqual(X.max(), 1.0, places=10)
        self.assertAlmostEqual(Y.min(), -1.0, places=10)
        self.assertAlmostEqual(Y.max(), 1.0, places=10)
        self.assertAlmostEqual(Z.min(), -1.0, places=10)
        self.assertAlmostEqual(Z.max(), 1.0, places=10)

    def test_create_field_direction_120(self) -> None:
        """Test field direction creation for 120° configuration."""
        n_x, n_y, n_z = self.geometries.create_field_direction(
            TorusConfiguration.CONFIG_120_DEG
        )

        # Check shapes
        expected_shape = (16, 16, 16)
        self.assertEqual(n_x.shape, expected_shape)
        self.assertEqual(n_y.shape, expected_shape)
        self.assertEqual(n_z.shape, expected_shape)

        # Check normalization (allow for small deviations near zero)
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        # Check that norm is either close to 1.0 or close to 0.0
        valid_norm = np.logical_or(
            np.abs(norm - 1.0) < 1e-10, np.abs(norm) < 1e-10
        )
        self.assertTrue(np.all(valid_norm), "Field normalization failed")

    def test_create_field_direction_clover(self) -> None:
        """Test field direction creation for clover configuration."""
        n_x, n_y, n_z = self.geometries.create_field_direction(
            TorusConfiguration.CONFIG_CLOVER
        )

        # Check shapes
        expected_shape = (16, 16, 16)
        self.assertEqual(n_x.shape, expected_shape)
        self.assertEqual(n_y.shape, expected_shape)
        self.assertEqual(n_z.shape, expected_shape)

        # Check normalization (allow for small deviations near zero)
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        # Check that norm is either close to 1.0 or close to 0.0
        valid_norm = np.logical_or(
            np.abs(norm - 1.0) < 1e-10, np.abs(norm) < 1e-10
        )
        self.assertTrue(np.all(valid_norm), "Field normalization failed")

    def test_create_field_direction_cartesian(self) -> None:
        """Test field direction creation for cartesian configuration."""
        n_x, n_y, n_z = self.geometries.create_field_direction(
            TorusConfiguration.CONFIG_CARTESIAN
        )

        # Check shapes
        expected_shape = (16, 16, 16)
        self.assertEqual(n_x.shape, expected_shape)
        self.assertEqual(n_y.shape, expected_shape)
        self.assertEqual(n_z.shape, expected_shape)

        # Check normalization (allow for small deviations near zero)
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        # Check that norm is either close to 1.0 or close to 0.0
        valid_norm = np.logical_or(
            np.abs(norm - 1.0) < 1e-10, np.abs(norm) < 1e-10
        )
        self.assertTrue(np.all(valid_norm), "Field normalization failed")

    def test_list_available_configurations(self) -> None:
        """Test listing available configurations."""
        configs = self.geometries.list_available_configurations()

        self.assertEqual(len(configs), 3)

        # Check that all configurations are present
        config_types = [config["type"] for config in configs]
        expected_types = [
            TorusConfiguration.CONFIG_120_DEG,
            TorusConfiguration.CONFIG_CLOVER,
            TorusConfiguration.CONFIG_CARTESIAN,
        ]

        for expected_type in expected_types:
            self.assertIn(expected_type, config_types)

    def test_cuda_integration(self) -> None:
        """Test CUDA integration."""
        # Test CUDA status
        cuda_status = self.geometries.get_cuda_status()
        self.assertIsInstance(cuda_status, str)

        # Test backend info
        backend_info = self.geometries.get_backend_info()
        self.assertIn("backend", backend_info)
        self.assertIn("cuda_status", backend_info)
        self.assertIn("cuda_available", backend_info)

        # Test CUDA availability
        cuda_available = self.geometries.is_cuda_available()
        self.assertIsInstance(cuda_available, bool)


if __name__ == "__main__":
    unittest.main()
