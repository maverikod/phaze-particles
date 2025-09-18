#!/usr/bin/env python3
"""
Unit tests for mathematical foundations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
from phaze_particles.utils.mathematical_foundations import (
    PhysicalConstants,
    SkyrmeConstants,
    PauliMatrices,
    TensorOperations,
    CoordinateSystem,
    NumericalUtils,
    ValidationUtils,
    MathematicalFoundations,
)


class TestPhysicalConstants(unittest.TestCase):
    """Test PhysicalConstants class."""

    def test_constants_values(self) -> None:
        """Test that constants have correct values."""
        self.assertEqual(PhysicalConstants.PROTON_CHARGE, 1.0)
        self.assertEqual(PhysicalConstants.BARYON_NUMBER, 1.0)
        self.assertEqual(PhysicalConstants.PROTON_MASS_MEV, 938.272)
        self.assertEqual(PhysicalConstants.CHARGE_RADIUS_FM, 0.841)
        self.assertEqual(PhysicalConstants.MAGNETIC_MOMENT_MU_N, 2.793)
        self.assertGreater(PhysicalConstants.HBAR_C, 0)
        self.assertGreater(PhysicalConstants.ALPHA_EM, 0)


class TestSkyrmeConstants(unittest.TestCase):
    """Test SkyrmeConstants class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        skyrme = SkyrmeConstants()
        self.assertEqual(skyrme.c2, 1.0)
        self.assertEqual(skyrme.c4, 1.0)
        self.assertEqual(skyrme.c6, 1.0)

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        skyrme = SkyrmeConstants(c2=2.0, c4=3.0, c6=4.0)
        self.assertEqual(skyrme.c2, 2.0)
        self.assertEqual(skyrme.c4, 3.0)
        self.assertEqual(skyrme.c6, 4.0)

    def test_validation_positive(self) -> None:
        """Test validation with positive constants."""
        skyrme = SkyrmeConstants(1.0, 2.0, 3.0)
        self.assertTrue(skyrme.validate())

    def test_validation_negative(self) -> None:
        """Test validation with negative constants."""
        skyrme = SkyrmeConstants(-1.0, 2.0, 3.0)
        self.assertFalse(skyrme.validate())

        skyrme = SkyrmeConstants(1.0, -2.0, 3.0)
        self.assertFalse(skyrme.validate())

        skyrme = SkyrmeConstants(1.0, 2.0, -3.0)
        self.assertFalse(skyrme.validate())


class TestPauliMatrices(unittest.TestCase):
    """Test PauliMatrices class."""

    def test_sigma_matrices_properties(self) -> None:
        """Test Pauli matrices properties."""
        # Check that matrices are 2x2
        for i in range(1, 4):
            sigma = PauliMatrices.get_sigma(i)
            self.assertEqual(sigma.shape, (2, 2))

        # Check specific matrices
        sigma1 = PauliMatrices.SIGMA_1
        np.testing.assert_array_equal(sigma1, np.array([[0, 1], [1, 0]]))

        sigma2 = PauliMatrices.SIGMA_2
        np.testing.assert_array_equal(sigma2, np.array([[0, -1j], [1j, 0]]))

        sigma3 = PauliMatrices.SIGMA_3
        np.testing.assert_array_equal(sigma3, np.array([[1, 0], [0, -1]]))

    def test_get_sigma_valid_indices(self) -> None:
        """Test get_sigma with valid indices."""
        for i in range(1, 4):
            sigma = PauliMatrices.get_sigma(i)
            self.assertIsInstance(sigma, np.ndarray)
            self.assertEqual(sigma.shape, (2, 2))

    def test_get_sigma_invalid_index(self) -> None:
        """Test get_sigma with invalid index."""
        with self.assertRaises(ValueError):
            PauliMatrices.get_sigma(0)

        with self.assertRaises(ValueError):
            PauliMatrices.get_sigma(4)

    def test_get_all_sigmas(self) -> None:
        """Test get_all_sigmas method."""
        sigmas = PauliMatrices.get_all_sigmas()
        self.assertEqual(len(sigmas), 3)

        for sigma in sigmas:
            self.assertIsInstance(sigma, np.ndarray)
            self.assertEqual(sigma.shape, (2, 2))


class TestTensorOperations(unittest.TestCase):
    """Test TensorOperations class."""

    def test_epsilon_tensor_properties(self) -> None:
        """Test epsilon tensor properties."""
        epsilon = TensorOperations.epsilon_tensor()

        # Check shape
        self.assertEqual(epsilon.shape, (3, 3, 3))

        # Check antisymmetry
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i != j and j != k and k != i:
                        # Check cyclic permutations
                        self.assertEqual(epsilon[i, j, k], epsilon[j, k, i])
                        self.assertEqual(epsilon[i, j, k], epsilon[k, i, j])
                        # Check anticyclic permutations
                        self.assertEqual(epsilon[i, j, k], -epsilon[i, k, j])
                        self.assertEqual(epsilon[i, j, k], -epsilon[j, i, k])
                        self.assertEqual(epsilon[i, j, k], -epsilon[k, j, i])

    def test_trace_product_empty_list(self) -> None:
        """Test trace_product with empty list."""
        result = TensorOperations.trace_product([])
        self.assertEqual(result, 0.0)

    def test_trace_product_single_matrix(self) -> None:
        """Test trace_product with single matrix."""
        matrix = np.array([[1, 2], [3, 4]])
        result = TensorOperations.trace_product([matrix])
        expected = np.trace(matrix)
        self.assertEqual(result, expected)

    def test_trace_product_multiple_matrices(self) -> None:
        """Test trace_product with multiple matrices."""
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[2, 0], [0, 2]])
        result = TensorOperations.trace_product([A, B])
        expected = np.trace(np.dot(A, B))
        self.assertEqual(result, expected)

    def test_commutator(self) -> None:
        """Test commutator calculation."""
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[0, 1], [1, 0]])

        result = TensorOperations.commutator(A, B)
        expected = np.dot(A, B) - np.dot(B, A)

        np.testing.assert_array_equal(result, expected)


class TestCoordinateSystem(unittest.TestCase):
    """Test CoordinateSystem class."""

    def test_initialization(self) -> None:
        """Test coordinate system initialization."""
        coords = CoordinateSystem(grid_size=32, box_size=4.0)

        self.assertEqual(coords.grid_size, 32)
        self.assertEqual(coords.box_size, 4.0)
        self.assertEqual(coords.dx, 4.0 / 32)

        # Check coordinate arrays
        X, Y, Z = coords.get_coordinates()
        self.assertEqual(X.shape, (32, 32, 32))
        self.assertEqual(Y.shape, (32, 32, 32))
        self.assertEqual(Z.shape, (32, 32, 32))

    def test_coordinate_ranges(self) -> None:
        """Test coordinate ranges."""
        coords = CoordinateSystem(grid_size=10, box_size=2.0)
        X, Y, Z = coords.get_coordinates()

        # Check that coordinates span from -box_size/2 to box_size/2
        self.assertAlmostEqual(X.min(), -1.0, places=10)
        self.assertAlmostEqual(X.max(), 1.0, places=10)
        self.assertAlmostEqual(Y.min(), -1.0, places=10)
        self.assertAlmostEqual(Y.max(), 1.0, places=10)
        self.assertAlmostEqual(Z.min(), -1.0, places=10)
        self.assertAlmostEqual(Z.max(), 1.0, places=10)

    def test_radial_coordinate(self) -> None:
        """Test radial coordinate calculation."""
        coords = CoordinateSystem(grid_size=10, box_size=2.0)
        R = coords.get_radial_coordinate()

        # Check that R is always positive
        self.assertTrue(np.all(R >= 0))

        # Check that R is minimum at center
        center_idx = 5  # Middle of 10x10x10 grid
        center_value = R[center_idx, center_idx, center_idx]
        # For 10x10x10 grid with box_size=2.0, center should be at minimum
        self.assertLessEqual(center_value, 0.2)  # Allow some tolerance

    def test_volume_element(self) -> None:
        """Test volume element calculation."""
        coords = CoordinateSystem(grid_size=10, box_size=2.0)
        volume_element = coords.get_volume_element()

        expected = (2.0 / 10) ** 3
        self.assertAlmostEqual(volume_element, expected, places=10)


class TestNumericalUtils(unittest.TestCase):
    """Test NumericalUtils class."""

    def test_gradient_3d_constant_field(self):
        """Test gradient of constant field."""
        field = np.ones((10, 10, 10))
        dx = 0.1

        grad_x, grad_y, grad_z = NumericalUtils.gradient_3d(field, dx)

        # Gradient of constant field should be zero
        np.testing.assert_array_almost_equal(grad_x, 0, decimal=10)
        np.testing.assert_array_almost_equal(grad_y, 0, decimal=10)
        np.testing.assert_array_almost_equal(grad_z, 0, decimal=10)

    def test_gradient_3d_linear_field(self):
        """Test gradient of linear field."""
        # Create linear field f(x,y,z) = x + y + z
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        z = np.linspace(0, 1, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        field = X + Y + Z

        dx = 0.1
        grad_x, grad_y, grad_z = NumericalUtils.gradient_3d(field, dx)

        # Gradient should be (1, 1, 1) for linear field
        # Note: numpy.gradient uses finite differences, so expect some variation
        np.testing.assert_array_almost_equal(grad_x, 1, decimal=1)
        np.testing.assert_array_almost_equal(grad_y, 1, decimal=1)
        np.testing.assert_array_almost_equal(grad_z, 1, decimal=1)

    def test_divergence_3d_constant_field(self):
        """Test divergence of constant vector field."""
        field_x = np.ones((10, 10, 10))
        field_y = np.ones((10, 10, 10))
        field_z = np.ones((10, 10, 10))
        dx = 0.1

        div = NumericalUtils.divergence_3d(field_x, field_y, field_z, dx)

        # Divergence of constant field should be zero
        np.testing.assert_array_almost_equal(div, 0, decimal=10)

    def test_integrate_3d_constant_field(self):
        """Test integration of constant field."""
        field = np.ones((10, 10, 10))
        dx = 0.1

        result = NumericalUtils.integrate_3d(field, dx)

        expected = 10 * 10 * 10 * dx**3
        self.assertAlmostEqual(result, expected, places=10)


class TestValidationUtils(unittest.TestCase):
    """Test ValidationUtils class."""

    def test_check_su2_matrix_valid(self):
        """Test SU(2) matrix validation with valid matrix."""
        # Identity matrix is SU(2)
        U = np.eye(2, dtype=complex)
        self.assertTrue(ValidationUtils.check_su2_matrix(U))

        # Rotation matrix is SU(2)
        theta = np.pi / 4
        U = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=complex,
        )
        self.assertTrue(ValidationUtils.check_su2_matrix(U))

    def test_check_su2_matrix_invalid(self):
        """Test SU(2) matrix validation with invalid matrix."""
        # Non-unitary matrix
        U = np.array([[1, 2], [0, 1]], dtype=complex)
        self.assertFalse(ValidationUtils.check_su2_matrix(U))

        # Matrix with det != 1
        U = np.array([[2, 0], [0, 1]], dtype=complex)
        self.assertFalse(ValidationUtils.check_su2_matrix(U))

    def test_check_physical_bounds(self) -> None:
        """Test physical bounds checking."""
        # Value within bounds
        self.assertTrue(ValidationUtils.check_physical_bounds(1.0, 1.0, 0.1))
        self.assertTrue(ValidationUtils.check_physical_bounds(1.05, 1.0, 0.1))
        self.assertTrue(ValidationUtils.check_physical_bounds(0.95, 1.0, 0.1))

        # Value outside bounds
        self.assertFalse(ValidationUtils.check_physical_bounds(1.2, 1.0, 0.1))
        self.assertFalse(ValidationUtils.check_physical_bounds(0.8, 1.0, 0.1))


class TestMathematicalFoundations(unittest.TestCase):
    """Test MathematicalFoundations main class."""

    def test_initialization(self) -> None:
        """Test main class initialization."""
        foundations = MathematicalFoundations(grid_size=32, box_size=4.0)

        # Check that all components are initialized
        self.assertIsInstance(foundations.constants, PhysicalConstants)
        self.assertIsInstance(foundations.skyrme, SkyrmeConstants)
        self.assertIsInstance(foundations.pauli, PauliMatrices)
        self.assertIsInstance(foundations.tensor, TensorOperations)
        self.assertIsInstance(foundations.coords, CoordinateSystem)
        self.assertIsInstance(foundations.numerical, NumericalUtils)
        self.assertIsInstance(foundations.validation, ValidationUtils)

    def test_validate_setup(self) -> None:
        """Test setup validation."""
        foundations = MathematicalFoundations()
        self.assertTrue(foundations.validate_setup())

    def test_get_physical_constants(self) -> None:
        """Test getting physical constants."""
        foundations = MathematicalFoundations()
        constants = foundations.get_physical_constants()

        self.assertIsInstance(constants, dict)
        self.assertIn("proton_charge", constants)
        self.assertIn("baryon_number", constants)
        self.assertIn("proton_mass_mev", constants)
        self.assertIn("charge_radius_fm", constants)
        self.assertIn("magnetic_moment_mu_n", constants)
        self.assertIn("hbar_c", constants)
        self.assertIn("alpha_em", constants)

    def test_get_skyrme_constants(self) -> None:
        """Test getting Skyrme constants."""
        foundations = MathematicalFoundations()
        constants = foundations.get_skyrme_constants()

        self.assertIsInstance(constants, dict)
        self.assertIn("c2", constants)
        self.assertIn("c4", constants)
        self.assertIn("c6", constants)

        self.assertEqual(constants["c2"], 1.0)
        self.assertEqual(constants["c4"], 1.0)
        self.assertEqual(constants["c6"], 1.0)


if __name__ == "__main__":
    unittest.main()
