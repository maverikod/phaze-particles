#!/usr/bin/env python3
"""
Tests for numerical methods module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch

from phaze_particles.utils.numerical_methods import (
    RelaxationMethod,
    RelaxationConfig,
    ConstraintConfig,
    SU2Projection,
    GradientDescent,
    LBFGSOptimizer,
    AdamOptimizer,
    ConstraintController,
    RelaxationSolver,
    NumericalMethods,
)
from phaze_particles.utils.mathematical_foundations import ArrayBackend


class TestSU2Projection(unittest.TestCase):
    """Test SU(2) projection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.projection = SU2Projection()

    def test_project_to_su2_unitary_matrix(self):
        """Test projection of unitary matrix."""
        # Create a unitary matrix
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        result = self.projection.project_to_su2(U)

        # Should remain unitary
        self.assertTrue(self.projection.validate_su2(result))

    def test_project_to_su2_non_unitary_matrix(self):
        """Test projection of non-unitary matrix."""
        # Create a non-unitary matrix
        U = np.array([[2, 1], [0, 1]], dtype=complex)
        result = self.projection.project_to_su2(U)

        # Should become unitary
        self.assertTrue(self.projection.validate_su2(result))

    def test_validate_su2_valid_matrix(self):
        """Test validation of valid SU(2) matrix."""
        # Identity matrix
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        self.assertTrue(self.projection.validate_su2(U))

    def test_validate_su2_invalid_matrix(self):
        """Test validation of invalid SU(2) matrix."""
        # Non-unitary matrix
        U = np.array([[2, 1], [0, 1]], dtype=complex)
        self.assertFalse(self.projection.validate_su2(U))

    def test_static_methods(self):
        """Test static methods for backward compatibility."""
        U = np.array([[1, 0], [0, 1]], dtype=complex)

        # Test static projection
        result = SU2Projection.project_to_su2_static(U)
        self.assertTrue(SU2Projection.validate_su2_static(result))

        # Test static validation
        self.assertTrue(SU2Projection.validate_su2_static(U))


class TestGradientDescent(unittest.TestCase):
    """Test gradient descent optimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RelaxationConfig(
            method=RelaxationMethod.GRADIENT_DESCENT, step_size=0.01, momentum=0.9
        )
        self.optimizer = GradientDescent(self.config)

    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.step_size, 0.01)
        self.assertEqual(self.optimizer.momentum, 0.9)
        self.assertIsNone(self.optimizer.velocity)

    def test_step_single_matrix(self):
        """Test single optimization step with matrix."""
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        gradient = np.array([[0.1, 0], [0, 0.1]], dtype=complex)

        result = self.optimizer.step(U, gradient)

        # Should return a matrix of same shape
        self.assertEqual(result.shape, U.shape)
        # Should be SU(2)
        self.assertTrue(self.optimizer.su2_projection.validate_su2(result))

    def test_step_3d_field(self):
        """Test single optimization step with 3D field."""
        U = np.zeros((2, 2, 2, 2, 2), dtype=complex)
        gradient = np.zeros((2, 2, 2, 2, 2), dtype=complex)

        # Initialize with identity matrices
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i, j, k] = np.eye(2, dtype=complex)

        result = self.optimizer.step(U, gradient)

        # Should return field of same shape
        self.assertEqual(result.shape, U.shape)

    def test_reset(self):
        """Test optimizer reset."""
        # Initialize velocity
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        gradient = np.array([[0.1, 0], [0, 0.1]], dtype=complex)
        self.optimizer.step(U, gradient)

        # Reset
        self.optimizer.reset()
        self.assertIsNone(self.optimizer.velocity)


class TestLBFGSOptimizer(unittest.TestCase):
    """Test L-BFGS optimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RelaxationConfig(method=RelaxationMethod.LBFGS, step_size=0.01)
        self.optimizer = LBFGSOptimizer(self.config)

    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.memory_size, 10)
        self.assertEqual(len(self.optimizer.s_history), 0)
        self.assertEqual(len(self.optimizer.y_history), 0)
        self.assertEqual(len(self.optimizer.rho_history), 0)

    def test_step(self):
        """Test single optimization step."""
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        gradient = np.array([[0.1, 0], [0, 0.1]], dtype=complex)

        result = self.optimizer.step(U, gradient)

        # Should return a matrix of same shape
        self.assertEqual(result.shape, U.shape)

    def test_reset(self):
        """Test optimizer reset."""
        # Add some history
        self.optimizer.s_history.append(np.array([1, 2]))
        self.optimizer.y_history.append(np.array([3, 4]))
        self.optimizer.rho_history.append(0.5)

        # Reset
        self.optimizer.reset()
        self.assertEqual(len(self.optimizer.s_history), 0)
        self.assertEqual(len(self.optimizer.y_history), 0)
        self.assertEqual(len(self.optimizer.rho_history), 0)


class TestAdamOptimizer(unittest.TestCase):
    """Test Adam optimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RelaxationConfig(
            method=RelaxationMethod.ADAM,
            step_size=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        self.optimizer = AdamOptimizer(self.config)

    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.beta1, 0.9)
        self.assertEqual(self.optimizer.beta2, 0.999)
        self.assertEqual(self.optimizer.epsilon, 1e-8)
        self.assertIsNone(self.optimizer.m)
        self.assertIsNone(self.optimizer.v)
        self.assertEqual(self.optimizer.t, 0)

    def test_step(self):
        """Test single optimization step."""
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        gradient = np.array([[0.1, 0], [0, 0.1]], dtype=complex)

        result = self.optimizer.step(U, gradient)

        # Should return a matrix of same shape
        self.assertEqual(result.shape, U.shape)
        # Should initialize moments
        self.assertIsNotNone(self.optimizer.m)
        self.assertIsNotNone(self.optimizer.v)
        self.assertEqual(self.optimizer.t, 1)

    def test_reset(self):
        """Test optimizer reset."""
        # Initialize moments
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        gradient = np.array([[0.1, 0], [0, 0.1]], dtype=complex)
        self.optimizer.step(U, gradient)

        # Reset
        self.optimizer.reset()
        self.assertIsNone(self.optimizer.m)
        self.assertIsNone(self.optimizer.v)
        self.assertEqual(self.optimizer.t, 0)


class TestConstraintController(unittest.TestCase):
    """Test constraint controller."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConstraintConfig(
            lambda_B=1000.0,
            lambda_Q=1000.0,
            lambda_virial=1000.0,
            tolerance_B=0.02,
            tolerance_Q=1e-6,
            tolerance_virial=0.01,
        )
        self.controller = ConstraintController(self.config)

    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.lambda_B, 1000.0)
        self.assertEqual(self.controller.lambda_Q, 1000.0)
        self.assertEqual(self.controller.lambda_virial, 1000.0)

    def test_compute_constraint_penalty(self):
        """Test constraint penalty computation."""
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        penalty = self.controller.compute_constraint_penalty(
            U, baryon_number=1.0, electric_charge=1.0, energy_balance=0.5
        )

        # Should be zero for perfect constraints
        self.assertEqual(penalty, 0.0)

    def test_compute_constraint_penalty_violations(self):
        """Test constraint penalty with violations."""
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        penalty = self.controller.compute_constraint_penalty(
            U, baryon_number=1.1, electric_charge=1.1, energy_balance=0.6
        )

        # Should be positive for violations
        self.assertGreater(penalty, 0.0)

    def test_check_constraints_satisfied(self):
        """Test constraint checking when satisfied."""
        result = self.controller.check_constraints(
            baryon_number=1.0, electric_charge=1.0, energy_balance=0.5
        )

        self.assertTrue(result["baryon_number"])
        self.assertTrue(result["electric_charge"])
        self.assertTrue(result["energy_balance"])

    def test_check_constraints_violated(self):
        """Test constraint checking when violated."""
        result = self.controller.check_constraints(
            baryon_number=1.1, electric_charge=1.1, energy_balance=0.6
        )

        self.assertFalse(result["baryon_number"])
        self.assertFalse(result["electric_charge"])
        self.assertFalse(result["energy_balance"])


class TestRelaxationSolver(unittest.TestCase):
    """Test relaxation solver."""

    def setUp(self):
        """Set up test fixtures."""
        self.relaxation_config = RelaxationConfig(
            method=RelaxationMethod.GRADIENT_DESCENT,
            max_iterations=10,
            convergence_tol=1e-6,
        )
        self.constraint_config = ConstraintConfig()
        self.solver = RelaxationSolver(self.relaxation_config, self.constraint_config)

    def test_initialization(self):
        """Test solver initialization."""
        self.assertEqual(self.solver.config, self.relaxation_config)
        self.assertIsInstance(self.solver.optimizer, GradientDescent)
        self.assertIsNotNone(self.solver.progress_tracker)

    def test_solve_with_mock_functions(self):
        """Test solve method with mock functions."""
        # Mock functions
        U_init = np.array([[1, 0], [0, 1]], dtype=complex)

        def energy_function(U):
            return 1.0

        def gradient_function(U):
            return np.zeros_like(U)

        def baryon_function(U):
            return 1.0

        def charge_function(U):
            return 1.0

        def energy_balance_function(U):
            return 0.5

        constraint_functions = {
            "baryon_number": baryon_function,
            "electric_charge": charge_function,
            "energy_balance": energy_balance_function,
        }

        result = self.solver.solve(
            U_init, energy_function, gradient_function, constraint_functions
        )

        # Check result structure
        self.assertIn("solution", result)
        self.assertIn("energy_history", result)
        self.assertIn("constraint_history", result)
        self.assertIn("iterations", result)
        self.assertIn("converged", result)
        self.assertIn("execution_time", result)
        self.assertIn("final_energy", result)
        self.assertIn("final_constraints", result)

    def test_reset(self):
        """Test solver reset."""
        # This should not raise an exception
        self.solver.reset()


class TestNumericalMethods(unittest.TestCase):
    """Test numerical methods class."""

    def setUp(self):
        """Set up test fixtures."""
        self.numerical = NumericalMethods(grid_size=8, box_size=2.0)

    def test_initialization(self):
        """Test numerical methods initialization."""
        self.assertEqual(self.numerical.grid_size, 8)
        self.assertEqual(self.numerical.box_size, 2.0)
        self.assertEqual(self.numerical.dx, 0.25)

    def test_compute_gradient(self):
        """Test gradient computation."""
        # Create a simple field
        field = self.numerical.X**2 + self.numerical.Y**2 + self.numerical.Z**2

        grad_x, grad_y, grad_z = self.numerical.compute_gradient(field)

        # Check shapes
        self.assertEqual(grad_x.shape, field.shape)
        self.assertEqual(grad_y.shape, field.shape)
        self.assertEqual(grad_z.shape, field.shape)

    def test_compute_divergence(self):
        """Test divergence computation."""
        # Create simple vector field
        field_x = self.numerical.X
        field_y = self.numerical.Y
        field_z = self.numerical.Z

        divergence = self.numerical.compute_divergence(field_x, field_y, field_z)

        # Check shape
        self.assertEqual(divergence.shape, field_x.shape)

    def test_integrate_3d(self):
        """Test 3D integration."""
        # Create a constant field
        field = np.ones((8, 8, 8))

        result = self.numerical.integrate_3d(field)

        # Should be volume * constant
        expected = 8 * 8 * 8 * 0.25**3
        self.assertAlmostEqual(result, expected, places=10)

    def test_create_initial_field_tanh(self):
        """Test initial field creation with tanh profile."""
        U = self.numerical.create_initial_field("tanh")

        # Check shape
        self.assertEqual(U.shape, (8, 8, 8, 2, 2))
        # Check that it's complex
        self.assertTrue(np.iscomplexobj(U))

    def test_create_initial_field_exp(self):
        """Test initial field creation with exp profile."""
        U = self.numerical.create_initial_field("exp")

        # Check shape
        self.assertEqual(U.shape, (8, 8, 8, 2, 2))
        # Check that it's complex
        self.assertTrue(np.iscomplexobj(U))

    def test_create_initial_field_gaussian(self):
        """Test initial field creation with gaussian profile."""
        U = self.numerical.create_initial_field("gaussian")

        # Check shape
        self.assertEqual(U.shape, (8, 8, 8, 2, 2))
        # Check that it's complex
        self.assertTrue(np.iscomplexobj(U))

    def test_create_initial_field_invalid(self):
        """Test initial field creation with invalid profile."""
        with self.assertRaises(ValueError):
            self.numerical.create_initial_field("invalid")

    def test_validate_solution(self):
        """Test solution validation."""
        # Create a valid SU(2) field
        U = self.numerical.create_initial_field("tanh")

        result = self.numerical.validate_solution(U)

        # Check result structure
        self.assertIn("su2_valid", result)
        self.assertIn("boundary_conditions", result)


class TestCUDAIntegration(unittest.TestCase):
    """Test CUDA integration."""

    def test_cuda_backend_initialization(self):
        """Test CUDA backend initialization."""
        backend = ArrayBackend()

        # Should not raise an exception
        self.assertIsNotNone(backend)
        self.assertIsInstance(backend.is_cuda_available, bool)

    def test_numerical_methods_with_cuda_backend(self):
        """Test numerical methods with CUDA backend."""
        backend = ArrayBackend()
        numerical = NumericalMethods(grid_size=4, box_size=1.0, backend=backend)

        # Should not raise an exception
        self.assertIsNotNone(numerical)
        self.assertEqual(numerical.backend, backend)

    def test_optimizers_with_cuda_backend(self):
        """Test optimizers with CUDA backend."""
        backend = ArrayBackend()
        config = RelaxationConfig(method=RelaxationMethod.GRADIENT_DESCENT)

        optimizer = GradientDescent(config, backend)

        # Should not raise an exception
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.backend, backend)


if __name__ == "__main__":
    unittest.main()
